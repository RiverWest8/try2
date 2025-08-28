

"""
Temporal Fusion Transformer (TFT) pipeline
==========================================
python3 TFT.py   --max_epochs 10   --batch_size 256   --max_encoder_length 64   --check_val_every_n_epoch 1   --log_every_n_steps 50   --num_workers 12   --prefetch_factor 2   --enable_perm_importance false   --perm_len 288   --fi_max_batches 15   --resume false   --gcs_data_prefix gs://river-ml-bucket/Data/CleanedData   --gcs_output_prefix gs://river-ml-bucket/Dissertation/Feature_Ablation/f_$(date -u +%s)


This script trains a single TFT model that jointly predicts:
  • realised volatility (quantile regression on an asinh‑transformed target)
  • the direction of the next period’s price move (binary classification)

It expects three parquet files:
    ▸ /Users/riverwest-gomila/Desktop/Data/CleanedData/universal_train.parquet
    ▸ /Users/riverwest-gomila/Desktop/Data/CleanedData/universal_val.parquet
    ▸ /Users/riverwest-gomila/Desktop/Data/CleanedData/universal_test.parquet

Required columns (exact names or common aliases will be auto‑detected):
    asset         : categorical asset identifier (aliases: symbol, ticker, instrument)
    Time          : timestamp (parsed to pandas datetime)
    realised_vol  : one‑period realised volatility target
    direction     : 0/1 label for next‑period price direction
plus any engineered numeric features

What the script does:
  1) Loads the parquet splits and standardises column names.
  2) Adds a per‑asset integer `time_idx` required by PyTorch‑Forecasting.
  3) Builds a `TimeSeriesDataSet` with **two targets**: ["realised_vol", "direction"].
     • Target normalisation:
         – realised_vol: `GroupNormalizer(..., transformation="asinh", scale_by_group=True)`
           (applies asinh in the normaliser, per asset)
         – direction: identity (no transform)
     • A per‑asset median `rv_scale` is also attached for **fallback decoding only**
       if a normaliser decode is unavailable.
  4) Fits a TemporalFusionTransformer with a dual head: 7 quantiles for vol and one
     logit for direction, using `LearnableMultiTaskLoss(AsymmetricQuantileLoss, LabelSmoothedBCE)`.
  5) Saves checkpoints, metrics, predictions, and quick feature‑importance summaries.

Adjust hyper‑parameters in the CONFIG section below.
"""

import warnings
warnings.filterwarnings("ignore")
from lightning.pytorch.callbacks import TQDMProgressBar
import os
from pathlib import Path
from typing import List
import json
import numpy as np
import pandas as pd
import pandas as _pd
pd = _pd  # Ensure pd always refers to pandas module
import lightning.pytorch as pl




# Extra belt-and-braces: swallow BrokenPipe errors on stdout.flush() if any other lib calls it.
try:
    import sys
    _orig_flush = sys.stdout.flush
    def _safe_flush(*a, **k):
        try:
            return _orig_flush(*a, **k)
        except BrokenPipeError:
            return None
    sys.stdout.flush = _safe_flush
except Exception:
    pass

# Route all prints to stderr and swallow BrokenPipe to avoid crashes during teardown
import builtins as _builtins, sys as _sys
__orig_print = _builtins.print
def __safe_print(*args, **kwargs):
    try:
        if "file" not in kwargs or kwargs["file"] is None:
            kwargs["file"] = _sys.stderr   # default to stderr
        return __orig_print(*args, **kwargs)
    except BrokenPipeError:
        return None
_builtins = _builtins  # keep a name bound
_builtins.print = __safe_print

from lightning.pytorch import Trainer, seed_everything
from torchmetrics.classification import BinaryAUROC
import torch
import torch.nn as nn
import torch.nn.functional as F
def _permute_series_inplace(df: pd.DataFrame, col: str, block: int, group_col: str = "asset") -> None:
    if col not in df.columns:
        return
    if group_col not in df.columns:
        vals = df[col].values.copy()
        np.random.shuffle(vals)
        df[col] = vals
        return
    for _, idx in df.groupby(group_col, observed=True).groups.items():
        idx = np.asarray(list(idx))
        if block and block > 1:
            shift = np.random.randint(1, max(2, len(idx)))
            df.loc[idx, col] = df.loc[idx, col].values.take(np.arange(len(idx)) - shift, mode='wrap')
        else:
            vals = df.loc[idx, col].values.copy()
            np.random.shuffle(vals)
            df.loc[idx, col] = vals


# helper (put once near the top of the file, or inline if you prefer)
def _first_not_none(d, keys):
    for k in keys:
        v = d.get(k, None)
        if v is not None:
            return v
    return None

# ------------------ Imports ------------------

from pytorch_forecasting import (
    BaseModel,
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)
from pytorch_forecasting.data import MultiNormalizer, TorchNormalizer

from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
# ---- Compatibility stub: safely neutralize TQDMProgressBar ----
try:
    from lightning.pytorch.callbacks import TQDMProgressBar as _RealTQDM
    # If the real class exists but progress bars are globally disabled, alias a no-op wrapper
    class TQDMProgressBar(pl.Callback):
        def __init__(self, *args, **kwargs):
            super().__init__()
        # No methods overridden → progress bar will not be used/initialized
except Exception:
    class TQDMProgressBar(pl.Callback):
        def __init__(self, *args, **kwargs):
            super().__init__()

# --- Cloud / performance helpers ---
import fsspec
try:
    import gcsfs  # ensure GCS protocol is registered with fsspec
except Exception:
    gcsfs = None

# CUDA perf knobs / utils
import shutil
import argparse

import pytorch_forecasting as pf
import inspect

#VOL_QUANTILES = [0.05, 0.165, 0.25, 0.50, 0.75, 0.835, 0.95] #The quantiles which yielded our best so far
#Q50_IDX = VOL_QUANTILES.index(0.50)  
VOL_QUANTILES = [0.05, 0.15, 0.35, 0.50, 0.65, 0.85, 0.95]
Q50_IDX = VOL_QUANTILES.index(0.50)

# Composite metric weights (override via --metric_weights "w_mae,w_rmse,w_qlike")
COMP_WEIGHTS = (1.0, 1.0, 0.333)  # default: slightly emphasise QLIKE for RV focus

def composite_score(mae, rmse, qlike,
                    b_mae=None, b_rmse=None, b_qlike=None,
                    weights=None, eps=1e-12):
    """If baselines are provided → normalised composite (for FI).
       Else → absolute composite (for validation)."""
    w_mae, w_rmse, w_qlike = (weights or COMP_WEIGHTS)
    if b_mae is None or b_rmse is None or b_qlike is None:
        return w_mae*float(mae) + w_rmse*float(rmse) + w_qlike*float(qlike)
    mae_r   = float(mae)   / max(eps, float(b_mae))
    rmse_r  = float(rmse)  / max(eps, float(b_rmse))
    qlike_r = float(qlike) / max(eps, float(b_qlike))
    return w_mae*mae_r + w_rmse*rmse_r + w_qlike*qlike_r
# -----------------------------------------------------------------------
# Ensure a robust "identity" transformation for GroupNormalizer
# -----------------------------------------------------------------------
#
# Different versions of PyTorch‑Forecasting store transformations as a
# dictionary mapping *name* ➜ {"forward": fn, "inverse": fn}.  Some older
# releases omit "identity" entirely, which triggers a KeyError.  Other
# versions include it but as a bare function instead of a dict, which then
# breaks later when `.setdefault()` is called on it.  The logic below
# handles both cases safely.
#

from pytorch_forecasting.data.encoders import GroupNormalizer

# Add a robust log1p mapping as well (for PF versions that lack it or store a bare function)
if ("log1p" not in GroupNormalizer.TRANSFORMATIONS
    or not isinstance(GroupNormalizer.TRANSFORMATIONS["log1p"], dict)):
    GroupNormalizer.TRANSFORMATIONS["log1p"] = {
        "forward": lambda x: torch.log1p(x) if torch.is_tensor(x) else np.log1p(x),
        "inverse": lambda x: torch.expm1(x) if torch.is_tensor(x) else np.expm1(x),
    }
if hasattr(GroupNormalizer, "INVERSE_TRANSFORMATIONS"):
    GroupNormalizer.INVERSE_TRANSFORMATIONS.setdefault(
        "log1p",
        lambda x: torch.expm1(x) if torch.is_tensor(x) else np.expm1(x),
    )

def _extract_norm_from_dataset(ds):
    """
    Return the *volatility* GroupNormalizer used for the first target (realised_vol).
    Works across PF versions where target_normalizer may be:
      - a GroupNormalizer,
      - a MultiNormalizer holding a list in .normalizers / .normalization,
      - a dict mapping target name -> normalizer.
    Preference order: GroupNormalizer for realised_vol, else first normalizer.
    """
    tn = getattr(ds, "target_normalizer", None)
    if tn is None:
        raise ValueError("TimeSeriesDataSet has no target_normalizer")

    # dict mapping target name -> normalizer
    if isinstance(tn, dict):
        for v in tn.values():
            if isinstance(v, GroupNormalizer):
                return v
        return next(iter(tn.values()))

    # MultiNormalizer (PF)
    try:
        from pytorch_forecasting.data import MultiNormalizer as _PFMulti
    except Exception:
        _PFMulti = MultiNormalizer

    if isinstance(tn, _PFMulti):
        norms = getattr(tn, "normalizers", None) or getattr(tn, "normalization", None) or getattr(tn, "_normalizers", None)
        if isinstance(norms, (list, tuple)) and norms:
            for n in norms:
                if isinstance(n, GroupNormalizer):
                    return n
            return norms[0]
        return tn  # unknown container

    # already a single normalizer
    return tn


def _point_from_quantiles(vol_q: torch.Tensor) -> torch.Tensor:
    """
    Enforce non-decreasing quantiles along last dim and return the median (q=0.5).
    Assumes VOL_QUANTILES has q=0.50 at index 3.
    """
    vol_q = torch.cummax(vol_q, dim=-1).values
    return vol_q[..., 3]  # Q50_IDX

#EXTRACT HEADSSSSS
def _extract_heads(pred):
    """
    Return (p_vol, p_dir) as 1D tensors [B] on DEVICE.

    Handles outputs as:
      • list/tuple: [vol(…,K=7), dir(…,K or 1)]
      • tensor [B, 2, K]        (vol quantiles + dir replicated or single)
      • tensor [B, 1, K]        (vol only)
      • tensor [B, K]           (vol only)
      • tensor [B, K+1]         (concatenated: first 7 vol, last 1 dir)
      • tensor [B, 1, K+1]      (concatenated with a singleton middle dim)
    """
    import torch

    K = 7         # len(VOL_QUANTILES)
    MID = 3       # Q50 index when K=7

    def _pick_median_q(vol_q: torch.Tensor) -> torch.Tensor:
        # make sure last dim has the K quantiles
        if vol_q.ndim == 3 and vol_q.size(1) == 1:
            vol_q = vol_q.squeeze(1)            # [B, K]
        if vol_q.ndim == 3 and vol_q.size(-1) == 1 and vol_q.size(1) == K:
            vol_q = vol_q.squeeze(-1)           # [B, K]
        if vol_q.ndim == 2 and vol_q.size(-1) >= K:
            v = vol_q[:, :K]                    # keep exactly 7
        elif vol_q.ndim == 2 and vol_q.size(-1) == K:
            v = vol_q
        else:
            v = vol_q.reshape(vol_q.size(0), -1)[:, :K]
        # enforce monotone and take median
        v = torch.cummax(v, dim=-1).values
        return v[:, MID]                        # [B]

    def _pick_dir_logit(dir_t):
        if dir_t is None:
            return None
        if torch.is_tensor(dir_t) and dir_t.ndim == 3 and dir_t.size(1) == 1:
            dir_t = dir_t.squeeze(1)            # [B, K] or [B, 1]
        if torch.is_tensor(dir_t) and dir_t.ndim == 3 and dir_t.size(-1) == 1:
            dir_t = dir_t.squeeze(-1)           # [B, L]
        if torch.is_tensor(dir_t) and dir_t.ndim == 2:
            # if replicated across K, take middle; if 1, take that; else take last
            L = dir_t.size(-1)
            if L >= K:
                return dir_t[:, MID]
            return dir_t[:, -1]
        if torch.is_tensor(dir_t) and dir_t.ndim == 1:
            return dir_t
        if torch.is_tensor(dir_t):
            return dir_t.reshape(dir_t.size(0), -1)[:, -1]
        return None

    # ---------------- cases ----------------

    # Case A: list/tuple from PF: [vol, dir]
    if isinstance(pred, (list, tuple)):
        vol_q = pred[0]
        dir_t = pred[1] if len(pred) > 1 else None
        if torch.is_tensor(vol_q) and vol_q.ndim == 3 and vol_q.size(1) == 1:
            vol_q = vol_q.squeeze(1)            # [B, K]
        p_vol = _pick_median_q(vol_q) if torch.is_tensor(vol_q) else None
        p_dir = _pick_dir_logit(dir_t)
        return p_vol, p_dir

    if not torch.is_tensor(pred):
        return None, None

    t = pred

    # Squeeze a singleton prediction-length dim if present
    if t.ndim == 4 and t.size(1) == 1:
        t = t.squeeze(1)                         # [B, C, D]
    if t.ndim == 3 and t.size(1) == 1 and t.size(-1) >= 1:
        t = t[:, 0, :]                           # [B, D]

    # Case B: [B, 2, K]  → first = vol quantiles, second = dir (K or 1 replicated)
    if t.ndim == 3 and t.size(1) == 2:
        vol_q = t[:, 0, :]                       # [B, K] (or [B, >=K], we’ll trim)
        dir_t = t[:, 1, :]
        # sometimes dir_t is shape [B, 1] replicated to [B, K]; handle both
        if dir_t.ndim == 2 and dir_t.size(-1) == 1:
            dir_t = dir_t[:, 0]
        p_vol = _pick_median_q(vol_q)
        p_dir = _pick_dir_logit(dir_t)
        return p_vol, p_dir

    # Case C: [B, K, 2]  → last dim=2 (vol, dir)
    if t.ndim == 3 and t.size(-1) == 2 and t.size(1) >= K:
        vol_q = t[:, :K, 0]                      # [B, K]
        dir_t = t[:, :K, 1]                      # [B, K] (or fewer)
        p_vol = _pick_median_q(vol_q)
        p_dir = _pick_dir_logit(dir_t)
        return p_vol, p_dir

    # Case D: [B, K+1] (concat: first K vol, last 1 dir) or [B, K] (vol-only)
    if t.ndim == 2:
        D = t.size(-1)
        if D >= K + 1:
            vol_q = t[:, :K]
            d_log = t[:, K]
            return _pick_median_q(vol_q), d_log
        if D == K:
            return _pick_median_q(t), None
        if D == 1:
            return t.squeeze(-1), None
        # Unknown 2D layout → try to peel first K as vol and last as dir
        vol_q = t[:, :K] if D > K else t
        d_log = t[:, -1] if D > K else None
        return _pick_median_q(vol_q), d_log

    # Fallback: flatten and try our best
    if t.ndim >= 1:
        flat = t.reshape(t.size(0), -1)
        vol_q = flat[:, :K]
        d_log = flat[:, K] if flat.size(-1) > K else None
        return _pick_median_q(vol_q), d_log

    return None, None


    


# -----------------------------------------------------------------------
# Robust manual inverse for GroupNormalizer (fallback when decode fails)
# -----------------------------------------------------------------------
@torch.no_grad()
def manual_inverse_transform_groupnorm(normalizer, y: torch.Tensor, group_ids: torch.Tensor | None):
    """
    Reverse GroupNormalizer:
      1) destandardize: y -> y*scale + center (center may be None when center=False)
      2) inverse transform: asinh -> sinh, identity -> no-op, else via registry
    Works with scale_by_group=True (index by group_ids) or global scale.
    """
    y_ = y.squeeze(-1) if y.ndim > 1 else y

    center = getattr(normalizer, "center", None)
    scale  = getattr(normalizer, "scale",  None)

    if scale is None:
        x = y_
    else:
        if group_ids is not None and torch.is_tensor(group_ids):
            g = group_ids
            if g.ndim > 1 and g.size(-1) == 1:
                g = g.squeeze(-1)
            g = g.long()
            s = scale[g] if isinstance(scale, torch.Tensor) else scale
            c = center[g] if isinstance(center, torch.Tensor) else 0.0
        else:
            s = scale
            c = center if isinstance(center, torch.Tensor) else 0.0
        x = y_ * s + c

    tfm = getattr(normalizer, "transformation", None)
    if tfm == "log1p":
        x = torch.expm1(x)
    elif tfm in (None, "identity"):
        pass
    else:
        try:
            inv = type(normalizer).TRANSFORMATIONS[tfm]["inverse"]
            x = inv(x)
        except Exception:
            pass

    return x.view_as(y)

@torch.no_grad()
def safe_decode_vol(y: torch.Tensor, normalizer, group_ids: torch.Tensor | None):
    """
    Try normalizer.decode(), then inverse_transform(), else manual fallback.
    y is expected as [B,1].
    """
    try:
        return normalizer.decode(y, group_ids=group_ids)
    except Exception:
        pass
    try:
        return normalizer.inverse_transform(y, group_ids=group_ids)
    except Exception:
        try:
            return normalizer.inverse_transform(y)
        except Exception:
            pass
    return manual_inverse_transform_groupnorm(normalizer, y, group_ids)


# ---------------- Regime-dependent calibration helper ----------------
@torch.no_grad()
def calibrate_vol_predictions(y_true_dec: torch.Tensor, y_pred_dec: torch.Tensor) -> torch.Tensor:
    """
    Simple 2‑regime multiplicative calibration computed on validation targets.
    We match the mean in the bottom and top terciles separately to stretch tails
    (helps when predictions are cramped at the low end).
    """
    # ensure 1D
    y = y_true_dec.reshape(-1)
    p = y_pred_dec.reshape(-1).clone()
    if y.numel() == 0 or p.numel() == 0:
        return y_pred_dec

    # compute terciles
    q33, q66 = torch.quantile(y, torch.tensor([0.33, 0.66], device=y.device))
    low_mask = y <= q33
    high_mask = y >= q66

    def _apply(mask: torch.Tensor):
        if mask is None or mask.sum() == 0:
            return
        yp = p[mask].mean()
        yt = y[mask].mean()
        if torch.isfinite(yp) and torch.isfinite(yt) and float(torch.abs(yp)) > 1e-12:
            s = (yt / yp).clamp(0.5, 2.0)
            p[mask] = p[mask] * s

    _apply(low_mask)
    _apply(high_mask)

    return p.view_as(y_pred_dec)

if not hasattr(GroupNormalizer, "decode"):
    def _gn_decode(self, y, group_ids=None, **kwargs):
        """
        Alias for `inverse_transform` to keep newer *and* older
        PyTorch‑Forecasting versions compatible with the same call‑site.

        The underlying `inverse_transform` API has changed a few times:
        ▸ Newer versions accept ``group_ids=`` keyword
        ▸ Some legacy variants want ``X=`` instead
        ▸ Very old releases implement the method but raise ``NotImplementedError``  
          (it was a placeholder).

        The cascading fall‑backs below try each signature in turn and, as a
        last resort, simply return the *input* unchanged so downstream code
        can continue without crashing.
        """
        try:
            # 1️⃣  Modern signature (>=0.10): accepts ``group_ids=``
            return self.inverse_transform(y, group_ids=group_ids, **kwargs)
        except (TypeError, NotImplementedError):
            try:
                # 2️⃣  Mid‑vintage signature: expects ``X=None`` instead
                return self.inverse_transform(y, X=None, **kwargs)
            except (TypeError, NotImplementedError):
                try:
                    # 3️⃣  Very early signature: just (y) positional
                    return self.inverse_transform(y)
                except (TypeError, NotImplementedError):
                    # 4️⃣  Ultimate fall‑back – give up on denorm, return y
                    return y

    GroupNormalizer.decode = _gn_decode



from pytorch_forecasting.metrics import QuantileLoss, MultiLoss

class LabelSmoothedBCE(nn.Module):
    def __init__(self, smoothing: float = 0.1, pos_weight: float = 1.0):
        super().__init__()
        self.smoothing = smoothing
        self.register_buffer("pos_weight", torch.tensor(pos_weight))

    def forward(self, y_pred, target):
        target = target.float()
        target = target * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(
            y_pred.squeeze(-1), target.squeeze(-1),
            pos_weight=self.pos_weight,
        )

class LabelSmoothedBCEWithBrier(nn.Module):
    """
    Direction loss = label-smoothed BCE + lambda * Brier score.
    Brier uses probabilities (sigmoid over logits) and the SAME smoothed
    targets as the BCE term for consistency. Set brier_weight=0 to
    recover plain label-smoothed BCE.
    """
    def __init__(self, smoothing: float = 0.1, pos_weight: float = 1.001, brier_weight: float = 0.18):
        super().__init__()
        self.smoothing = float(smoothing)
        self.register_buffer("pos_weight", torch.tensor(pos_weight))
        self.brier_weight = float(brier_weight)

    def forward(self, y_pred, target):
        target = target.float()
        # label smoothing as in LabelSmoothedBCE
        target_smooth = target * (1.0 - self.smoothing) + 0.5 * self.smoothing

        # BCE on logits
        bce = F.binary_cross_entropy_with_logits(
            y_pred.squeeze(-1), target_smooth.squeeze(-1),
            pos_weight=self.pos_weight,
        )

        if self.brier_weight <= 0.0:
            return bce

        # Brier on probabilities
        probs = torch.sigmoid(y_pred.squeeze(-1))
        brier = ((probs - target_smooth.squeeze(-1)) ** 2).mean()

        return bce + self.brier_weight * brier

import torch
import torch.nn.functional as F


class AsymmetricQuantileLoss(QuantileLoss):
    """
    Quantile loss with:
      • asymmetric underestimation penalty on all quantiles
      • optional mean-bias regulariser on the median (q=0.5)
      • optional tail emphasis (up-weights samples above a quantile threshold)
      • numerically-stable QLIKE term on a scale-free decoded space via log1p()

    Notes
    -----
    - Use with GroupNormalizer(transformation="log1p", center=False, scale_by_group=True)
      so that the y/ŷ ratio is scale-free across assets.
    - QLIKE is stabilized by:
        * clipping encoded values before log1p (med_clip)
        * working in log-variance space with clamped log-ratio (log_ratio_clip)
        * epsilon floors
    """

    def __init__(
        self,
        quantiles,
        underestimation_factor: float = 1.0, #1.1115
        mean_bias_weight: float = 0.0,
        tail_q: float = 0.90,         # ← was 0.85
        tail_weight: float = 0.0,
        qlike_weight: float = 0.0,   # set to 0.0 because we cannot safely decode inside the loss
        eps: float = 1e-8,
        med_clip: float = 3.0,
        log_ratio_clip: float = 12.0,
        **kwargs,
    ):
        super().__init__(quantiles=quantiles, **kwargs)
        self.underestimation_factor = float(underestimation_factor)
        self.mean_bias_weight = float(mean_bias_weight)
        self.tail_q = float(tail_q)
        self.tail_weight = float(tail_weight)
        self.qlike_weight = float(qlike_weight)
        self.eps = float(eps)
        self.med_clip = float(med_clip)
        self.log_ratio_clip = float(log_ratio_clip)

        try:
            self._q50_idx = self.quantiles.index(0.5)
        except Exception:
            self._q50_idx = len(self.quantiles) // 2

    # ----- core quantile component (with optional tail up-weighting) -----
    def loss_per_prediction(self, y_pred, target):
        if isinstance(target, tuple):
            target = target[0]

        diff = target.unsqueeze(-1) - y_pred
        q = y_pred.new_tensor(self.quantiles).view(*([1] * (diff.ndim - 1)), -1)
        alpha = y_pred.new_tensor(self.underestimation_factor)

        loss = torch.where(
            diff >= 0,
            alpha * q * diff,             # under-prediction → amplified
            (1.0 - q) * (-diff),          # over-prediction
        )

        if self.tail_weight and self.tail_weight > 0:
            try:
                thresh = torch.quantile(target.detach(), self.tail_q)
                w = torch.where(target.unsqueeze(-1) > thresh,
                                 1.0 + (self.tail_weight - 1.0),
                                 1.0)
                loss = loss * w
            except Exception:
                pass

        return loss

    def forward(self, y_pred, target):
        base = self.loss_per_prediction(y_pred, target).mean()

        # ---- mean-bias regulariser on the median (q=0.5) ----
        if self.mean_bias_weight > 0:
            try:
                if isinstance(target, tuple):
                    target = target[0]
                med = y_pred[..., self._q50_idx]
                mean_diff = (target - med).mean()
                base = base + (mean_diff ** 2) * self.mean_bias_weight
            except Exception:
                pass

            # ---- Optional surrogate QLIKE (encoded scale; no decode) ----
            # If qlike_weight > 0, add a stable proxy using encoded magnitudes.
            # We approximate sigma by |·| of the encoded target/median prediction.
        if self.qlike_weight and self.qlike_weight > 0:
            try:
                if isinstance(target, tuple):
                    target = target[0]
                # median (q=0.5) prediction on encoded scale
                med = y_pred[..., self._q50_idx]

                # positive "scale" proxies from encoded values
                # clamp to avoid extreme gradients, then floor by eps
                sigma_y = torch.clamp(target.abs(), min=self.eps)
                sigma_p = torch.clamp(med.abs(),    min=self.eps)

                # QLIKE core on scale ratio
                ratio = (sigma_y ** 2) / (sigma_p ** 2)
                # clip log-ratio for numerical stability
                log_ratio = torch.log(torch.clamp(ratio, min=torch.exp(-self.log_ratio_clip), max=torch.exp(self.log_ratio_clip)))
                qlike = ratio - log_ratio - 1.0

                base = base + float(self.qlike_weight) * qlike.mean()
            except Exception:
                # if anything goes sideways, just fall back to quantile-only
                pass

        return torch.nan_to_num(base, nan=0.0, posinf=1e4, neginf=1e4)



class PerAssetMetrics(pl.Callback):
    """Collects per-asset predictions during validation and prints/saves metrics.
    Computes MAE, RMSE, MSE, QLIKE for realised_vol and Accuracy for direction.
    """
    def __init__(self, id_to_name: dict, vol_normalizer, max_print: int = 10):
        super().__init__()
        self.id_to_name = {int(k): str(v) for k, v in id_to_name.items()}
        self.vol_norm = vol_normalizer
        self.max_print = max_print
        self.reset()

    def reset(self):
        # device-resident accumulators (concatenate at epoch end)
        self._g_dev = []   # group ids per sample (device, flattened)
        self._yv_dev = []  # realised vol target (NORMALISED, device)
        self._pv_dev = []  # realised vol pred   (NORMALISED, device)
        self._yd_dev = []  # direction target (device)
        self._pd_dev = []  # direction pred logits/probs (device)
        self._t_dev = []   # decoder time_idx (device) if provided
        # cached final rows/overall from last epoch
        self._last_rows = None
        self._last_overall = None

    @torch.no_grad()
    def on_validation_epoch_start(self, trainer, pl_module):
        self.reset()
        print(f"[VAL HOOK] start epoch {getattr(trainer,'current_epoch',-1)+1}")

    @torch.no_grad()
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0):
        # batch is (x, y, weight) from PF dataloader
        if not isinstance(batch, (list, tuple)):
            return
        x = batch[0]
        if not isinstance(x, dict):
            return
        groups = None
        for k in ("groups", "group_ids", "group_id"):
            if k in x and x[k] is not None:
                groups = x[k]
                break
        dec_t = x.get("decoder_target")

        # optional time index for plotting/joining later
        dec_time = x.get("decoder_time_idx", None)
        if dec_time is None:
            # some PF versions may expose relative index or time via different keys; try a few
            dec_time = x.get("decoder_relative_idx", None)

        # also fetch explicit targets from batch[1] as a fallback
        y_batch = batch[1] if isinstance(batch, (list, tuple)) and len(batch) >= 2 else None
        if groups is None:
            return

        # groups can be a Tensor or a list[Tensor]; take the first if list
        groups_raw = groups[0] if isinstance(groups, (list, tuple)) else groups
        g = groups_raw
        # squeeze trailing singleton dims to get [B]
        while torch.is_tensor(g) and g.ndim > 1 and g.size(-1) == 1:
            g = g.squeeze(-1)
        if not torch.is_tensor(g):
            return

        # --- Extract targets (try decoder_target first, else fall back to batch[1]) ---
        y_vol_t, y_dir_t = None, None
        if dec_t is not None:
            if torch.is_tensor(dec_t):
                y = dec_t
                if y.ndim == 3 and y.size(-1) == 1:
                    y = y[..., 0]  # → [B, n_targets]
                if y.ndim == 2 and y.size(1) >= 1:
                    y_vol_t = y[:, 0]
                    if y.size(1) > 1:
                        y_dir_t = y[:, 1]
            elif isinstance(dec_t, (list, tuple)) and len(dec_t) >= 1:
                y_vol_t = dec_t[0]
                if torch.is_tensor(y_vol_t):
                    if y_vol_t.ndim == 3 and y_vol_t.size(-1) == 1:
                        y_vol_t = y_vol_t[..., 0]
                    if y_vol_t.ndim == 2 and y_vol_t.size(-1) == 1:
                        y_vol_t = y_vol_t[:, 0]
                if len(dec_t) > 1 and torch.is_tensor(dec_t[1]):
                    y_dir_t = dec_t[1]
                    if y_dir_t.ndim == 3 and y_dir_t.size(-1) == 1:
                        y_dir_t = y_dir_t[..., 0]
                    if y_dir_t.ndim == 2 and y_dir_t.size(-1) == 1:
                        y_dir_t = y_dir_t[:, 0]

        # Fallback: PF sometimes provides targets in batch[1] as [B, pred_len, n_targets]
        if (y_vol_t is None or y_dir_t is None) and torch.is_tensor(y_batch):
            yb = y_batch
            if yb.ndim == 3 and yb.size(1) == 1:
                yb = yb[:, 0, :]
            if yb.ndim == 2 and yb.size(1) >= 1:
                if y_vol_t is None:
                    y_vol_t = yb[:, 0]
                if y_dir_t is None and yb.size(1) > 1:
                    y_dir_t = yb[:, 1]

        if y_vol_t is None:
            return

        # Forward pass to get predictions for this batch
        y_hat = pl_module(x)
        pred = getattr(y_hat, "prediction", y_hat)
        if isinstance(pred, dict) and "prediction" in pred:
            pred = pred["prediction"]

        # --- NEW: use shared head extractor ---
        p_vol, p_dir = _extract_heads(pred)
        if p_vol is None:
            return  # nothing usable in this batch

        # Store device tensors; no decode/CPU here
        L = g.shape[0]
        self._g_dev.append(g.reshape(L))
        self._yv_dev.append(y_vol_t.reshape(L))
        self._pv_dev.append(p_vol.reshape(L))

        # capture time index if available and shape-compatible
        if dec_time is not None and torch.is_tensor(dec_time):
            tvec = dec_time
            # squeeze to [B]
            while tvec.ndim > 1 and tvec.size(-1) == 1:
                tvec = tvec.squeeze(-1)
            if tvec.numel() >= L:
                self._t_dev.append(tvec.reshape(-1)[:L])

        if y_dir_t is not None and p_dir is not None:
            y_flat = y_dir_t.reshape(-1)
            p_flat = p_dir.reshape(-1)
            L2 = min(L, y_flat.numel(), p_flat.numel())
            if L2 > 0:
                self._yd_dev.append(y_flat[:L2])
                self._pd_dev.append(p_flat[:L2])

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        # If nothing collected, exit quietly
        if not self._g_dev:
            return

        # Gather device tensors accumulated during validation
        device = self._g_dev[0].device
        g  = torch.cat(self._g_dev).to(device)
        yv = torch.cat(self._yv_dev).to(device)            # realised_vol (encoded)
        pv = torch.cat(self._pv_dev).to(device)            # realised_vol pred (encoded)
        yd = torch.cat(self._yd_dev).to(device) if self._yd_dev else None  # direction labels
        pd = torch.cat(self._pd_dev).to(device) if self._pd_dev else None  # direction logits/probs

        # --- Decode realised_vol to physical scale (robust to PF version)
        yv_dec = safe_decode_vol(yv.unsqueeze(-1), self.vol_norm, g.unsqueeze(-1)).squeeze(-1)
        pv_dec = safe_decode_vol(pv.unsqueeze(-1), self.vol_norm, g.unsqueeze(-1)).squeeze(-1)

        # Guard against near-zero decoded vols (use global floor if defined)
        floor_val = globals().get("EVAL_VOL_FLOOR", 1e-8)
        yv_dec = torch.clamp(yv_dec, min=floor_val)
        pv_dec = torch.clamp(pv_dec, min=floor_val)


        # Debug prints
        print("DEBUG transformation:", getattr(self.vol_norm, "transformation", None))
        print("DEBUG mean after decode:", yv_dec.mean().item())
        print(
            "DEBUG: mean(yv_dec)=",
            yv_dec.mean().item(),
            "mean(pv_dec)=",
            pv_dec.mean().item(),
            "ratio=",
            (yv_dec.mean() / pv_dec.mean()).item() if float(pv_dec.mean()) != 0.0 else float("inf"),
        )

        # --- Regime-dependent calibration (stretches tails) ---

        # Move to CPU for metric computation
        y_cpu  = yv_dec.detach().cpu()
        p_cpu  = pv_dec.detach().cpu()
        yd_cpu = yd.detach().cpu() if yd is not None else None
        pd_cpu = pd.detach().cpu() if pd is not None else None

        # Calibration diagnostic (not used in loss)
        try:
            mean_scale = float((y_cpu.mean() / (p_cpu.mean() + 1e-12)).item())
        except Exception:
            mean_scale = float("nan")
        print(f"[CAL DEBUG] mean(y)/mean(p)={mean_scale:.4f} (1==perfect)")
        try:
            trainer.callback_metrics["val_mean_scale"] = torch.tensor(mean_scale)
        except Exception:
            pass

        # --- Decoded regression metrics ---
        eps = 1e-8
        overall_mae  = (p_cpu - y_cpu).abs().mean().item()
        overall_mse  = ((p_cpu - y_cpu) ** 2).mean().item()
        overall_rmse = overall_mse ** 0.5

        sigma2_p = torch.clamp(p_cpu.abs(), min=eps) ** 2
        sigma2_y = torch.clamp(y_cpu.abs(), min=eps) ** 2
        ratio    = sigma2_y / sigma2_p
        overall_qlike = (ratio - torch.log(ratio) - 1.0).mean().item()

        # --- Optional direction accuracy ---
        acc = None
        if yd_cpu is not None and pd_cpu is not None and yd_cpu.numel() > 0 and pd_cpu.numel() > 0:
            pt = pd_cpu
            try:
                if torch.isfinite(pt).any() and (pt.min() < 0 or pt.max() > 1):
                    pt = torch.sigmoid(pt)  # logits → probs
            except Exception:
                pt = torch.sigmoid(pt)
            pt = torch.clamp(pt, 0.0, 1.0)
            acc = ((pt >= 0.5).int() == yd_cpu.int()).float().mean().item()

           
        brier = None
        if yd_cpu is not None and pd_cpu is not None and yd_cpu.numel() > 0 and pd_cpu.numel() > 0:
            pt_b = pd_cpu
            try:
                if torch.isfinite(pt_b).any() and (pt_b.min() < 0 or pt_b.max() > 1):
                    pt_b = torch.sigmoid(pt_b)  # logits → probs
            except Exception:
                pt_b = torch.sigmoid(pt_b)
            pt_b = torch.clamp(pt_b, 0.0, 1.0)
            brier = ((pt_b - yd_cpu.float()) ** 2).mean().item()
            trainer.callback_metrics["val_brier_overall"] = torch.tensor(float(brier))

        auroc = None
        if yd_cpu is not None and pd_cpu is not None and yd_cpu.numel() > 0 and pd_cpu.numel() > 0:
            probs = pd_cpu
            try:
                if torch.isfinite(probs).any() and (probs.min() < 0 or probs.max() > 1):
                    probs = torch.sigmoid(probs)  # logits → probs
            except Exception:
                probs = torch.sigmoid(probs)
            probs = torch.clamp(probs, 0.0, 1.0)

            try:
                au = BinaryAUROC()
                auroc = float(au(probs.detach().cpu(), yd_cpu.detach().cpu()).item())
                trainer.callback_metrics["val_auroc_overall"] = torch.tensor(float(auroc))
            except Exception as e:
                print(f"[WARN] AUROC failed: {e}")

        # --- Epoch summary ---
        N = int(y_cpu.numel())
        try:
            epoch_num = int(getattr(trainer, "current_epoch", -1)) + 1
        except Exception:
            epoch_num = None

        # 🔑 Use QLIKE as the validation loss
        # 🔑 Use composite(MAE,RMSE,QLIKE) as the validation loss (absolute form)
        val_composite  = composite_score(overall_mae, overall_rmse, overall_qlike)
        val_loss_value = float(val_composite)
        trainer.callback_metrics["val_composite_overall"] = torch.tensor(float(val_composite))
        trainer.callback_metrics["val_loss_source"] = "composite(MAE,RMSE,QLIKE)"

        msg = (
            f"[VAL EPOCH {epoch_num}] "
            f"(decoded) MAE={overall_mae:.6f} "
            f"RMSE={overall_rmse:.6f} "
            f"MSE={overall_mse:.6f} "
            f"QLIKE={overall_qlike:.6f} "
            f"CompLoss = {val_composite:.6f}"
            + (f"| ACC={acc:.3f} " if acc is not None else "")
            + (f"| Brier={brier:.4f} " if brier is not None else "")
            + (f"| AUROC={auroc:.3f} " if auroc is not None else "")
            + f"| N={N} | VAL_LOSS(comp)={val_loss_value:.6f}"
        )
        print(msg)

        # --- Log metrics for checkpoints/early stopping ---
        trainer.callback_metrics["val_mae_overall"]   = torch.tensor(float(overall_mae))
        trainer.callback_metrics["val_rmse_overall"]  = torch.tensor(float(overall_rmse))
        trainer.callback_metrics["val_mse_overall"]   = torch.tensor(float(overall_mse))
        trainer.callback_metrics["val_qlike_overall"] = torch.tensor(float(overall_qlike))
        trainer.callback_metrics["val_loss"]          = torch.tensor(float(val_loss_value))
        trainer.callback_metrics["val_loss_decoded"]  = torch.tensor(float(val_loss_value))
        if acc is not None:
            trainer.callback_metrics["val_acc_overall"] = torch.tensor(float(acc))
        if auroc is not None:  # 👈 ADD HERE
            trainer.callback_metrics["val_auroc_overall"] = torch.tensor(float(auroc))
        trainer.callback_metrics["val_N_overall"]     = torch.tensor(float(N))

        self._last_overall = {
            "mae": overall_mae,
            "rmse": overall_rmse,
            "mse": overall_mse,
            "qlike": overall_qlike,
            "val_loss": val_loss_value,
            "dir_bce": brier,
        }
        self._last_rows = []


    @torch.no_grad()
    def on_fit_end(self, trainer, pl_module):
        if self._last_rows is None or self._last_overall is None:
            return
        rows = self._last_rows
        overall = self._last_overall
        print("\nOverall decoded metrics (final):")
        print(f"MAE: {overall['mae']:.6f} | RMSE: {overall['rmse']:.6f} | MSE: {overall['mse']:.6f} | QLIKE: {overall['qlike']:.6f}")

        print("\nPer-asset validation metrics (top by samples):")
        print("asset | MAE | RMSE | MSE | QLIKE | ACC | N")
        for r in rows[: self.max_print]:
            acc_str = "-" if r[5] is None else f"{r[5]:.3f}"
            print(f"{r[0]} | {r[1]:.6f} | {r[2]:.6f} | {r[3]:.6f} | {r[4]:.6f} | {acc_str} | {r[6]}")

        dir_overall = None
        yd = overall.get("yd", None)
        pd_all = overall.get("pd", None)
        if yd is not None and pd_all is not None and yd.numel() > 0 and pd_all.numel() > 0:
            try:
                L = min(yd.numel(), pd_all.numel())
                yd1 = yd[:L].float()
                pd1 = pd_all[:L]
                try:
                    if torch.isfinite(pd1).any() and (pd1.min() < 0 or pd1.max() > 1):
                        pd1 = torch.sigmoid(pd1)
                except Exception:
                    pd1 = torch.sigmoid(pd1)
                pd1 = torch.clamp(pd1, 0.0, 1.0)
                acc = ((pd1 >= 0.5).int() == yd1.int()).float().mean().item()
                brier = ((pd1 - yd1) ** 2).mean().item()
                auroc = None
                try:
                    from torchmetrics.classification import BinaryAUROC
                    au = BinaryAUROC()
                    auroc = float(au(pd1.detach().cpu(), yd1.detach().cpu()).item())
                except Exception:
                    auroc = None
                dir_overall = {"accuracy": acc, "brier": brier, "auroc": auroc}
                print("\nDirection (final):")
                msg = f"Accuracy: {acc:.3f} | Brier: {brier:.4f}"
                if auroc is not None:
                    msg += f" | AUROC: {auroc:.3f}"
                print(msg)
            except Exception as e:
                print(f"[WARN] Could not compute final direction metrics: {e}")

        try:
            import json
            out = {
                "decoded": True,
                "overall": {k: v for k, v in overall.items() if k in ("mae","rmse","mse","qlike","val_loss","dir_bce")},
                "direction_overall": dir_overall,
                "per_asset": [
                    {"asset": r[0], "mae": r[1], "rmse": r[2], "mse": r[3], "qlike": r[4], "acc": r[5], "n": r[6]}
                    for r in rows
                ],
            }
            path = str(LOCAL_RUN_DIR / f"tft_val_asset_metrics_e{MAX_EPOCHS}_{RUN_SUFFIX}.json")
            with open(path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"✓ Saved per-asset validation metrics (decoded, final) → {path}")
        except Exception as e:
            print(f"[WARN] Could not save final per-asset metrics: {e}")

        # Optionally save per-sample predictions for plotting
        try:
            import pandas as pd  # ensure pd is bound locally and not shadowed
            # Recompute decoded tensors from the stored device buffers
            g_cpu = torch.cat(self._g_dev).detach().cpu() if self._g_dev else None
            yv_cpu = torch.cat(self._yv_dev).detach().cpu() if self._yv_dev else None
            pv_cpu = torch.cat(self._pv_dev).detach().cpu() if self._pv_dev else None
            yd_cpu = torch.cat(self._yd_dev).detach().cpu() if self._yd_dev else None
            pd_cpu = torch.cat(self._pd_dev).detach().cpu() if self._pd_dev else None
            # decode vol back to physical scale
            if g_cpu is not None and yv_cpu is not None and pv_cpu is not None:
                yv_dec = self.vol_norm.decode(yv_cpu.unsqueeze(-1), group_ids=g_cpu.unsqueeze(-1)).squeeze(-1)
                pv_dec = self.vol_norm.decode(pv_cpu.unsqueeze(-1), group_ids=g_cpu.unsqueeze(-1)).squeeze(-1)
                # Apply the same calibration used in metrics so saved preds match the plots
                #pv_dec = calibrate_vol_predictions(yv_dec, pv_dec)
                # map group id -> name
                assets = [self.id_to_name.get(int(i), str(int(i))) for i in g_cpu.tolist()]
                # time index (may be missing)
                t_cpu = torch.cat(self._t_dev).detach().cpu() if self._t_dev else None
                df_out = pd.DataFrame({
                    "asset": assets,
                    "time_idx": t_cpu.numpy().tolist() if t_cpu is not None else [None] * len(assets),
                    "y_vol": yv_dec.numpy().tolist(),
                    "y_vol_pred": pv_dec.numpy().tolist(),
                })
                if yd_cpu is not None and pd_cpu is not None and yd_cpu.numel() > 0 and pd_cpu.numel() > 0:
                    # ensure pd is probability
                    pdp = pd_cpu
                    try:
                        if torch.isfinite(pdp).any() and (pdp.min() < 0 or pdp.max() > 1):
                            pdp = torch.sigmoid(pdp)
                    except Exception:
                        pdp = torch.sigmoid(pdp)
                    pdp = torch.clamp(pdp, 0.0, 1.0)
                    Lm = min(len(df_out), yd_cpu.numel(), pdp.numel())
                    df_out = df_out.iloc[:Lm].copy()
                    df_out["y_dir"] = yd_cpu[:Lm].numpy().tolist()
                    df_out["y_dir_prob"] = pdp[:Lm].numpy().tolist()
                # --- Attach real timestamps from a source DataFrame, if available ---
                try:
                    # Look for a likely source dataframe that contains ['asset','time_idx','Time']
                    candidate_names = ["val_df"]
                    src_df = None
                    for _name in candidate_names:
                        obj = globals().get(_name)
                        if isinstance(obj, pd.DataFrame) and {"asset", "time_idx", "Time"}.issubset(obj.columns):
                            src_df = obj[["asset", "time_idx", "Time"]].copy()
                            break
                    if src_df is not None:
                        # Harmonise dtypes prior to merge
                        src_df["asset"] = src_df["asset"].astype(str)
                        src_df["time_idx"] = pd.to_numeric(src_df["time_idx"], errors="coerce").astype("Int64").astype("int64")
                        df_out["asset"] = df_out["asset"].astype(str)
                        df_out["time_idx"] = pd.to_numeric(df_out["time_idx"], errors="coerce").astype("Int64").astype("int64")

                        # Merge on ['asset','time_idx'] to bring in the real Time column
                        df_out = df_out.merge(src_df, on=["asset", "time_idx"], how="left", validate="m:1")

                        # Coerce Time to timezone-naive pandas datetimes
                        df_out["Time"] = pd.to_datetime(df_out["Time"], errors="coerce")
                        try:
                            # If tz-aware, drop timezone info; if already naive this may raise — ignore
                            df_out["Time"] = df_out["Time"].dt.tz_localize(None)
                        except Exception:
                            pass
                    else:
                        print(
                            "[WARN] Could not locate a source dataframe with ['asset','time_idx','Time'] among candidates: "
                            "raw_df, raw_data, full_df, df (also checked val_df/train_df/test_df)."
                        )
                except Exception as e:
                    print(f"[WARN] Failed to attach real timestamps: {e}")
                pred_path = LOCAL_OUTPUT_DIR / f"tft_val_predictions_e{MAX_EPOCHS}_{RUN_SUFFIX}.parquet"
                df_out.to_parquet(pred_path, index=False)
                print(f"✓ Saved validation predictions (Parquet) → {pred_path}")
                try:
                    upload_file_to_gcs(str(pred_path), f"{GCS_OUTPUT_PREFIX}/{pred_path.name}")
                except Exception as e:
                    print(f"[WARN] Could not upload validation predictions: {e}")
        except Exception as e:
            print(f"[WARN] Could not save validation predictions: {e}")

import lightning.pytorch as pl

class BiasWarmupCallback(pl.Callback):
    """
    Adaptive warm-up with a safety guard.

    • EMA of mean(y)/mean(p) steers underestimation bias (alpha).
    • Warm-ups are frozen if validation worsens for `guard_patience` epochs.
    • QLIKE ramps only when scale is near 1 to avoid fighting calibration.
    """
    def __init__(
        self,
        vol_loss=None,
        target_under: float = 1.12,
        target_mean_bias: float = 0.04,
        warmup_epochs: int = 4,
        qlike_target_weight: float | None = 0.04,
        start_mean_bias: float = 0.0,
        mean_bias_ramp_until: int = 8,
        guard_patience: int = 2,
        guard_tol: float = 0.0,
        alpha_step: float = 0.05,
    ):
        super().__init__()
        self._vol_loss_hint = vol_loss
        self.target_under = float(target_under)
        self.target_mean_bias = float(target_mean_bias)
        self.qlike_target_weight = None if qlike_target_weight is None else float(qlike_target_weight)
        self.warm = int(max(0, warmup_epochs))
        self.start_mean_bias = float(start_mean_bias)
        self.mean_bias_ramp_until = int(max(mean_bias_ramp_until, self.warm))
        self.guard_patience = int(max(1, guard_patience))
        self.guard_tol = float(guard_tol)
        self.alpha_step = float(alpha_step)

        self._scale_ema = None
        self._prev_val = None
        self._worse_streak = 0
        self._frozen = False

    def _resolve_vol_loss(self, pl_module):
        import inspect
        def is_vol_loss(obj):
            if obj is None:
                return False
            name = obj.__class__.__name__
            return hasattr(obj, "loss_per_prediction") and hasattr(obj, "quantiles") and (
                name == "AsymmetricQuantileLoss" or hasattr(obj, "qlike_weight")
            )
        cand = self._vol_loss_hint
        if is_vol_loss(cand):
            return cand
        for _, v in inspect.getmembers(pl_module):
            if is_vol_loss(v):
                return v
        return None

    def on_validation_end(self, trainer, pl_module):
        val = trainer.callback_metrics.get("val_loss") or trainer.callback_metrics.get("val_qlike_overall")
        try:
            val = float(val.item() if hasattr(val, "item") else val)
        except Exception:
            val = None
        if val is None:
            return
        if self._prev_val is not None and val > self._prev_val + self.guard_tol:
            self._worse_streak += 1
        else:
            self._worse_streak = 0
        self._prev_val = val
        if self._worse_streak >= self.guard_patience:
            self._frozen = True
        if self._worse_streak == 0:  # improved or equal
            self._frozen = False

    def on_train_epoch_start(self, trainer, pl_module):
        # Global CLI switch: if --disable_warmups is true, do nothing
        try:
            if globals().get("ARGS") is not None and getattr(ARGS, "disable_warmups", False):
                print(f"[BIAS] epoch={int(getattr(trainer,'current_epoch',0))} DISABLED via --disable_warmups; skipping")
                return
        except Exception:
            pass
        vol_loss = self._resolve_vol_loss(pl_module)
        if vol_loss is None:
            print("[BIAS] could not resolve vol loss; skipping warm-up tweaks")
            return

        # read last validation mean scale diagnostic
        scale = None
        try:
            val = trainer.callback_metrics.get("val_mean_scale")
            if val is not None:
                scale = float(val.item() if hasattr(val, "item") else val)
        except Exception:
            pass

        if scale is not None:
            self._scale_ema = scale if self._scale_ema is None else 0.8 * self._scale_ema + 0.2 * scale

        e = int(getattr(trainer, "current_epoch", 0))
        prog = min(1.0, float(e) / float(max(self.warm, 1)))

        # Freeze if getting worse
        if self._frozen:
            vol_loss.underestimation_factor = 1.0
            if hasattr(vol_loss, "qlike_weight"):
                vol_loss.qlike_weight = 0.0
            vol_loss.mean_bias_weight = min(getattr(vol_loss, "mean_bias_weight", 0.0), self.target_mean_bias)
            print(f"[BIAS] epoch={e} FROZEN: alpha=1.0 qlike_w=0.0 mean_bias={vol_loss.mean_bias_weight:.3f}")
            return

        # proportional (small) adjustment to alpha to avoid overshoot
        alpha = 1.0
        if self._scale_ema is not None:
            err = np.log(max(1e-6, self._scale_ema))   # log mean(y)/mean(p)
            step = np.clip(1.0 + self.alpha_step * err, 1.0 - self.alpha_step, 1.0 + self.alpha_step)
            alpha = (1.0 + (self.target_under - 1.0) * prog) * step
            if self._scale_ema < 0.995:  # already over-predicting
                alpha = min(alpha, 1.0)
        vol_loss.underestimation_factor = float(max(0.9, min(alpha, self.target_under)))

        # mean-bias gentle ramp
        if e <= self.mean_bias_ramp_until:
            mb_prog = min(1.0, max(0.0, e / float(max(1, self.mean_bias_ramp_until))))
            vol_loss.mean_bias_weight = self.start_mean_bias + (self.target_mean_bias - self.start_mean_bias) * mb_prog
        else:
            vol_loss.mean_bias_weight = self.target_mean_bias

        # qlike: only when calibration is roughly correct
        if hasattr(vol_loss, "qlike_weight") and self.qlike_target_weight is not None:
            q_target = float(self.qlike_target_weight)
            q_prog = min(1.0, float(e) / float(max(self.warm, 8)))
            near_ok = (self._scale_ema is None) or (0.9 <= self._scale_ema <= 1.1)
            vol_loss.qlike_weight = (q_target * q_prog) if near_ok else 0.0

        try:
            lr0 = trainer.optimizers[0].param_groups[0]["lr"]
        except Exception:
            lr0 = None
        print(
            f"[BIAS] epoch={e} under={vol_loss.underestimation_factor:.3f} "
            f"mean_bias={vol_loss.mean_bias_weight:.3f} "
            f"qlike_w={getattr(vol_loss, 'qlike_weight', 'n/a')} "
            f"scale_ema={self._scale_ema if self._scale_ema is not None else 'n/a'} "
            f"guard={'ON' if self._frozen else 'off'} "
            f"lr={lr0 if lr0 is not None else 'n/a'}"
        )

import sys

class SafeTQDMProgressBar(TQDMProgressBar):
    """Write tqdm to stderr to avoid BrokenPipe on stdout; reduce flush frequency."""
    def __init__(self, refresh_rate: int = 50):
        super().__init__(refresh_rate=refresh_rate)
        if not hasattr(self, "_tqdm_kwargs") or self._tqdm_kwargs is None:
            self._tqdm_kwargs = {}
        self._tqdm_kwargs.update({
            "file": sys.stderr,
            "mininterval": 0.5,      # fewer flushes
            "dynamic_ncols": True,
            "leave": True,
        })



class MedianMSELoss(nn.Module):
    def forward(self, y_hat_quantiles, y_true):
        q50 = y_hat_quantiles[:, 3]  # index of 0.50
        return F.mse_loss(q50, y_true)

class EpochLRDecay(pl.Callback):
    def __init__(self, gamma: float = 1, start_epoch: int = 1):
        """
        gamma: multiplicative decay per epoch (0.95 = -5%/epoch)
        start_epoch: begin decaying after this epoch index (0-based)
        """
        super().__init__()
        self.gamma = float(gamma)
        self.start_epoch = int(start_epoch)

    def on_train_epoch_end(self, trainer, pl_module):
        e = int(getattr(trainer, "current_epoch", 0))
        if e < self.start_epoch:
            return
        # scale all param_group LRs
        try:
            for opt in trainer.optimizers:
                for pg in opt.param_groups:
                    if "lr" in pg and pg["lr"] is not None:
                        pg["lr"] = float(pg["lr"]) * self.gamma
            new_lr = trainer.optimizers[0].param_groups[0]["lr"]
            print(f"[LR] epoch={e} → decayed lr to {new_lr:.6g}")
        except Exception as err:
            print(f"[LR] decay skipped: {err}")
            
# -----------------------------------------------------------------------
# Compute / device configuration (optimised for NVIDIA L4 on GCP)
# -----------------------------------------------------------------------
if torch.cuda.is_available():
    ACCELERATOR = "gpu"
    DEVICES = "auto"          # use all visible GPUs if more than one
    # default to bf16 but fall back to fp16 if unsupported (e.g., T4)
    PRECISION = "32" #bf16-mixed
    try:
        if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
            PRECISION = "16-mixed"
    except Exception:
        try:
            major, _minor = torch.cuda.get_device_capability()
            if major < 8:  # pre-Ampere
                PRECISION = "16-mixed"
        except Exception:
            PRECISION = "16-mixed"
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    ACCELERATOR = "mps"
    DEVICES = 1
    PRECISION = "16-mixed"
else:
    ACCELERATOR = "cpu"
    DEVICES = 1
    PRECISION = 32

# -----------------------------------------------------------------------
# CLI overrides for common CONFIG knobs
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# CLI overrides for common CONFIG knobs
# -----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="TFT training with optional permutation importance", add_help=True)
parser.add_argument("--disable_warmups", type=lambda s: str(s).lower() in ("1","true","t","yes","y","on"),
                    default=False, help="Disable bias/QLIKE warm-ups and tail ramp")
parser.add_argument("--warmup_guard_patience", type=int, default=2,
                    help="Consecutive worsening epochs before freezing warm-ups")
parser.add_argument("--warmup_guard_tol", type=float, default=0.0,
                    help="Minimum delta in val_loss to count as worsening")
parser.add_argument("--max_encoder_length", type=int, default=None, help="Max encoder length")
parser.add_argument("--max_epochs", type=int, default=None, help="Max training epochs")
parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
parser.add_argument("--early_stop_patience", "--patience", type=int, default=None, help="Early stopping patience")
parser.add_argument("--perm_len", type=int, default=None, help="Permutation block length for importance")
parser.add_argument("--perm_block_size", type=int, default=None, help="Alias for --perm_len (permutation block length)")
parser.add_argument(
    "--enable_perm_importance", "--enable-feature-importance",
    type=lambda s: str(s).lower() in ("1","true","t","yes","y","on"),
    default=None,
    help="Enable permutation feature importance (true/false)"
)
# Cloud paths / storage overrides
parser.add_argument("--gcs_bucket", type=str, default=None, help="GCS bucket name to read/write from")
parser.add_argument("--gcs_data_prefix", type=str, default=None, help="Full GCS prefix for data parquet folder")
parser.add_argument("--gcs_output_prefix", type=str, default=None, help="Full GCS prefix for outputs/checkpoints")
# Performance / input control
parser.add_argument("--data_dir", type=str, default=None, help="Local folder containing universal_*.parquet; if set, prefer local over GCS")
parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers (defaults to CPU count - 1)")
parser.add_argument("--prefetch_factor", type=int, default=8, help="DataLoader prefetch factor (per worker)")
# Performance / input control
parser.add_argument("--check_val_every_n_epoch", type=int, default=1, help="Validate every N epochs")
parser.add_argument("--log_every_n_steps", type=int, default=200, help="How often to log train steps")
parser.add_argument("--learning_rate", type=float, default=None, help="Override model learning rate")
parser.add_argument(
    "--resume",
    type=lambda s: str(s).lower() in ("1","true","t","yes","y","on"),
    default=False,                       # <— make default False
    help="Resume from latest checkpoint if available"
)
# Quick-run subsetting for speed
parser.add_argument("--train_max_rows", type=int, default=None, help="Limit number of rows in TRAIN for fast iterations")
parser.add_argument("--val_max_rows", type=int, default=None, help="Limit number of rows in VAL (optional; default full)")
parser.add_argument(
    "--subset_mode",
    type=str,
    default="per_asset_tail",
    choices=["per_asset_tail", "per_asset_head", "random"],
    help="Strategy for selecting a subset when limiting rows"
)
parser.add_argument("--fi_max_batches", type=int, default=20, help="Max val batches per feature in FI.")
# Parse known args so stray platform args do not crash the script
ARGS, _UNKNOWN = parser.parse_known_args()

# -----------------------------------------------------------------------
# CONFIG – tweak as required (GCS-aware)
# -----------------------------------------------------------------------
GCS_BUCKET = os.environ.get("GCS_BUCKET", "river-ml-bucket")
GCS_DATA_PREFIX = f"gs://{GCS_BUCKET}/Data/CleanedData"
GCS_OUTPUT_PREFIX = f"gs://{GCS_BUCKET}/Dissertation/TFT"

# Apply CLI overrides (if provided)
if getattr(ARGS, "gcs_bucket", None):
    GCS_BUCKET = ARGS.gcs_bucket
    # recompute defaults if specific prefixes are not provided
    if not getattr(ARGS, "gcs_data_prefix", None):
        GCS_DATA_PREFIX = f"gs://{GCS_BUCKET}/CleanedData"
    if not getattr(ARGS, "gcs_output_prefix", None):
        GCS_OUTPUT_PREFIX = f"gs://{GCS_BUCKET}/Dissertation/Feature_Ablation"
if getattr(ARGS, "gcs_data_prefix", None):
    GCS_DATA_PREFIX = ARGS.gcs_data_prefix
if getattr(ARGS, "gcs_output_prefix", None):
    GCS_OUTPUT_PREFIX = ARGS.gcs_output_prefix

# Local ephemerals (good for GCE/Vertex)
LOCAL_DATA_DIR = Path(os.environ.get("LOCAL_DATA_DIR", "/tmp/data/CleanedData"))
LOCAL_OUTPUT_DIR = Path(os.environ.get("LOCAL_OUTPUT_DIR", "/tmp/feature_ablation"))
LOCAL_RUN_DIR = Path(os.environ.get("LOCAL_RUN_DIR", "/tmp/tft_run"))
LOCAL_LOG_DIR = LOCAL_RUN_DIR / "lightning_logs"
LOCAL_CKPT_DIR = LOCAL_RUN_DIR / "checkpoints"
for p in [LOCAL_DATA_DIR, LOCAL_OUTPUT_DIR, LOCAL_CKPT_DIR, LOCAL_LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ---------------- TensorBoard Logger ----------------
logger = TensorBoardLogger(save_dir=str(LOCAL_LOG_DIR.parent), name=LOCAL_LOG_DIR.name)

# Remote checkpoint prefix on GCS
CKPT_GCS_PREFIX = f"{GCS_OUTPUT_PREFIX}/checkpoints"

# ---- Upload helper (GCS-aware) ----
def upload_file_to_gcs(local_path: str, gcs_uri: str):
    if fs is None:
        print(f"[WARN] GCS not available (gcsfs not installed); skipping upload: {gcs_uri}")
        return
    try:
        with fsspec.open(gcs_uri, "wb") as f_out, open(local_path, "rb") as f_in:
            shutil.copyfileobj(f_in, f_out)
        print(f"✓ Uploaded {local_path} → {gcs_uri}")
    except Exception as e:
        print(f"[WARN] Failed to upload {local_path} to {gcs_uri}: {e}")


# Prefer GCS if the files exist there
try:
    fs = fsspec.filesystem("gcs")
except Exception:
    fs = None  # gcsfs not installed / protocol unavailable

def gcs_exists(path: str) -> bool:
    if fs is None:
        return False
    try:
        return fs.exists(path)
    except Exception:
        return False

TRAIN_URI = f"{GCS_DATA_PREFIX}/universal_train.parquet"
VAL_URI   = f"{GCS_DATA_PREFIX}/universal_val.parquet"
TEST_URI  = f"{GCS_DATA_PREFIX}/universal_test.parquet"

# Resolve data paths: prefer explicit --data_dir, then GCS if available, else local default
def _local_cleaned_dir():
    if getattr(ARGS, "data_dir", None):
        return Path(ARGS.data_dir).expanduser().resolve()
    return Path("/Users/riverwest-gomila/Desktop/Data/CleanedData")

LOCAL_TRAIN = _local_cleaned_dir() / "universal_train.parquet"
LOCAL_VAL   = _local_cleaned_dir() / "universal_val.parquet"
LOCAL_TEST  = _local_cleaned_dir() / "universal_test.parquet"

def _all_exists_local(paths):
    try:
        return all(Path(p).exists() for p in paths)
    except Exception:
        return False

def _all_exists_gcs(paths):
    if fs is None:
        return False
    try:
        return all(fs.exists(p) for p in paths)
    except Exception:
        return False

# Choose the source of truth for READ_PATHS
if _all_exists_gcs([TRAIN_URI, VAL_URI, TEST_URI]):
    READ_PATHS = [TRAIN_URI, VAL_URI, TEST_URI]
elif _all_exists_local([LOCAL_TRAIN, LOCAL_VAL, LOCAL_TEST]):
    READ_PATHS = [str(LOCAL_TRAIN), str(LOCAL_VAL), str(LOCAL_TEST)]
else:
    raise FileNotFoundError(
        "Could not locate the dataset. Checked GCS URIs and the local default. "
        "Provide --data_dir to a folder containing universal_train/val/test.parquet, "
        "or ensure GCS is configured (install gcsfs & correct URIs)."
    )

def get_resume_ckpt_path():
    if not RESUME_ENABLED:
        return None

    try:
        local_ckpts = sorted(LOCAL_CKPT_DIR.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if local_ckpts:
            return str(local_ckpts[0])
    except Exception:
        pass
    # Fallback: try GCS "last.ckpt" then the lexicographically latest .ckpt
    try:
        if fs is not None:
            last_uri = f"{CKPT_GCS_PREFIX}/last.ckpt"
            if fs.exists(last_uri):
                dst = LOCAL_CKPT_DIR / "last.ckpt"
                with fsspec.open(last_uri, "rb") as f_in, open(dst, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                return str(dst)
            # else get the latest by name (filenames contain ISO timestamps)
            entries = fs.glob(f"{CKPT_GCS_PREFIX}/*.ckpt") or []
            if entries:
                latest = sorted(entries)[-1]
                dst = LOCAL_CKPT_DIR / Path(latest).name
                with fsspec.open(latest, "rb") as f_in, open(dst, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                return str(dst)
    except Exception as e:
        print(f"[WARN] Could not fetch resume checkpoint from GCS: {e}")
    return None

def mirror_local_ckpts_to_gcs():
    if fs is None:
        print("[WARN] GCS not available; skipping checkpoint upload.")
        return
    try:
        for p in LOCAL_CKPT_DIR.glob("*.ckpt"):
            remote = f"{CKPT_GCS_PREFIX}/{p.name}"
            with fsspec.open(remote, "wb") as f_out, open(p, "rb") as f_in:
                shutil.copyfileobj(f_in, f_out)
            #print(f"✓ Mirrored checkpoint {p} → {remote}")
    except Exception as e:
        print(f"[WARN] Failed to mirror checkpoints: {e}")

class MirrorCheckpoints(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        mirror_local_ckpts_to_gcs()
    def on_exception(self, trainer, pl_module, err):
        mirror_local_ckpts_to_gcs()
    def on_train_end(self, trainer, pl_module):
        mirror_local_ckpts_to_gcs()

class TailWeightRamp(pl.Callback):
    def __init__(self, vol_loss, start: float = 1.0, end: float = 1.25, ramp_epochs: int = 12):
        super().__init__()
        self.vol_loss, self.start, self.end, self.ramp = vol_loss, float(start), float(end), int(ramp_epochs)

    def on_train_epoch_start(self, trainer, pl_module):
        # Global CLI switch: if --disable_warmups is true, hold tail_weight steady
        try:
            if globals().get("ARGS") is not None and getattr(ARGS, "disable_warmups", False):
                print(f"[TAIL] epoch={int(getattr(trainer,'current_epoch',0))} DISABLED via --disable_warmups; tail_weight stays {self.vol_loss.tail_weight}")
                return
        except Exception:
            pass
        frozen = False
        for cb in getattr(trainer, 'callbacks', []):
            if isinstance(cb, BiasWarmupCallback) and getattr(cb, '_frozen', False):
                frozen = True
                break
        if frozen:
            print(f"[TAIL] epoch={int(getattr(trainer, 'current_epoch', 0))} frozen; tail_weight stays {self.vol_loss.tail_weight}")
            return

        e = int(getattr(trainer, "current_epoch", 0))
        eff_end = min(self.end, 1.5)
        eff_ramp = max(self.ramp, 12)
        prog = 1.0 if eff_ramp <= 0 else min(1.0, e / float(eff_ramp))
        self.vol_loss.tail_weight = self.start + (eff_end - self.start) * prog
        print(f"[TAIL] epoch={e} tail_weight={self.vol_loss.tail_weight}")

class ReduceLROnPlateauCallback(pl.Callback):
    def __init__(self, monitor="val_qlike_overall", factor=0.5, patience=3, min_lr=1e-5, cooldown=0):
        self.monitor, self.factor, self.patience, self.min_lr, self.cooldown = monitor, factor, patience, min_lr, cooldown
        self.best, self.bad, self.cool = float("inf"), 0, 0

    def on_validation_end(self, trainer, pl_module):
        if self.cool:
            self.cool -= 1
            return
        val = trainer.callback_metrics.get(self.monitor)
        if val is None:
            return
        try:
            val = float(val.item())
        except Exception:
            val = float(val)
        if val + 1e-12 < self.best:
            self.best, self.bad = val, 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                for opt in trainer.optimizers:
                    for pg in opt.param_groups:
                        if "lr" in pg and pg["lr"] is not None:
                            pg["lr"] = max(self.min_lr, float(pg["lr"]) * self.factor)
                new_lr = trainer.optimizers[0].param_groups[0]["lr"]
                print(f"[LR Plateau] ↓ lr → {new_lr:.6g}")
                self.bad, self.cool = 0, self.cooldown


VOL_LOSS = AsymmetricQuantileLoss(
    quantiles=VOL_QUANTILES,
    underestimation_factor=1.00,  # managed by BiasWarmupCallback
    mean_bias_weight=0.01,        # small centering on the median for MAE
    tail_q=0.85,
    tail_weight=1.0,              # will be ramped by TailWeightRamp
    qlike_weight=0.0,             # QLIKE weight is ramped safely in BiasWarmupCallback
    reduction="mean",
)
# ---------------- Callback bundle (bias warm-up, tail ramp, LR control) ----------------
EXTRA_CALLBACKS = [
    # Gated calibration: ramps alpha/mean-bias and only enables QLIKE when scale is near 1
    BiasWarmupCallback(
        vol_loss=VOL_LOSS,
        target_under=1.00,
        target_mean_bias=0.04,
        warmup_epochs=4,
        qlike_target_weight=0.20,   # ramps in when mean(y)/mean(p) ~ 1
        start_mean_bias=0.0,
        mean_bias_ramp_until=8,
        guard_patience=getattr(ARGS, "warmup_guard_patience", 2),
        guard_tol=getattr(ARGS, "warmup_guard_tol", 0.0),
        alpha_step=0.05,
    ),
    # Gradually emphasise upper tail once training is stable → protects QLIKE
    TailWeightRamp(vol_loss=VOL_LOSS, start=1.0, end=1.22, ramp_epochs=12),
    # Safety net if QLIKE plateaus
    ReduceLROnPlateauCallback(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, cooldown=1),
]

class ValLossHistory(pl.Callback):
    """
    Records per-epoch validation metrics to a CSV so you can plot later.
    """
    def __init__(self, out_csv: Path):
        super().__init__()
        self.out_csv = Path(out_csv)
        self.rows = []

    def _as_float(self, x):
        try:
            return float(x.item() if hasattr(x, "item") else x)
        except Exception:
            return float("nan")

    def _current_lr(self, trainer):
        try:
            return float(trainer.optimizers[0].param_groups[0]["lr"])
        except Exception:
            return float("nan")

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        row = {
            "epoch": int(getattr(trainer, "current_epoch", -1)) + 1,
            "val_loss":            self._as_float(m.get("val_loss", float("nan"))),
            "val_qlike_overall":   self._as_float(m.get("val_qlike_overall", float("nan"))),
            "val_mae_overall":     self._as_float(m.get("val_mae_overall", float("nan"))),
            "val_rmse_overall":    self._as_float(m.get("val_rmse_overall", float("nan"))),
            "val_acc_overall":     self._as_float(m.get("val_acc_overall", float("nan"))),
            "val_brier_overall":   self._as_float(m.get("val_brier_overall", float("nan"))),
            "val_auroc_overall":   self._as_float(m.get("val_auroc_overall", float("nan"))),
            "lr":                  self._current_lr(trainer),
        }
        self.rows.append(row)
        try:
            import pandas as _pd
            _pd.DataFrame(self.rows).to_csv(self.out_csv, index=False)
            print(f"[VAL-HIST] wrote {self.out_csv}")
        except Exception as e:
            print(f"[VAL-HIST] write failed: {e}")

GROUP_ID: List[str] = ["asset"]
TIME_COL = "Time"
TARGETS  = ["realised_vol", "direction"]

MAX_ENCODER_LENGTH = 96
MAX_PRED_LENGTH    = 1

EMBEDDING_CARDINALITY = {}

BATCH_SIZE   = 256
MAX_EPOCHS   = 5
EARLY_STOP_PATIENCE = 12
PERM_BLOCK_SIZE = 288

# Artifacts are written locally then uploaded to GCS
from datetime import datetime, timezone, timedelta
RUN_SUFFIX = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
MODEL_SAVE_PATH = (LOCAL_CKPT_DIR / f"tft_realised_vol_e{MAX_EPOCHS}_{RUN_SUFFIX}.ckpt")

SEED = 8
# Full-run determinism for reproducible validation metrics
try:
    seed_everything(SEED, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except Exception:
    pass
WEIGHT_DECAY = 0.0001 #0.00578350719515325     # weight decay for AdamW
GRADIENT_CLIP_VAL = 0.78 #0.78    # gradient clipping value for Trainer
# Feature-importance controls
ENABLE_FEATURE_IMPORTANCE = True   # gate FI so you can toggle it
FI_MAX_BATCHES = 40       # number of val batches to sample for FI

# ---- Apply CLI overrides (only when provided) ----
if ARGS.batch_size is not None:
    BATCH_SIZE = int(ARGS.batch_size)
if ARGS.max_encoder_length is not None:
    MAX_ENCODER_LENGTH = int(ARGS.max_encoder_length)
if ARGS.max_epochs is not None:
    MAX_EPOCHS = int(ARGS.max_epochs)
if ARGS.early_stop_patience is not None:
    EARLY_STOP_PATIENCE = int(ARGS.early_stop_patience)
if ARGS.perm_len is not None:
    PERM_BLOCK_SIZE = int(ARGS.perm_len)
if ARGS.enable_perm_importance is not None:
    ENABLE_FEATURE_IMPORTANCE = bool(ARGS.enable_perm_importance)
if getattr(ARGS, "fi_max_batches", None) is not None:
    FI_MAX_BATCHES = int(ARGS.fi_max_batches)

# ---- Learning rate and resume CLI overrides ----
LR_OVERRIDE = float(ARGS.learning_rate) if getattr(ARGS, "learning_rate", None) is not None else None
RESUME_ENABLED = bool(getattr(ARGS, "resume", True))
RESUME_CKPT = get_resume_ckpt_path() if RESUME_ENABLED else None

# -----------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------
def load_split(path: str) -> pd.DataFrame:
    """
    Load a parquet split and guarantee we have usable `id` and timestamp columns.

    1. Converts `TIME_COL` to pandas datetime.
    2. Ensures an identifier column called `id` (or whatever `GROUP_ID[0]` is).
       If not present, auto-detects common synonyms and renames
    """
    path_str = str(path)
    df = pd.read_parquet(path)
    df = df.reset_index(drop=True).copy()

    # --- convert timestamp ---
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    # --- identifier column handling ---
    if GROUP_ID[0] not in df.columns:
        cand = next(
            (c for c in df.columns if c.lower() in
             {"symbol", "ticker", "asset", "security_id", "instrument"}), None
        )
        if cand is None:
            raise ValueError(
                f"No identifier column '{GROUP_ID[0]}' in {path_str}. Edit GROUP_ID or rename your column."
            )

        df.rename(columns={cand: GROUP_ID[0]}, inplace=True)
        print(f"[INFO] Renamed '{cand}' ➜ '{GROUP_ID[0]}'")

    df[GROUP_ID[0]] = df[GROUP_ID[0]].astype(str)

    # --- target alias handling ------------------------------------------------
    TARGET_ALIASES = {
        "realised_vol": ["Realised_Vol", "rs_sigma", "realized_vol", "rv"],
        "direction":    ["Sign_Label", "sign_label", "Direction", "direction_label"],
    }
    for canonical, aliases in TARGET_ALIASES.items():
        if canonical not in df.columns:
            alias_found = next((a for a in aliases if a in df.columns), None)
            if alias_found:
                df.rename(columns={alias_found: canonical}, inplace=True)
                print(f"[INFO] Renamed '{alias_found}' ➜ '{canonical}'")
            else:
                # Only warn; the downstream code will raise if the column is truly required
                print(f"[WARN] Column '{canonical}' not found in {path_str} and no alias detected.")

    return df


def add_time_idx(df: pd.DataFrame) -> pd.DataFrame:
    """Add monotonically increasing integer time index per asset."""
    df = df.sort_values(GROUP_ID + [TIME_COL])
    df["time_idx"] = (
        df.groupby(GROUP_ID)
          .cumcount()
          .astype("int64")
    )
    return df

# -----------------------------------------------------------------------
# Split sanity: enforce chronological, non-overlapping 80/10/10 per asset
# -----------------------------------------------------------------------
def _chronological_resplit(df: pd.DataFrame, train_frac: float = 0.8, val_frac: float = 0.1):
    """
    Deterministically re-split a combined dataframe into train/val/test by time,
    per asset, using the provided fractions (train, val, test = rest).
    Assumes columns: 'asset', 'Time'.
    Returns: (train_df, val_df, test_df)
    """
    assert 0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1
    df = df.sort_values(["asset", "Time"]).reset_index(drop=True)
    parts = []
    for asset, g in df.groupby("asset", observed=True, sort=False):
        n = len(g)
        if n == 0:
            continue
        tr_end = int(n * train_frac)
        va_end = int(n * (train_frac + val_frac))
        g = g.copy()
        g.loc[g.index[:tr_end], "split"] = "train"
        g.loc[g.index[tr_end:va_end], "split"] = "val"
        g.loc[g.index[va_end:], "split"] = "test"
        parts.append(g)
    out = pd.concat(parts, axis=0, ignore_index=False)
    train_df = out[out["split"] == "train"].drop(columns=["split"]).copy()
    val_df   = out[out["split"] == "val"].drop(columns=["split"]).copy()
    test_df  = out[out["split"] == "test"].drop(columns=["split"]).copy()
    return train_df, val_df, test_df


def assert_and_fix_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Ensure each split is chronologically sorted per asset and there is no time overlap:
      max(train) < min(val) and max(val) < min(test), per asset.
    If violated for any asset, re-split deterministically (80/10/10) from the union.
    Returns possibly corrected (train_df, val_df, test_df).
    """
    # Sort each split per asset/time
    train_df = train_df.sort_values(["asset", "Time"]).copy()
    val_df   = val_df.sort_values(["asset", "Time"]).copy()
    test_df  = test_df.sort_values(["asset", "Time"]).copy()

    def _bounds(df):
        if df.empty:
            return pd.DataFrame(columns=["asset","min","max"]).set_index("asset")
        return df.groupby("asset", observed=True)["Time"].agg(["min","max"])

    tb = _bounds(train_df).rename(columns={"min":"tr_min","max":"tr_max"})
    vb = _bounds(val_df).rename(columns={"min":"va_min","max":"va_max"})
    sb = _bounds(test_df).rename(columns={"min":"te_min","max":"te_max"})

    bounds = tb.join(vb, how="outer").join(sb, how="outer")

    bad_assets = []
    for a, r in bounds.iterrows():
        tr_max = r.get("tr_max", pd.NaT)
        va_min = r.get("va_min", pd.NaT)
        va_max = r.get("va_max", pd.NaT)
        te_min = r.get("te_min", pd.NaT)
        ok = True
        if pd.notna(tr_max) and pd.notna(va_min):
            ok = ok and (tr_max < va_min)
        if pd.notna(va_max) and pd.notna(te_min):
            ok = ok and (va_max < te_min)
        if not ok:
            bad_assets.append(a)

    if bad_assets:
        print(f"[SPLIT] Detected chronological overlap for assets: {bad_assets} — re-splitting 80/10/10 by time.")
        union = pd.concat([
            train_df.assign(_orig_split="train"),
            val_df.assign(_orig_split="val"),
            test_df.assign(_orig_split="test"),
        ], axis=0, ignore_index=False).drop_duplicates().copy()
        union = union.sort_values(["asset","Time"]).drop_duplicates(subset=["asset","Time"], keep="first")
        train_df, val_df, test_df = _chronological_resplit(union, train_frac=0.8, val_frac=0.1)
    else:
        print("[SPLIT] Splits look chronological and non-overlapping.")
    return train_df, val_df, test_df

# -----------------------------------------------------------------------
# Calendar features (help the model learn intraday/weekly seasonality)
# -----------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    minute_of_day = df[TIME_COL].dt.hour * 60 + df[TIME_COL].dt.minute
    df["sin_tod"] = np.sin(2 * np.pi * minute_of_day / 1440.0).astype("float32")
    df["cos_tod"] = np.cos(2 * np.pi * minute_of_day / 1440.0).astype("float32")
    dow = df[TIME_COL].dt.dayofweek
    df["sin_dow"] = np.sin(2 * np.pi * dow / 7.0).astype("float32")
    df["cos_dow"] = np.cos(2 * np.pi * dow / 7.0).astype("float32")
    df["Is_Weekend"] = (dow >= 5).astype("int8")
    return df

# -----------------------------------------------------------------------
# Fast subset helper (keeps temporal structure per asset when possible)
# -----------------------------------------------------------------------
def subset_time_series(df: pd.DataFrame, max_rows: int | None, mode: str = "per_asset_tail") -> pd.DataFrame:
    """Return a subset of *approximately* max_rows, preserving sequence order.
    modes:
      • per_asset_tail: take roughly equal tail slices per asset, then trim
      • per_asset_head: same but from the head
      • random: global random sample (may break sequences; use only for quick smoke tests)
    """
    if max_rows is None or max_rows <= 0 or max_rows >= len(df):
        return df
    df = df.sort_values(GROUP_ID + [TIME_COL]).reset_index(drop=True)
    if mode == "random":
        out = df.sample(n=int(max_rows), random_state=SEED).sort_values(GROUP_ID + [TIME_COL])
        return out.reset_index(drop=True)

    # Per-asset slicing
    groups = list(df.groupby(GROUP_ID[0], observed=True)) if GROUP_ID else [(None, df)]
    n_assets = max(1, len(groups))
    take_each = max(1, int(np.ceil(max_rows / n_assets)))
    parts = []
    for _, gdf in groups:
        if mode == "per_asset_head":
            parts.append(gdf.head(take_each))
        else:  # per_asset_tail
            parts.append(gdf.tail(take_each))
    out = pd.concat(parts, axis=0, ignore_index=True)
    # trim to target size while preserving order (head or tail consistent with mode)
    out = out.sort_values(GROUP_ID + [TIME_COL]).reset_index(drop=True)
    if len(out) > max_rows:
        if mode == "per_asset_head":
            out = out.head(int(max_rows))
        else:
            out = out.tail(int(max_rows))
    return out.reset_index(drop=True)

# -----------------------------------------------------------------------
# Permutation Importance helpers at module scope (decoded metric = MAE + RMSE + 0.05 * DirBCE)
# -----------------------------------------------------------------------


# NOTE: If you explicitly instantiate a TQDMProgressBar in your callbacks, remove it or comment it out.
# For example, if you see:
# callbacks.append(TQDMProgressBar(...))
# or
# callbacks = [TQDMProgressBar(...), ...]
# remove the TQDMProgressBar from the list.

@torch.no_grad()
def _evaluate_decoded_metrics(
    model,
    ds: TimeSeriesDataSet,
    batch_size: int,
    max_batches: int,
    num_workers: int,
    prefetch: int,
    pin_memory: bool,
    vol_norm,                 # GroupNormalizer from TRAIN (for realised_vol)
):
    """
    Evaluate model on a dataset configured with predict=True, computing:
      • MAE, RMSE, QLIKE on decoded realised_vol
      • Brier on direction (if available)
    Returns: (mae, rmse, brier, qlike, N)
    """
    model.eval()
    model_device = next(model.parameters()).device  # <<< NEW

    # dataloader mirrors validation loader; dataset already has predict=True
    loader = ds.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=False,
        prefetch_factor=prefetch if num_workers and num_workers > 0 else None,
    )

    # --- helper to move nested batch dicts to model_device ---
    def _move_to_device(x):
        if torch.is_tensor(x):
            return x.to(model_device, non_blocking=True)
        if isinstance(x, dict):
            return {k: _move_to_device(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            t = [_move_to_device(v) for v in x]
            return type(x)(t) if not isinstance(x, tuple) else tuple(t)
        return x

    g_list, yv_list, pv_list = [], [], []
    yd_list, pd_list = [], []

    batches_seen = 0
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x = batch[0]
            y = batch[1]
        else:
            continue
        if not isinstance(x, dict):
            continue

        # --- MOVE EVERYTHING to the model's device BEFORE forward ---
        x_dev = _move_to_device(x)

        # group ids (taken from x_dev so same device)
        groups = None
        for k in ("groups", "group_ids", "group_id"):
            if k in x_dev and x_dev[k] is not None:
                groups = x_dev[k]
                break
        if groups is None:
            continue
        g = groups[0] if isinstance(groups, (list, tuple)) else groups
        while torch.is_tensor(g) and g.ndim > 1 and g.size(-1) == 1:
            g = g.squeeze(-1)

        # targets from decoder_target (fallback to y)
        dec_t = x_dev.get("decoder_target", None)
        y_vol_t, y_dir_t = None, None
        if torch.is_tensor(dec_t):
            t = dec_t
            if t.ndim == 3 and t.size(-1) == 1:
                t = t[..., 0]
            if t.ndim == 2 and t.size(1) >= 1:
                y_vol_t = t[:, 0]
                if t.size(1) > 1:
                    y_dir_t = t[:, 1]
        elif isinstance(dec_t, (list, tuple)) and len(dec_t) >= 1:
            y_vol_t = dec_t[0]
            if torch.is_tensor(y_vol_t) and y_vol_t.ndim == 3 and y_vol_t.size(-1) == 1:
                y_vol_t = y_vol_t[..., 0]
            if len(dec_t) > 1 and torch.is_tensor(dec_t[1]):
                y_dir_t = dec_t[1]
                if y_dir_t.ndim == 3 and y_dir_t.size(-1) == 1:
                    y_dir_t = y_dir_t[..., 0]

        if (y_vol_t is None) and torch.is_tensor(y):
            yy = y.to(model_device, non_blocking=True)  # <<< ensure same device if used
            if yy.ndim == 3 and yy.size(1) == 1:
                yy = yy[:, 0, :]
            if yy.ndim == 2 and yy.size(1) >= 1:
                y_vol_t = yy[:, 0]
                if yy.size(1) > 1:
                    y_dir_t = yy[:, 1]

        if (y_vol_t is None) or (not torch.is_tensor(g)):
            continue

        # forward pass and head extraction
        y_hat = model(x_dev)  # <<< CUDA-safe now
        pred = getattr(y_hat, "prediction", y_hat)
        if isinstance(pred, dict) and "prediction" in pred:
            pred = pred["prediction"]

        # split heads (same logic as before)
        def _extract_heads(prediction):
            if isinstance(prediction, (list, tuple)):
                vol_q = prediction[0]
                d_log = prediction[1] if len(prediction) > 1 else None
                return vol_q, d_log
            t = prediction
            if torch.is_tensor(t):
                if t.ndim >= 4 and t.size(1) == 1:
                    t = t.squeeze(1)
                if t.ndim == 3 and t.size(1) == 1:
                    t = t[:, 0, :]
                if t.ndim == 2:
                    vol_q = t[:, :-1]
                    d_log = t[:, -1]
                    return vol_q, d_log
            return None, None

        p_vol, p_dir = _extract_heads(pred)
        if p_vol is None:
            continue

        # take median quantile for vol
        def _median_from_quantiles(vol_q: torch.Tensor) -> torch.Tensor:
            vol_q = torch.cummax(vol_q, dim=-1).values
            # assumes 7 quantiles with 0.50 at index 3
            return vol_q[..., 3]

        p_vol_med = _median_from_quantiles(p_vol)

        L = g.shape[0]
        g_list.append(g.reshape(L))
        yv_list.append(y_vol_t.reshape(L))
        pv_list.append(p_vol_med.reshape(L))

        if y_dir_t is not None and p_dir is not None:
            yd = y_dir_t.reshape(-1)
            pd = p_dir.reshape(-1)
            L2 = min(L, yd.numel(), pd.numel())
            if L2 > 0:
                yd_list.append(yd[:L2])
                pd_list.append(pd[:L2])

        batches_seen += 1
        if max_batches is not None and max_batches > 0 and batches_seen >= max_batches:
            break

    if not g_list:
        # empty input → return NaNs-ish but finite
        return 1.0, 1.0, 0.25, 10.0, 0

    device = g_list[0].device
    g_all  = torch.cat(g_list).to(device)
    y_all  = torch.cat(yv_list).to(device)
    p_all  = torch.cat(pv_list).to(device)

    # decode realised_vol
    y_dec = safe_decode_vol(y_all.unsqueeze(-1), vol_norm, g_all.unsqueeze(-1)).squeeze(-1)
    p_dec = safe_decode_vol(p_all.unsqueeze(-1), vol_norm, g_all.unsqueeze(-1)).squeeze(-1)
    floor_val = globals().get("EVAL_VOL_FLOOR", 1e-8)
    y_dec = torch.clamp(y_dec, min=floor_val)
    p_dec = torch.clamp(p_dec, min=floor_val)

    # metrics
    eps = 1e-8
    diff = (p_dec - y_dec)
    mae  = diff.abs().mean().item()
    rmse = (diff.pow(2).mean().sqrt().item())

    sigma2_p = torch.clamp(p_dec.abs(), min=eps) ** 2
    sigma2_y = torch.clamp(y_dec.abs(), min=eps) ** 2
    ratio    = sigma2_y / sigma2_p
    qlike    = (ratio - torch.log(ratio) - 1.0).mean().item()

    brier = None
    if yd_list and pd_list:
        yd = torch.cat(yd_list).to(device).float()
        pd = torch.cat(pd_list).to(device)
        try:
            if torch.isfinite(pd).any() and (pd.min() < 0 or pd.max() > 1):
                pd = torch.sigmoid(pd)
        except Exception:
            pd = torch.sigmoid(pd)
        pd = torch.clamp(pd, 0.0, 1.0)
        brier = ((pd - yd) ** 2).mean().item()

    return float(mae), float(rmse), (float(brier) if brier is not None else float("nan")), float(qlike), int(y_dec.numel())

# --- after trainer.fit(...), before running FI ---
def _resolve_best_model(trainer, fallback):
    # try any ModelCheckpoint attached to the trainer
    best_path = None
    try:
        for cb in getattr(trainer, "callbacks", []):
            if isinstance(cb, ModelCheckpoint) and getattr(cb, "best_model_path", ""):
                best_path = cb.best_model_path
                if best_path:
                    break
    except Exception:
        pass

    # fallback to newest local ckpt
    if not best_path:
        try:
            ckpts = sorted(LOCAL_CKPT_DIR.glob("*.ckpt"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
            if ckpts:
                best_path = str(ckpts[0])
        except Exception:
            pass

    # load or return the in-memory model
    if best_path:
        try:
            print(f"Best checkpoint: {best_path}")
            return TemporalFusionTransformer.load_from_checkpoint(best_path)
        except Exception as e:
            print(f"[WARN] load_from_checkpoint failed: {e}")
    return fallback

@torch.no_grad()
def run_permutation_importance(
    model,
    template_ds,
    base_df: pd.DataFrame,
    features: List[str],
    block_size: int,
    batch_size: int,
    max_batches: int,
    num_workers: int,
    prefetch: int,
    pin_memory: bool,
    vol_norm,                       # 👈 pass GroupNormalizer from TRAIN
    out_csv_path: str,
    uploader,
    build_ds_fn=lambda: None,       # compatibility; unused
) -> None:
    """
    Permutation Feature Importance on **decoded** metrics with a **baseline-normalised composite**:

        Comp = w_mae*(MAE/MAE0) + w_rmse*(RMSE/RMSE0) + w_qlike*(QLIKE/QLIKE0)

    We log per-feature deltas:
        Δ_COMP = Comp_perm - Comp_base  (positive = harmful; negative = helpful)
    """

    import os

    # ---------- optional RESUME ----------
    already = set()
    existing_rows = []
    if out_csv_path and os.path.exists(out_csv_path):
        try:
            prev = pd.read_csv(out_csv_path)
            if "feature" in prev.columns:
                already = set(prev["feature"].astype(str).tolist())
                existing_rows = prev.to_dict("records")
                print(f"[FI] Resuming — {len(already)} features already in {out_csv_path}")
        except Exception as e:
            print(f"[FI WARN] Could not read existing FI CSV for resume: {e}")

    # ---------- BASELINE (predict=True, last step) ----------
    ds_base = TimeSeriesDataSet.from_dataset(
        template_ds,
        base_df,
        predict=False,
        stop_randomization=True,
    )

    b_mae, b_rmse, b_brier, b_qlike, n_base = _evaluate_decoded_metrics(
        model=model,
        ds=ds_base,
        batch_size=batch_size,
        max_batches=max_batches,
        num_workers=num_workers,
        prefetch=prefetch,
        pin_memory=pin_memory,
        vol_norm=vol_norm,
    )

    base_comp = composite_score(b_mae, b_rmse, b_qlike, b_mae, b_rmse, b_qlike, COMP_WEIGHTS)

    print(
        f"[FI] Dataset size (samples): {len(base_df)} | batch_size={batch_size}\n"
        f"[FI] Baseline: QLIKE={b_qlike:.6f} | MAE={b_mae:.6f} | RMSE={b_rmse:.6f} | "
        f"COMPOSITE(base)={base_comp:.4f} | N={n_base}"
    )
    if b_mae > 0.01:
        print("[FI DEBUG] WARNING: baseline decoded MAE unusually large (expected ~0.001–0.003). "
              "Check vol_norm and decode pipeline.")

    new_rows = []
    work_df = base_df.copy()

    # ---------- PERMUTE LOOP ----------
    for feat in features:
        if str(feat) in already:
            print(f"[FI] {feat:>20s} | SKIP (already present)")
            continue

        df_perm = work_df.copy()
        _permute_series_inplace(df_perm, feat, block=block_size, group_col="asset")

        ds_perm = TimeSeriesDataSet.from_dataset(
            template_ds,
            df_perm,
            predict=False,
            stop_randomization=True,
        )

        p_mae, p_rmse, p_brier, p_qlike, n_p = _evaluate_decoded_metrics(
            model=model,
            ds=ds_perm,
            batch_size=batch_size,
            max_batches=max_batches,
            num_workers=num_workers,
            prefetch=prefetch,
            pin_memory=pin_memory,
            vol_norm=vol_norm,
        )

        # individual deltas (lower is better)
        d_mae   = float(p_mae   - b_mae)
        d_rmse  = float(p_rmse  - b_rmse)
        d_qlike = float(p_qlike - b_qlike)

        # baseline-normalised composite
        p_comp = composite_score(p_mae, p_rmse, p_qlike, b_mae, b_rmse, b_qlike, COMP_WEIGHTS)
        d_comp = float(p_comp - base_comp)  # + = harmful (we minimise)

        print(
            f"[FI] {feat:>20s} | QLIKE={p_qlike:.6f} Δ_QLIKE={d_qlike:+.6f} | "
            f"Δ_MAE={d_mae:+.6f} Δ_RMSE={d_rmse:+.6f} | "
            f"COMPOSITE={p_comp:.4f} Δ_COMP={d_comp:+.4f} "
            f"{'(↑ worse ⇒ useful)' if d_comp > 0 else '(↓ better ⇒ harmful/redundant)'}"
        )

        new_rows.append({
            "feature": str(feat),
            "baseline_mae": float(b_mae),
            "baseline_rmse": float(b_rmse),
            "baseline_qlike": float(b_qlike),
            "permuted_mae": float(p_mae),
            "permuted_rmse": float(p_rmse),
            "permuted_qlike": float(p_qlike),
            "delta_mae": float(d_mae),
            "delta_rmse": float(d_rmse),
            "delta_qlike": float(d_qlike),
            "composite_base": float(base_comp),
            "composite_perm": float(p_comp),
            "delta_composite": float(d_comp),
            "brier": (float(p_brier) if p_brier == p_brier else None),
            "weights": f"mae={COMP_WEIGHTS[0]},rmse={COMP_WEIGHTS[1]},qlike={COMP_WEIGHTS[2]}",
            "n": int(n_p),
        })

    # ---------- SAVE / UPLOAD ----------
    try:
        out_df = pd.DataFrame((existing_rows or []) + new_rows)
        out_df.to_csv(out_csv_path, index=False)
        print(f"[FI] wrote {out_csv_path}")
        if uploader is not None:
            try:
                uploader(out_csv_path, f"{GCS_OUTPUT_PREFIX}/{Path(out_csv_path).name}")
            except Exception as e:
                print(f"[FI WARN] upload failed: {e}")
    except Exception as e:
        print(f"[FI WARN] could not save FI CSV: {e}")

# -----------------------------------------------------------------------
# Standard Variable Importance (SVI) via TFT.interpret_output
# -----------------------------------------------------------------------

@torch.no_grad()
def save_standard_variable_importance(
    model,
    dataloader,
    ds,                                # pass your validation_dataset here
    out_csv_path="standard_variable_importance.csv",
    uploader=None,
    min_samples: int = 10_000,         # target number of samples to aggregate
    max_batches: int | None = None     # optional hard cap on batches
):
    """
    Compute Standard Variable Importance (SVI) by averaging PyTorch-Forecasting's
    `interpret_output` across many batches until at least `min_samples` examples
    are seen (or the dataloader is exhausted). Robust to PF versions that return
    dicts or tuples, and to runs where certain keys are missing.

    We aggregate three groups when available:
      • encoder variable importance
      • decoder variable importance
      • static variable importance
    """
    import numpy as _np
    import pandas as _pd
    import torch as _torch
    from pathlib import Path as _Path

    model.eval()

    # ---- helpers ----
    def _move_to_device(x, device):
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                if _torch.is_tensor(v):
                    out[k] = v.to(device, non_blocking=True)
                elif isinstance(v, (list, tuple)):
                    out[k] = [vv.to(device, non_blocking=True) if _torch.is_tensor(vv) else vv for vv in v]
                else:
                    out[k] = v
            return out
        return x

    def _get_any(d, keys, default=None):
        for k in keys:
            if isinstance(d, dict) and k in d and d[k] is not None:
                return d[k]
        return default

    def _norm_out(out_obj):
        # normalise PF interpret_output return to a dict
        if isinstance(out_obj, dict):
            return out_obj
        if isinstance(out_obj, (list, tuple)):
            for o in out_obj:
                if isinstance(o, dict):
                    return o
            # fallback: wrap positional outputs (rare)
            return {"attention": out_obj}
        return {}

    def _to_numpy_1d(a):
        if a is None:
            return None
        if isinstance(a, _torch.Tensor):
            a = a.detach().cpu()
        a = _np.asarray(a)
        # PF often returns [B, n_vars] for reduction='none' → average over B
        if a.ndim == 2:
            a = a.mean(axis=0)
        elif a.ndim > 2:
            a = a.reshape(-1, a.shape[-1]).mean(axis=0)
        return a.astype(float).ravel()

    # names from the dataset (never rely on model.dataset)
    enc_names = list(getattr(ds, "encoder_variables", []) or [])
    dec_names = list(getattr(ds, "decoder_variables", []) or [])
    stat_names = list(
        getattr(ds, "static_categoricals", []) or
        getattr(ds, "static_variables", []) or []
    )

    device = next(model.parameters()).device
    n_seen = 0

    # running sums and counters
    enc_sum = None
    dec_sum = None
    stat_sum = None
    batch_count = 0

    it = iter(dataloader)
    while True:
        if max_batches is not None and batch_count >= int(max_batches):
            break
        try:
            batch = next(it)
        except StopIteration:
            break

        # x only; interpret_output will run forward internally
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        if not isinstance(x, dict):
            continue

        x_dev = _move_to_device(x, device)

        # Prefer reduction='none' so we can average across many batches ourselves.
        # If PF version errors on that, fall back to 'mean'.
        try:
            out = model.interpret_output(x_dev, reduction="none")
        except KeyError:
            # PF version/config did not return attentions (e.g., 'decoder_attention' missing)
            # Exit SVI quietly; nothing to aggregate.
            return
        except Exception:
            try:
                out = model.interpret_output(x_dev, reduction="mean")
            except KeyError:
                return
            except Exception:
                # Skip this batch quietly
                continue

        res = _norm_out(out)

        enc_imp = _get_any(res, [
            "encoder_variables", "encoder_variable_importance", "encoder_importance"
        ])
        dec_imp = _get_any(res, [
            "decoder_variables", "decoder_variable_importance", "decoder_importance"
        ])
        stat_imp = _get_any(res, [
            "static_variables", "static_variable_importance", "static_importance"
        ])

        enc_vec = _to_numpy_1d(enc_imp)
        dec_vec = _to_numpy_1d(dec_imp)
        stat_vec = _to_numpy_1d(stat_imp)

        # Determine how many samples this batch contributed. With reduction='none'
        # encoder/decoder often come back as [B, n_vars]; otherwise assume batch size.
        contrib = 0
        try:
            rep = None
            for cand in (enc_imp, dec_imp, stat_imp):
                if isinstance(cand, _torch.Tensor) and cand.ndim >= 2:
                    rep = cand
                    break
            if rep is None:
                # fallback to batch size if we can find it in x
                for k, v in x.items():
                    if _torch.is_tensor(v) and v.shape[0] > 0:
                        rep = v
                        break
            if isinstance(rep, _torch.Tensor):
                contrib = int(rep.shape[0])
        except Exception:
            contrib = 0

        # accumulate; align length to names if needed
        if enc_vec is not None:
            if enc_sum is None:
                enc_sum = _np.zeros_like(enc_vec, dtype=float)
            if len(enc_vec) == len(enc_sum):
                enc_sum += enc_vec
        if dec_vec is not None:
            if dec_sum is None:
                dec_sum = _np.zeros_like(dec_vec, dtype=float)
            if len(dec_vec) == len(dec_sum):
                dec_sum += dec_vec
        if stat_vec is not None:
            if stat_sum is None:
                stat_sum = _np.zeros_like(stat_vec, dtype=float)
            if len(stat_vec) == len(stat_sum):
                stat_sum += stat_vec

        n_seen += int(contrib)
        batch_count += 1

        if n_seen >= int(min_samples):
            break

    if (enc_sum is None) and (stat_sum is None) and (dec_sum is None):
        return

    # average the sums across batches processed
    denom = max(1, batch_count)
    enc_avg = None if enc_sum is None else (enc_sum / denom)
    dec_avg = None if dec_sum is None else (dec_sum / denom)
    stat_avg = None if stat_sum is None else (stat_sum / denom)

    rows = []
    if enc_avg is not None and len(enc_names) == enc_avg.shape[0]:
        rows += [{"group": "encoder", "feature": n, "importance": float(v)}
                 for n, v in zip(enc_names, enc_avg)]
    if dec_avg is not None and len(dec_names) == dec_avg.shape[0]:
        rows += [{"group": "decoder", "feature": n, "importance": float(v)}
                 for n, v in zip(dec_names, dec_avg)]
    if stat_avg is not None and len(stat_names) == stat_avg.shape[0]:
        rows += [{"group": "static", "feature": n, "importance": float(v)}
                 for n, v in zip(stat_names, stat_avg)]

    if not rows:
        print("[SVI] Parsed no variable importances (shape/name mismatch?).")
        return

    df = _pd.DataFrame(rows).sort_values(["group", "importance"], ascending=[True, False])
    df.to_csv(out_csv_path, index=False)
    print(f"[SVI] wrote {out_csv_path} | batches={batch_count} | aggregated_samples≈{n_seen}")

    # optional upload
    if uploader is not None:
        try:
            uploader(out_csv_path, f"{GCS_OUTPUT_PREFIX}/{_Path(out_csv_path).name}")
        except Exception as e:
            print(f"[SVI] upload failed: {e}")


        
# Data preparation
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print(
        f"[CONFIG] batch_size={BATCH_SIZE} | encoder={MAX_ENCODER_LENGTH} | epochs={MAX_EPOCHS} | "
        f"patience={EARLY_STOP_PATIENCE} | perm_len={PERM_BLOCK_SIZE} | "
        f"perm_importance={'on' if ENABLE_FEATURE_IMPORTANCE else 'off'} | fi_max_batches={FI_MAX_BATCHES} | "
        f"train_max_rows={getattr(ARGS, 'train_max_rows', None)} | val_max_rows={getattr(ARGS, 'val_max_rows', None)} | subset_mode={getattr(ARGS, 'subset_mode', 'per_asset_tail')} | "
        f"warmups={'off' if getattr(ARGS, 'disable_warmups', False) else 'on'}"
    )
    print("▶ Loading data …")
    train_df = add_time_idx(load_split(READ_PATHS[0]))
    val_df   = add_time_idx(load_split(READ_PATHS[1]))
    test_df  = add_time_idx(load_split(READ_PATHS[2]))
    # -------------------------------------------------------------------
    # Compute per‑asset median realised_vol scale (rv_scale) **once** on the TRAIN split
    # and attach it to every split. This is used only as a **fallback** for manual decode
    # if a normaliser decode is not available.
    # -------------------------------------------------------------------
    asset_scales = (
        train_df.groupby("asset", observed=True)["realised_vol"]
                .median()
                .clip(lower=1e-8)                 # guard against zeros
                .rename("rv_scale")
                .reset_index()
    )

    # Attach rv_scale without using the deprecated/invalid `inplace` kwarg
    asset_scale_map = asset_scales.set_index("asset")["rv_scale"]
    for df in (train_df, val_df, test_df):
        # map() preserves the original row order and keeps dtype float64
        df["rv_scale"] = df["asset"].map(asset_scale_map)
        # If an asset appears only in val/test, fall back to overall median
        df["rv_scale"].fillna(asset_scale_map.median(), inplace=True)
    # Global decoded-scale floor for vol (prevents QLIKE blow-ups on near-zero preds)
    try:
        EVAL_VOL_FLOOR = max(1e-8, float(asset_scales["rv_scale"].median() * 0.02))
    except Exception:
        EVAL_VOL_FLOOR = 1e-8
    print(f"[EVAL] Global vol floor (decoded) set to {EVAL_VOL_FLOOR:.6g}")

    # Add calendar features to all splits
    train_df = add_calendar_features(train_df)
    val_df   = add_calendar_features(val_df)
    test_df  = add_calendar_features(test_df)


    # Optional quick-run subsetting for speed
    _mode = getattr(ARGS, "subset_mode", "per_asset_tail")
    if getattr(ARGS, "train_max_rows", None):
        before = len(train_df)
        train_df = subset_time_series(train_df, int(ARGS.train_max_rows), mode=_mode)
        print(f"[SUBSET] TRAIN: {before} -> {len(train_df)} rows using mode='{_mode}'")
    if getattr(ARGS, "val_max_rows", None):
        before = len(val_df)
        val_df = subset_time_series(val_df, int(ARGS.val_max_rows), mode=_mode)
        print(f"[SUBSET] VAL:   {before} -> {len(val_df)} rows using mode='{_mode}'")

    # Ensure VAL/TEST only contain assets present in TRAIN and have enough history
    _val_df_raw = val_df.copy()
    _test_df_raw = test_df.copy()
    min_required = int(MAX_ENCODER_LENGTH + MAX_PRED_LENGTH)
    _val_before_rows, _val_before_assets = len(val_df), val_df["asset"].nunique()
    _test_before_rows, _test_before_assets = len(test_df), test_df["asset"].nunique()
    train_assets = set(train_df["asset"].unique())

    # Keep only assets seen in TRAIN
    val_df = val_df[val_df["asset"].isin(train_assets)]
    test_df = test_df[test_df["asset"].isin(train_assets)]

    # Drop groups with too few timesteps for at least one window
    val_df = val_df.groupby("asset", observed=True).filter(lambda g: len(g) >= min_required)
    test_df = test_df.groupby("asset", observed=True).filter(lambda g: len(g) >= min_required)

    print(
        f"[FILTER] VAL rows {_val_before_rows}→{len(val_df)} | assets {_val_before_assets}→{val_df['asset'].nunique()} (min_len={min_required})"
    )
    print(
        f"[FILTER] TEST rows {_test_before_rows}→{len(test_df)} | assets {_test_before_assets}→{test_df['asset'].nunique()} (min_len={min_required})"
    )

    # If overly strict and we filtered everything out, relax to overlap-only and let PF handle windows
    if len(val_df) == 0:
        val_df = _val_df_raw[_val_df_raw["asset"].isin(train_assets)]
        print(f"[FILTER] VAL relaxed to overlap-only: {len(val_df)} rows")
    if len(test_df) == 0:
        test_df = _test_df_raw[_test_df_raw["asset"].isin(train_assets)]
        print(f"[FILTER] TEST relaxed to overlap-only: {len(test_df)} rows")


    # -----------------------------------------------------------------------
    # Feature definitions
    # -----------------------------------------------------------------------
    static_categoricals = GROUP_ID
    static_reals: List[str] = []

    base_exclude = set(GROUP_ID + [TIME_COL, "time_idx", "rv_scale"] + TARGETS)

    all_numeric = [c for c, dt in train_df.dtypes.items()
                   if (c not in base_exclude) and pd.api.types.is_numeric_dtype(dt)]

    # Specify future-known and unknown real features
    calendar_cols = ["sin_tod", "cos_tod", "sin_dow", "cos_dow"]
    time_varying_known_reals = calendar_cols + ["Is_Weekend"]
    time_varying_unknown_reals = [c for c in all_numeric if c not in (calendar_cols + ["Is_Weekend"]) ]



    # -----------------------------------------------------------------------
    # TimeSeriesDataSets
    # -----------------------------------------------------------------------
    def build_dataset(df: pd.DataFrame, predict: bool) -> TimeSeriesDataSet:
        return TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target=TARGETS,
            group_ids=GROUP_ID,
            max_encoder_length=MAX_ENCODER_LENGTH,
            max_prediction_length=MAX_PRED_LENGTH,
            # ------------------------------------------------------------------
            # Target normalisation
            #   • realised_vol → GroupNormalizer(asinh, per‑asset scaling)
            #   • direction    → identity (classification logits)
            # ------------------------------------------------------------------
            target_normalizer = MultiNormalizer([
                GroupNormalizer(
                    groups=GROUP_ID,
                    center=False,
                    scale_by_group= True, #True
                    transformation="log1p",
                ),
                TorchNormalizer(method="identity", center=False),   # direction
            ]),
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_reals=time_varying_known_reals,   # known at prediction time
            time_varying_unknown_reals=time_varying_unknown_reals # same set, allows learning lagged targets
            ,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
            predict_mode=predict
        )

    print("▶ Building TimeSeriesDataSets …")

    training_dataset = build_dataset(train_df, predict=False)

    # Build validation/test from TRAIN template so group ID mapping and normalizer stats MATCH
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, val_df, predict=False, stop_randomization=True
    )
    test_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, test_df, predict=False, stop_randomization=True
    )
    vol_normalizer = _extract_norm_from_dataset(training_dataset)  # must be from TRAIN
    # make it available to both the model and the metrics callback
    # (optional) quick alignment check
    try:
        train_vocab = training_dataset.get_parameters()["categorical_encoders"]["asset"].classes_
        val_vocab   = validation_dataset.get_parameters()["categorical_encoders"]["asset"].classes_
        print(f"[ALIGN] val uses train encoders: {np.array_equal(train_vocab, val_vocab)}; "
            f"len(train_vocab)={len(train_vocab)} len(val_vocab)={len(val_vocab)}")
    except Exception:
        pass

    batch_size = min(BATCH_SIZE, len(training_dataset))

    # DataLoader performance knobs — prefer multi-worker, fall back to single-process
    default_workers = max(2, (os.cpu_count() or 4) - 1)
    _cli_workers = int(ARGS.num_workers) if getattr(ARGS, "num_workers", None) is not None else default_workers
    prefetch = int(getattr(ARGS, "prefetch_factor", 8))
    pin = torch.cuda.is_available()

    # Use CLI/default worker count; only disable when explicitly set to 0
    worker_cnt = max(0, _cli_workers)
    use_persist = worker_cnt > 0
    # Only pass prefetch_factor when num_workers > 0
    prefetch_kw = ({"prefetch_factor": prefetch} if worker_cnt > 0 else {})

    test_loader = test_dataset.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=worker_cnt,
        pin_memory=pin,
        persistent_workers=use_persist,
        **prefetch_kw,
    )

    train_dataloader = training_dataset.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=worker_cnt,
        persistent_workers=use_persist,
        pin_memory=pin,
        **prefetch_kw,
    )
    val_dataloader = validation_dataset.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=worker_cnt,
        persistent_workers=use_persist,
        pin_memory=pin,
        **prefetch_kw,
    )
    # --- Test dataset ---
    test_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        test_df,
        predict=False,
        stop_randomization=True,
    )

    # use the same loader knobs as train/val to avoid None / tensor-bool pitfalls
    test_dataloader = test_dataset.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=worker_cnt,
        persistent_workers=use_persist,
        pin_memory=pin,
        **prefetch_kw,
    )

    # ---- derive id→asset-name mapping for callbacks ----
    asset_vocab = (
        training_dataset.get_parameters()["categorical_encoders"]["asset"].classes_
    )
    rev_asset = {i: lbl for i, lbl in enumerate(asset_vocab)}

    vol_normalizer = _extract_norm_from_dataset(training_dataset)


    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    seed_everything(SEED, workers=True)
    # Loss and output_size for multi-target: realised_vol (quantile regression), direction (classification)
    print("▶ Building model …")
    print(f"[LR] learning_rate={LR_OVERRIDE if LR_OVERRIDE is not None else 0.0017978}")
    
    es_cb = EarlyStopping(
    monitor="val_loss",
    patience=EARLY_STOP_PATIENCE,
    mode="min",
    min_delta = 1e-4
    )



    metrics_cb = PerAssetMetrics(
        id_to_name=rev_asset,
        vol_normalizer= vol_normalizer
    )

    # If you have a custom checkpoint mirroring callback
    mirror_cb = MirrorCheckpoints()




    from pytorch_forecasting.metrics import MultiLoss
    # one-off in your data prep (TRAIN split)
    counts = train_df["direction"].value_counts()
    n_pos = counts.get(1, 1)
    n_neg = counts.get(0, 1)
    pos_weight = float(n_neg / n_pos)

    # then build the loss with:
    DIR_LOSS = LabelSmoothedBCEWithBrier(smoothing=0.03, pos_weight=pos_weight)


    FIXED_VOL_WEIGHT = 1.0
    FIXED_DIR_WEIGHT = 0.1
 

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        hidden_size=96,
        attention_head_size=4,
        dropout=0.13, #0.0833704625250354,
        hidden_continuous_size=24,
        learning_rate=(LR_OVERRIDE if LR_OVERRIDE is not None else 0.00035), #0.0019 0017978
        optimizer="AdamW",
        optimizer_params={"weight_decay": WEIGHT_DECAY},
        output_size=[7, 1],  # 7 quantiles + 1 logit
        loss=MultiLoss([VOL_LOSS, DIR_LOSS], weights=[FIXED_VOL_WEIGHT, FIXED_DIR_WEIGHT]),
        logging_metrics=[],
        log_interval=50,
        log_val_interval=10,
    )
    vol_normalizer = _extract_norm_from_dataset(training_dataset)
    tft.vol_norm = vol_normalizer          # if your LightningModule is 'tft'
    metrics_cb = PerAssetMetrics(
        id_to_name=rev_asset,
        vol_normalizer=vol_normalizer,
        max_print=10
    )

    lr_cb = LearningRateMonitor(logging_interval="step")


    best_ckpt_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename=f"tft_best_e{MAX_EPOCHS}_{RUN_SUFFIX}",
        dirpath=str(LOCAL_CKPT_DIR),
    )

  

    val_hist_csv = LOCAL_RUN_DIR / f"tft_val_history_e{MAX_EPOCHS}_{RUN_SUFFIX}.csv"
    val_hist_cb  = ValLossHistory(val_hist_csv)
    
    lr_decay_cb = EpochLRDecay(gamma=0.95, start_epoch=8) 

    # ----------------------------
    # Trainer instance
    # ----------------------------
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(LOCAL_OUTPUT_DIR),
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",      # or val_qlike_overall if you want
        save_top_k=1,            # keep best only
        save_last = True,
        auto_insert_metric_name=False,
        mode="min",              # because lower val_loss is better
    )
    trainer = Trainer(
        accelerator=ACCELERATOR,
        devices=DEVICES,
        precision=PRECISION,
        max_epochs=MAX_EPOCHS,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        num_sanity_val_steps = 0,
        logger=logger,
        callbacks=[best_ckpt_cb, TQDMProgressBar(refresh_rate=50), es_cb, metrics_cb, mirror_cb, lr_decay_cb, lr_cb, val_hist_cb] + EXTRA_CALLBACKS,
        check_val_every_n_epoch=int(ARGS.check_val_every_n_epoch),
        log_every_n_steps=int(ARGS.log_every_n_steps),
        enable_progress_bar=True,
    )


    from types import MethodType

    # Completely skip PF's internal figure creation & TensorBoard logging during validation.
    def _no_log_prediction(self, *args, **kwargs):
        # Intentionally do nothing so BaseModel.create_log won't attempt to log a Matplotlib figure.
        return

    tft.log_prediction = MethodType(_no_log_prediction, tft)

    # (Optional, extra safety) If any PF path calls plot_prediction directly, hand back a blank figure
    # so nothing touches your tensors or bf16 -> NumPy conversion.
    def _blank_plot(self, *args, **kwargs):
        import matplotlib
        matplotlib.use("Agg", force=True)  # headless-safe backend
        import matplotlib.pyplot as plt
        fig = plt.figure()
        return fig

    tft.plot_prediction = MethodType(_blank_plot, tft)
    resume_ckpt = get_resume_ckpt_path() if RESUME_ENABLED else None
    if resume_ckpt:
        print(f"↩️  Resuming from checkpoint: {resume_ckpt}")
    else:
        print("▶ Starting a fresh run (no resume)")

    # Train the model
    trainer.fit(tft, train_dataloader, val_dataloader, ckpt_path=resume_ckpt)

    # Resolve the best checkpoint
    model_for_fi = _resolve_best_model(trainer, fallback=tft)

    # Extract the same normalizer used for volatility during training
    try:
        train_vol_norm = _extract_norm_from_dataset(training_dataset)
    except Exception as e:
        print(f"[WARN] could not extract vol_norm from training dataset: {e}")
        train_vol_norm = None
    # ---- Permutation Importance (decoded) ----
if ENABLE_FEATURE_IMPORTANCE:
    fi_csv = str(LOCAL_OUTPUT_DIR / f"TFTPFI_e{MAX_EPOCHS}_{RUN_SUFFIX}.csv")
    feats = time_varying_unknown_reals.copy()
    # optional: drop calendar features from FI
    feats = [f for f in feats if f not in ("sin_tod", "cos_tod", "sin_dow", "cos_dow", "Is_Weekend")]

    # Prefer TRAIN dataset as template → guarantees same encoders & normalizers
    try:
        train_vol_norm = _extract_norm_from_dataset(training_dataset)
    except Exception:
        # fallback to validation dataset if needed
        train_vol_norm = _extract_norm_from_dataset(validation_dataset)

    run_permutation_importance(
        model=model_for_fi,
        template_ds=training_dataset,          # 👈 use TRAIN template
        base_df=val_df,
        features=feats,
        block_size=int(PERM_BLOCK_SIZE) if PERM_BLOCK_SIZE else 1,
        batch_size=256,
        max_batches=int(FI_MAX_BATCHES) if FI_MAX_BATCHES else 40,
        num_workers=4,
        prefetch=2,
        pin_memory=pin,
        vol_norm=train_vol_norm,               # 👈 ensure decoded metrics
        out_csv_path=fi_csv,
        uploader=upload_file_to_gcs,
    )
    # --- Safe plotting/logging: deep-cast any nested tensors to CPU float32 ---
    # --- Safe plotting/logging: class-level patch to handle bf16 + integer lengths robustly ---

    from pytorch_forecasting.models.base._base_model import BaseModel # type: ignore

    def _deep_cpu_float(x):
        if torch.is_tensor(x):
            # keep integer tensors as int64; cast others to float32 for matplotlib
            if x.dtype in (
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
                getattr(torch, "long", torch.int64)
            ):
                return x.detach().to(device="cpu", dtype=torch.int64)
            return x.detach().to(device="cpu", dtype=torch.float32)
        if isinstance(x, list):
            return [_deep_cpu_float(v) for v in x]
        if isinstance(x, tuple):
            casted = tuple(_deep_cpu_float(v) for v in x)
            try:
                # preserve namedtuple types
                return x.__class__(*casted)
            except Exception:
                return casted
        if isinstance(x, dict):
            return {k: _deep_cpu_float(v) for k, v in x.items()}
        if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number):
            if np.issubdtype(x.dtype, np.integer):
                return x.astype(np.int64, copy=False)
            return x.astype(np.float32, copy=False)
        return x

    def _to_numpy_int64_array(v):
        if torch.is_tensor(v):
            return v.detach().cpu().long().numpy()
        if isinstance(v, np.ndarray):
            return v.astype(np.int64, copy=False)
        if isinstance(v, (list, tuple)):
            out = []
            for el in v:
                if torch.is_tensor(el):
                    out.append(int(el.detach().cpu().item()))
                else:
                    out.append(int(el))
            return np.asarray(out, dtype=np.int64)
        if isinstance(v, (int, np.integer)):
            return np.asarray([int(v)], dtype=np.int64)
        return v  # leave unknowns as-is

    def _fix_lengths_in_x(x):
        # PF expects max() & python slicing with these; make them numpy int64 arrays
        if isinstance(x, dict):
            for key in ("encoder_lengths", "decoder_lengths"):
                if key in x and x[key] is not None:
                    x[key] = _to_numpy_int64_array(x[key])
        return x
    
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
import fsspec
import pandas as pd

print("✓ Training finished — reloading best checkpoint for test inference")

# --- Resolve best checkpoint path and load ---
best_model_path = None

# Try the explicit callback you created
try:
    if "checkpoint_callback" in globals() and getattr(checkpoint_callback, "best_model_path", ""):
        best_model_path = checkpoint_callback.best_model_path
except Exception:
    pass

# Try any ModelCheckpoint callbacks attached to the trainer
if not best_model_path:
    try:
        from lightning.pytorch.callbacks import ModelCheckpoint
        for cb in getattr(trainer, "callbacks", []):
            if isinstance(cb, ModelCheckpoint) and getattr(cb, "best_model_path", ""):
                best_model_path = cb.best_model_path
                break
    except Exception:
        pass

# Fall back to newest local ckpt
if not best_model_path:
    try:
        ckpts = sorted(LOCAL_CKPT_DIR.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if ckpts:
            best_model_path = str(ckpts[0])
    except Exception:
        pass

# Fall back to GCS (optional)
if not best_model_path and fs is not None:
    try:
        last_uri = f"{CKPT_GCS_PREFIX}/last.ckpt"
        if fs.exists(last_uri):
            dst = LOCAL_CKPT_DIR / "last.ckpt"
            with fsspec.open(last_uri, "rb") as f_in, open(dst, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            best_model_path = str(dst)
    except Exception as e:
        print(f"[WARN] Could not fetch checkpoint from GCS: {e}")

# Load the best checkpoint, or fall back to the in-memory model
if best_model_path:
    print(f"Best checkpoint (local or remote): {best_model_path}")
    try:
        best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    except Exception as e:
        print(f"[WARN] Loading best checkpoint failed ({e}); using in-memory model.")
        best_model = globals().get("tft") or globals().get("model")
else:
    print("[WARN] No checkpoint found; using in-memory model.")
    best_model = globals().get("tft") or globals().get("model")

if best_model is None:
    raise RuntimeError("No model available for testing (checkpoint and in-memory both unavailable).")

print(f"Best checkpoint (local or remote): {best_model_path}")

# Compute variable importance from TFT gating
# -----------------------------------------------------------------------
# Save Standard Variable Importance
# -----------------------------------------------------------------------
try:
    feature_names = (
        training_dataset.encoder_variables
        + training_dataset.decoder_variables
        + training_dataset.static_categoricals
    )
except Exception:
    feature_names = None

save_standard_variable_importance(
    best_model,
    val_dataloader,
    ds=validation_dataset,                                # names come from dataset
    out_csv_path=str(LOCAL_OUTPUT_DIR / "standard_variable_importance.csv"),
    uploader=upload_file_to_gcs,
    min_samples=10_000,                                   # ← your requirement
    max_batches=None                                      # or set e.g. 60 as a guard
)


# --- Run TEST loop with the best checkpoint & print decoded metrics ---
try:
    test_results = trainer.test(best_model, dataloaders=test_dataloader, verbose=True)
    print(f"Test results (trainer.test): {test_results}")
except Exception as e:
    print(f"[WARN] Test evaluation failed: {e}")

# Helpers
def _median_from_quantiles(vol_q: torch.Tensor) -> torch.Tensor:
    vol_q = torch.cummax(vol_q, dim=-1).values
    return vol_q[..., 3]

def _safe_prob(t: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t):
        return t
    try:
        if torch.isfinite(t).any() and (t.min() < 0 or t.max() > 1):
            t = torch.sigmoid(t)
    except Exception:
        t = torch.sigmoid(t)
    return torch.clamp(t, 0.0, 1.0)

@torch.no_grad()
def _collect_test_predictions(model, dl, id_to_name, vol_norm):
    model_device = next(model.parameters()).device
    assets_all, t_idx_all = [], []
    yv_all, pv_all, yd_all, pdprob_all = [], [], [], []

    for batch in dl:
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            continue
        x, y = batch[0], batch[1]
        if not isinstance(x, dict):
            continue

        # groups / time_idx
        g = x.get("groups")
        if g is None:
            g = x.get("group_ids")
        if isinstance(g, (list, tuple)):
            g = g[0] if g else None
        if torch.is_tensor(g) and g.ndim > 1 and g.size(-1) == 1:
            g = g.squeeze(-1)
        if not torch.is_tensor(g):
            continue

        t_idx = x.get("decoder_time_idx")
        if t_idx is None:
            t_idx = x.get("time_idx")
        if torch.is_tensor(t_idx) and t_idx.ndim > 1 and t_idx.size(-1) == 1:
            t_idx = t_idx.squeeze(-1)

        # true targets (encoded)
        yv, yd = None, None
        if torch.is_tensor(y):
            t = y
            if t.ndim == 3 and t.size(1) == 1:
                t = t[:, 0, :]
            if t.ndim == 2 and t.size(1) >= 1:
                yv = t[:, 0]
                yd = (t[:, 1] if t.size(1) > 1 else None)

        # forward
        x_dev = {k: (v.to(model_device, non_blocking=True) if torch.is_tensor(v)
                     else [vv.to(model_device, non_blocking=True) if torch.is_tensor(vv) else vv for vv in v]
                     if isinstance(v, (list, tuple)) else v)
                 for k, v in x.items()}
        out = model(x_dev)
        pred = getattr(out, "prediction", out)
        if isinstance(pred, dict) and "prediction" in pred:
            pred = pred["prediction"]

        # split heads
        if isinstance(pred, (list, tuple)):
            vol_q = pred[0]
            d_log = pred[1] if len(pred) > 1 else None
        else:
            t = pred
            if t.ndim >= 4 and t.size(1) == 1:
                t = t.squeeze(1)
            if t.ndim == 3 and t.size(1) == 1:
                t = t[:, 0, :]
            vol_q, d_log = t[:, : t.size(-1)-1], t[:, -1]
        p_vol = _median_from_quantiles(vol_q) if torch.is_tensor(vol_q) else vol_q

        # decode to physical scale
        y_dec = None
        if torch.is_tensor(yv):
            y_dec = safe_decode_vol(yv.unsqueeze(-1), vol_norm, g.unsqueeze(-1)).squeeze(-1)
        p_dec = safe_decode_vol(p_vol.unsqueeze(-1), vol_norm, g.unsqueeze(-1)).squeeze(-1)
        p_dec = torch.clamp(p_dec, min=1e-8)

        # direction prob
        p_dir_prob = _safe_prob(d_log) if torch.is_tensor(d_log) else None

        # append
        gn_list = g.detach().cpu().long().tolist()
        assets_all.extend([id_to_name.get(int(i), str(int(i))) for i in gn_list])

        if torch.is_tensor(t_idx):
            t_idx_all.extend(t_idx.detach().cpu().long().tolist())
        else:
            t_idx_all.extend([None] * len(gn_list))

        if y_dec is not None:
            yv_all.append(y_dec.detach().cpu())
        pv_all.append(p_dec.detach().cpu())

        if torch.is_tensor(yd):
            yd_all.append(yd.detach().cpu())
        if p_dir_prob is not None:
            pdprob_all.append(p_dir_prob.detach().cpu())

    # stack safely
    yv_list = torch.cat(yv_all).numpy().tolist() if yv_all else [None] * len(assets_all)
    pv_list = torch.cat(pv_all).numpy().tolist() if pv_all else []
    yd_list = torch.cat(yd_all).numpy().tolist() if yd_all else [None] * len(assets_all)
    pd_list = torch.cat(pdprob_all).numpy().tolist() if pdprob_all else [None] * len(assets_all)

    L = min(len(assets_all), len(t_idx_all), len(pv_list), len(yv_list), len(yd_list), len(pd_list))
    return pd.DataFrame({
        "asset":             assets_all[:L],
        "time_idx":          t_idx_all[:L],
        "realised_vol":      yv_list[:L],
        "pred_realised_vol": pv_list[:L],
        "direction":         yd_list[:L],
        "pred_direction":    pd_list[:L],
    })

# Build dataframe of predictions and compute TEST metrics on decoded scale
try:
    df_test_preds = _collect_test_predictions(
        best_model, test_dataloader, id_to_name=metrics_cb.id_to_name, vol_norm=metrics_cb.vol_norm
    )

    # Print decoded test metrics
    _y = torch.tensor([v for v in df_test_preds["realised_vol"].tolist() if v is not None], dtype=torch.float32)
    _p = torch.tensor(df_test_preds["pred_realised_vol"].tolist(), dtype=torch.float32)[: _y.numel()]
    eps = 1e-8
    mae  = torch.mean(torch.abs(_p - _y)).item()
    mse  = torch.mean((_p - _y) ** 2).item()
    rmse = mse ** 0.5
    sigma2_p = torch.clamp(torch.abs(_p), min=eps) ** 2
    sigma2_y = torch.clamp(torch.abs(_y), min=eps) ** 2
    ratio = sigma2_y / sigma2_p
    qlike = torch.mean(ratio - torch.log(ratio) - 1.0).item()

    # Direction metrics if available
    auroc_val, brier_val, acc_val = None, None, None
    if "direction" in df_test_preds.columns and "pred_direction" in df_test_preds.columns:
        pairs = [(a, b) for a, b in zip(df_test_preds["direction"], df_test_preds["pred_direction"]) if a is not None and b is not None]
        if pairs:
            yd_t = torch.tensor([a for a, _ in pairs], dtype=torch.float32)
            pd_t = torch.tensor([b for _, b in pairs], dtype=torch.float32)
            pd_t = _safe_prob(pd_t)
            acc_val   = float(((pd_t >= 0.5).int() == yd_t.int()).float().mean().item())
            brier_val = float(torch.mean((pd_t - yd_t) ** 2).item())
            try:
                from torchmetrics.classification import BinaryAUROC
                auroc_val = float(BinaryAUROC()(pd_t, yd_t).item())
            except Exception as e:
                print(f"[WARN] AUROC (test) failed: {e}")

    print(
        f"[TEST] (decoded) MAE={mae:.6f} RMSE={rmse:.6f} MSE={mse:.6f} QLIKE={qlike:.6f}"
        + (f" | ACC={acc_val:.3f}" if acc_val is not None else "")
        + (f" | Brier={brier_val:.4f}" if brier_val is not None else "")
        + (f" | AUROC={auroc_val:.3f}" if auroc_val is not None else "")
    )

    # Save parquet & upload
    pred_path = LOCAL_OUTPUT_DIR / f"tft_test_predictions_e{MAX_EPOCHS}_{RUN_SUFFIX}.parquet"
    df_test_preds.to_parquet(pred_path, index=False)
    print(f"✓ Saved TEST predictions → {pred_path}")
    try:
        upload_file_to_gcs(str(pred_path), f"{GCS_OUTPUT_PREFIX}/{pred_path.name}")
    except Exception as e:
        print(f"[WARN] Could not upload test predictions: {e}")

except Exception as e:
    print(f"[WARN] Failed to save test predictions: {e}")
    