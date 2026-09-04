"""Evaluation metrics for drug response prediction (pure numpy/scipy)."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard regression metrics.

    Args:
        y_true: true values, shape (N,) or (N, 1)
        y_pred: predicted values, same shape

    Returns:
        dict with rmse, mae, r2, pearson_r, spearman_r
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return {
            "rmse": float("nan"), "mae": float("nan"),
            "r2": float("nan"),
            "pearson_r": float("nan"), "spearman_r": float("nan"),
        }

    # RMSE
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # MAE
    mae = float(np.mean(np.abs(y_true - y_pred)))
    # R-squared
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

    # Pearson correlation
    p_r, p_p = pearsonr(y_true, y_pred)
    # Spearman correlation
    s_r, s_p = spearmanr(y_true, y_pred)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson_r": float(p_r),
        "spearman_r": float(s_r),
    }


def format_metrics(metrics: dict[str, float]) -> str:
    vals = [
        f"RMSE={metrics.get('rmse', float('nan')):.4f}",
        f"MAE={metrics.get('mae', float('nan')):.4f}",
        f"R2={metrics.get('r2', float('nan')):.4f}",
        f"Pearson={metrics.get('pearson_r', float('nan')):.4f}",
        f"Spearman={metrics.get('spearman_r', float('nan')):.4f}",
    ]
    return " ".join(vals)
