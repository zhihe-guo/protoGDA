"""Regression metrics: RMSE, MAE, R2, Pearson, Spearman.

Pure numpy + scipy implementation. No sklearn dependency.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    diff = y_pred - y_true
    n = len(y_true)

    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))

    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    if n < 2 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        pearson = 0.0
        spearman = 0.0
    else:
        pearson = float(stats.pearsonr(y_true, y_pred)[0])
        spearman = float(stats.spearmanr(y_true, y_pred)[0])

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
    }


def format_metrics(metrics: dict[str, float]) -> str:
    return (
        f"RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  "
        f"R2={metrics['r2']:.4f}  Pearson={metrics['pearson']:.4f}  "
        f"Spearman={metrics['spearman']:.4f}"
    )
