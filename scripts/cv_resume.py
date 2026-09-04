#!/usr/bin/env python
"""Generic 6-fold CV runner with per-fold resume.

Folds whose best.pt already exists are reused (inference only). New folds are
trained with the config's training settings. Writes a standardized
cv_results.json in the same schema as cv_runner.py.

Used to (a) run multi-seed drug_cold sweeps, (b) resume an interrupted ablation
CV without re-training completed folds.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_device
from src.eval.metrics import format_metrics
from scripts.cv_runner import run_cellquery, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True,
                    help="Ablation config under config/ (must set split_mode + model switches)")
    ap.add_argument("--seed", type=int, default=None, help="Override training seed")
    ap.add_argument("--split_by", type=str, default=None,
                    choices=["interpolation", "drug_cold", "cell_cold", "scaffold_cold"])
    ap.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    ap.add_argument("--patience", type=int, default=None,
                    help="Override early_stopping_patience (use a large value e.g. 999 "
                         "to disable early stopping / run full epochs)")
    ap.add_argument("--checkpoint-metric", choices=["rmse", "pearson_r"], default=None,
                    help="Validation metric used to select best.pt")
    ap.add_argument("--out", type=str, default=None, help="Override output dir")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--force-retrain", action="store_true",
                    help="Retrain all folds even if best.pt exists")
    args = ap.parse_args()

    cfg = load_config(str(ROOT / args.config))
    if args.seed is not None:
        cfg.training.seed = args.seed
    if args.split_by is not None:
        cfg.cv.split_by = args.split_by
        cfg.data.split_mode = args.split_by
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.patience is not None:
        cfg.training.early_stopping_patience = args.patience
    if args.checkpoint_metric is not None:
        cfg.training.checkpoint_metric = args.checkpoint_metric

    out_root = Path(args.out) if args.out else Path(cfg.training.checkpoint_dir)
    out_root = out_root if out_root.is_absolute() else ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    cfg.training.checkpoint_dir = str(out_root)
    cfg.cv.k = args.k
    device = resolve_device(cfg.training.device)

    print(f"config={args.config}  split_by={cfg.cv.split_by}  seed={cfg.training.seed}  "
          f"epochs={cfg.training.epochs}  out={out_root}", flush=True)

    results = []
    all_preds, all_labels, all_drug_ids = [], [], []
    for fi in range(args.k):
        fold_dir = out_root / f"fold_{fi+1}"
        reuse = not args.force_retrain and (fold_dir / "best.pt").exists()
        print(f"\n=== FOLD {fi+1}/{args.k} (reuse={reuse}) ===", flush=True)
        # Keep data splits keyed to cfg.training.seed, but make initialization,
        # dropout, and loader shuffling reproducible independently per fold.
        set_seed(cfg.training.seed + fi)
        r = run_cellquery(cfg, None, fi, device, out_root, skip_train=reuse)
        results.append(r["test_metrics"]["pearson_r"])
        all_preds.append(r["test_preds"])
        all_labels.append(r["test_labels"])
        all_drug_ids.append(np.asarray(r["test_drug_ids"]))
        print(f"  fold {fi+1} pearson={r['test_metrics']['pearson_r']:.4f}", flush=True)

    mean, std = float(np.mean(results)), float(np.std(results))
    print(f"\n{cfg.cv.split_by} CV pearson = {mean:.4f} +/- {std:.4f}  "
          f"per-fold: {[round(v, 4) for v in results]}")

    from scripts.cv_runner import compute_drug_level_metrics
    from src.eval.metrics import compute_metrics
    pooled = compute_metrics(np.concatenate(all_labels), np.concatenate(all_preds))
    drug_level = compute_drug_level_metrics(
        np.concatenate(all_drug_ids), np.concatenate(all_preds), np.concatenate(all_labels))

    np.savez(out_root / "cv_predictions.npz",
             drug_ids=np.concatenate(all_drug_ids),
             preds=np.concatenate(all_preds),
             labels=np.concatenate(all_labels))

    (out_root / "cv_results.json").write_text(json.dumps({
        "config": args.config, "split_by": cfg.cv.split_by, "k": args.k,
        "seed": cfg.training.seed,
        "fold_results": [{"fold": i + 1, "pearson": round(v, 4)} for i, v in enumerate(results)],
        "mean_std": {"pearson_r": {"mean": round(mean, 4), "std": round(std, 4)}},
        "pooled_metrics": {k: float(v) for k, v in pooled.items()},
        "drug_level_metrics": {k: float(v) for k, v in drug_level.items()},
    }, indent=2))
    print(f"\nSaved -> {out_root / 'cv_results.json'}")
    print(f"Pooled: {format_metrics(pooled)}")
    print(f"Drug-level: {format_metrics(drug_level)}")


if __name__ == "__main__":
    main()
