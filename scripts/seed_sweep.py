#!/usr/bin/env python
"""Multi-seed drug_cold CV sweep.

Re-runs the entity-level drug_cold 6-fold CV under several seeds to measure
the range of the CV mean that is *purely* caused by random fold partitioning
(same grouping logic). If seed=42 gives 0.531 while another seed gives ~0.33,
the scaffold_cold value (0.329) is just one draw from the same random
distribution, i.e. the drop is a sampling/random event, not the scaffold
grouping logic.

Folds whose best.pt already exists are reused (no re-training).
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
from scripts.cv_runner import run_cellquery


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    cfg = load_config(str(ROOT / "config" / "cellquery_drug_cold.yaml"))
    cfg.training.seed = args.seed
    cfg.cv.k = args.k
    cfg.cv.split_by = "drug_cold"
    cfg.training.checkpoint_dir = str(ROOT / "results" / "seed_var" / f"drugcold_seed{args.seed}")

    out_root = Path(cfg.training.checkpoint_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.training.device)

    results = []
    for fi in range(args.k):
        fold_dir = out_root / f"fold_{fi+1}"
        reuse = bool((fold_dir / "best.pt").exists())
        print(f"\n=== FOLD {fi+1}/{args.k} (seed={args.seed}, reuse={reuse}) ===", flush=True)
        r = run_cellquery(cfg, None, fi, device, out_root, skip_train=reuse)
        results.append(r["test_metrics"]["pearson_r"])
        print(f"  fold {fi+1} pearson={r['test_metrics']['pearson_r']:.4f}", flush=True)

    mean, std = float(np.mean(results)), float(np.std(results))
    print(f"\nseed={args.seed}  drug_cold CV pearson = {mean:.4f} +/- {std:.4f}  "
          f"per-fold: {[round(v, 4) for v in results]}")

    (out_root / "cv_results.json").write_text(json.dumps({
        "seed": args.seed, "split_by": "drug_cold", "k": args.k,
        "per_fold_pearson": [round(v, 4) for v in results],
        "mean": round(mean, 4), "std": round(std, 4),
    }, indent=2))


if __name__ == "__main__":
    main()
