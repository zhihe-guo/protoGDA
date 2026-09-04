#!/usr/bin/env python
"""Measure wall-clock time for one drug_cold fold (60 epochs) to decide
whether a multi-seed CV sweep is affordable."""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_device
from scripts.cv_runner import run_cellquery


def main():
    cfg = load_config(str(ROOT / "config" / "cellquery_drug_cold.yaml"))
    cfg.training.seed = 7  # different from the official 42
    cfg.cv.k = 6
    cfg.cv.split_by = "drug_cold"
    out = ROOT / "results" / "seed_var" / "timing"
    device = resolve_device(cfg.training.device)
    t0 = time.time()
    r = run_cellquery(cfg, None, 0, device, out, skip_train=False)
    dt = time.time() - t0
    print(f"\nfold1 pearson={r['test_metrics']['pearson_r']:.4f} wall={dt/60:.1f} min")


if __name__ == "__main__":
    main()
