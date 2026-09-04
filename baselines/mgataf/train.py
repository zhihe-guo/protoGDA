#!/usr/bin/env python
"""Train MGATAF baseline (completely self-contained, zero src/ dependency)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Make this directory importable for local modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import load_config, resolve_device
from data import get_mgataf_dataloaders
from model import MGATAF
from trainer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train MGATAF")
    parser.add_argument("--config", type=str, default="config_interpolation.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.training.seed)
    device = resolve_device(cfg.training.device)
    print(f"Using device: {device}")

    print("Loading data...")
    tr, va, te, registries, meta = get_mgataf_dataloaders(cfg)
    print(f"Cell feature dim: {meta['cell_dim']}")

    model = MGATAF.from_config(
        cell_dim=meta["cell_dim"],
        cfg=cfg,
        drug_graphs=registries["drug_graphs"],
        cell_table=registries["cell_table"],
        drug_fp_table=registries["drug_fp_table"],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    trainer = Trainer(
        model=model, cfg=cfg, train_loader=tr, valid_loader=va, device=device,
        label_scaler=registries.get("label_scaler"),
    )

    print("Starting training...")
    result = trainer.train()
    print(f"Training finished. Best validation RMSE: {result['best_val_rmse']:.4f}")

    # Load best checkpoint
    best_path = Path(cfg.training.checkpoint_dir) / "best.pt"
    if best_path.exists():
        print(f"Loading best checkpoint: {best_path}")
        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        print(f"  Loaded epoch {state['epoch']} (val_rmse={state['best_val_rmse']:.4f})")

    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(te)
    from metrics import format_metrics
    print(f"Test: {format_metrics(test_metrics)}")
    if "rmse_raw" in test_metrics:
        print(f"  (original log10(IC50) scale: RMSE_raw={test_metrics['rmse_raw']:.4f}, "
              f"PCC_raw={test_metrics.get('pearson_raw', 0.0):.4f})")

    summary = {
        "model": "MGATAF",
        "best_epoch": state["epoch"] if best_path.exists() else trainer.best_epoch,
        "best_val_rmse": trainer.best_val_rmse,
        "test": test_metrics,
    }
    summary_path = Path(cfg.training.checkpoint_dir) / "summary.json"
    tmp_path = summary_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    tmp_path.replace(summary_path)


if __name__ == "__main__":
    main()
