#!/usr/bin/env python
"""Train CANDELA baseline (completely self-contained, zero src/ dependency)."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Make this directory importable for local modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import load_config, resolve_device
from data import get_candela_dataloaders
from model import CANDELA
from trainer import Trainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train CANDELA")
    parser.add_argument("--config", type=str, default="config_interpolation.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.training.seed)
    device = resolve_device(cfg.training.device)
    print(f"Using device: {device}")

    # Resolve checkpoint dir relative to project root
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = PROJECT_ROOT / ckpt_dir
    cfg.training.checkpoint_dir = str(ckpt_dir)
    print(f"Checkpoint dir: {ckpt_dir}")

    print("Loading data...")
    tr, va, te, registries, meta = get_candela_dataloaders(cfg)
    print(f"Cell feature dim: {meta['cell_dim']}")

    model = CANDELA.from_config(
        cell_dim=meta["cell_dim"],
        cfg=cfg,
        drug_graphs=registries["drug_graphs"],
        cell_table=registries["cell_table"],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    trainer = Trainer(
        model=model, cfg=cfg, train_loader=tr, valid_loader=va, device=device,
    )

    print("Starting training...")
    result = trainer.train()
    print(f"Training finished. Best validation RMSE: {result['best_val_rmse']:.4f}")

    # Load best checkpoint
    best_path = ckpt_dir / "best.pt"
    if best_path.exists():
        print(f"Loading best checkpoint: {best_path}")
        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        print(f"  Loaded epoch {state['epoch']} (val_rmse={state['best_val_rmse']:.4f})")

    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(te)
    from metrics import format_metrics
    print(f"Test: {format_metrics(test_metrics)}")
    if "gamma_ratio" in test_metrics:
        print(f"  Score decomposition: alpha={test_metrics.get('alpha_mean', 0):.4f}, "
              f"beta={test_metrics.get('beta_mean', 0):.4f}, "
              f"gamma_ratio={test_metrics.get('gamma_ratio', 0):.3f}")

    # Append test results to summary.json
    summary_path = ckpt_dir / "summary.json"
    if summary_path.exists():
        import json as _json
        with open(summary_path) as f:
            summary = _json.load(f)
        summary["test"] = test_metrics
        with open(summary_path, "w") as f:
            _json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
