#!/usr/bin/env python
"""Train GraphDRP baseline (completely self-contained, zero src/ dependency)."""

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
from data import get_graphdrp_dataloaders
from model import GraphDRP
from trainer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train GraphDRP")
    parser.add_argument("--config", type=str, default="config_interpolation.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.training.seed)
    device = resolve_device(cfg.training.device)
    print(f"Using device: {device}")

    print("Loading data...")
    tr, va, te, registries, meta = get_graphdrp_dataloaders(cfg)
    print(f"Cell feature dim: {meta['cell_dim']}")

    model = GraphDRP.from_config(
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


if __name__ == "__main__":
    main()
