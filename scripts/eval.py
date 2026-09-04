#!/usr/bin/env python
"""Evaluate a trained CellDrugModel checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_device
from src.data.dataset import get_dataloaders
from src.eval.metrics import format_metrics
from src.models.model import CellDrugModel


def main():
    parser = argparse.ArgumentParser(description="Evaluate CellDrug model")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "config" / "default.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (default: from config eval.checkpoint)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "valid", "test"],
        default="test",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_path = args.checkpoint or cfg.eval.checkpoint
    device = resolve_device(cfg.training.device)

    _, valid_loader, test_loader, registries, meta = get_dataloaders(cfg)
    loaders = {
        "train": None,
        "valid": valid_loader,
        "test": test_loader,
    }
    if args.split == "train":
        train_loader, _, _, _, _ = get_dataloaders(cfg)
        loaders["train"] = train_loader

    loader = loaders[args.split]
    if loader is None and args.split == "train":
        raise RuntimeError("Train split loader not available.")

    model = CellDrugModel.from_config(
        cell_dim=meta["cell_dim"],
        cfg=cfg,
        drug_graphs=registries["drug_graphs"],
        cell_features_table=registries["cell_table"],
        drug_morgan_table=registries.get("drug_morgan_table"),
        drug_chemberta_table=registries.get("drug_chemberta_table"),
    )
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    from src.training.trainer import Trainer

    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=loader,
        valid_loader=loader,
        device=device,
    )
    metrics = trainer.evaluate(loader)
    print(f"Split: {args.split} | Checkpoint: {ckpt_path}")
    print(format_metrics(metrics))
    print(f"Loss: {metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
