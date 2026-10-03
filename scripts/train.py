#!/usr/bin/env python
"""Train CellDrugModel on GDSC2."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_device
from src.data.dataset import get_dataloaders
from src.models.model import CellDrugModel
from src.training.trainer import Trainer
from src.training.checkpointing import load_checkpoint


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train drug response prediction model")
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "default.yaml"),
                        help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume an epoch-boundary last.pt checkpoint after compatibility checks")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.training.seed)
    device = resolve_device(cfg.training.device)
    print(f"Using device: {device}")

    # Resolve checkpoint dir relative to project root (safe from any CWD)
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = ROOT / ckpt_dir
    cfg.training.checkpoint_dir = str(ckpt_dir)

    print("Loading data...")
    if cfg.data.get("external", {}).get("enabled", False):
        from src.data.external_data import get_external_dataloaders
        tr, va, te, registries, meta = get_external_dataloaders(cfg)
    else:
        tr, va, te, registries, meta = get_dataloaders(cfg)
    print(f"Cell feature dim: {meta['cell_dim']}")

    model = CellDrugModel.from_config(
        cell_dim=meta["cell_dim"],
        cfg=cfg,
        drug_graphs=registries["drug_graphs"],
        cell_features_table=registries["cell_table"],
        drug_morgan_table=registries.get("drug_morgan_table"),
        drug_chemberta_table=registries.get("drug_chemberta_table"),
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    if cfg.get("meta", {}).get("enabled", False):
        from src.training.meta import MetaTrainer
        trainer = MetaTrainer(
            model=model, cfg=cfg,
            train_df=registries["train_df"],
            drug_id_to_idx=registries["drug_id_to_idx"],
            cell_id_to_idx=registries["cell_id_to_idx"],
            valid_loader=va,
            test_loader=te,
            device=device,
        )
        print("Using MetaTrainer (Reptile meta-learning)")
    else:
        trainer = Trainer(
            model=model, cfg=cfg, train_loader=tr, valid_loader=va, device=device,
        )

    print("Starting training...")
    result = trainer.train(resume_path=args.resume)
    print(f"Training finished. Best validation RMSE: {result['best_val_rmse']:.4f}")

    # Load best checkpoint before test evaluation
    best_path = Path(cfg.training.checkpoint_dir) / "best.pt"
    state = load_checkpoint(best_path, map_location=device)
    print(f"Loading best checkpoint: {best_path}")
    model.load_state_dict(state["model_state_dict"])
    print(f"  Loaded epoch {state['epoch']} (val_rmse={state['trainer_state']['best_val_rmse']:.4f})")

    print("Evaluating on test set...")
    if hasattr(trainer, "evaluate_with_adapt") and cfg.get("meta", {}).get("adapt_eval", True):
        test_metrics = trainer.evaluate_with_adapt(te)
    elif hasattr(trainer, "evaluate"):
        test_metrics = trainer.evaluate(te)
    else:
        test_metrics = {"rmse": float("inf"), "note": "no evaluate method"}
    from src.eval.metrics import format_metrics
    print(f"Test: {format_metrics(test_metrics)}")


if __name__ == "__main__":
    main()
