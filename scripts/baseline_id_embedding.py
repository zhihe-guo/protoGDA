"""Naive baseline: drug-id embedding + cell-id embedding + linear head.

Pure diagnostic tool — bypasses all GNN/MLP machinery to measure the upper
bound achievable from identity information alone (drug + cell biases).
Used to verify the new Trainer is healthy: should reach Pearson ~0.88 on
validation within a few epochs.
"""
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_device
from src.data.dataset import get_dataloaders
from src.training.trainer import Trainer


class IdBaseline(nn.Module):
    """drug-id embedding + cell-id embedding + linear head."""

    def __init__(self, n_drugs, n_cells, emb_dim=32):
        super().__init__()
        self.drug_emb = nn.Embedding(n_drugs, emb_dim)
        self.cell_emb = nn.Embedding(n_cells, emb_dim)
        self.head = nn.Linear(emb_dim * 2, 1)

    def forward(self, batch):
        d = self.drug_emb(batch["drug_ids"])
        c = self.cell_emb(batch["cell_ids"])
        x = torch.cat([d, c], dim=-1)
        return {"prediction": self.head(x)}

    def get_branch_params(self):
        return {
            "drug": list(self.drug_emb.parameters()),
            "cell": list(self.cell_emb.parameters()),
            "shared": list(self.head.parameters()),
        }

    def get_branch_grad_norms(self):
        norms = {}
        key_map = {"cell": "cell", "drug": "drug_gnn", "shared": "shared"}
        for branch, params in self.get_branch_params().items():
            target = key_map[branch]
            total = 0.0
            rms_sq = 0.0
            count = 0
            for p in params:
                if p.grad is not None:
                    total += p.grad.norm(2).item() ** 2
                    rms_sq += (p.grad ** 2).sum().item()
                    count += p.grad.numel()
            norms[target] = float(total ** 0.5) if total > 0 else 0.0
            norms[f"{target}_rms"] = float((rms_sq / count) ** 0.5) if count > 0 else 0.0
        if "drug_gnn_rms" in norms:
            norms["drug_rms"] = norms["drug_gnn_rms"]
        return norms


def main():
    cfg = load_config(ROOT / "config" / "mgataf_interpolation.yaml")
    cfg.training.epochs = 15
    cfg.training.early_stopping_patience = 50
    cfg.training.lr = 1.0e-3
    cfg.training.weight_decay = 1.0e-5
    cfg.training.warmup_epochs = 0
    cfg.training.checkpoint_dir = "checkpoints/baseline_id_interpolation"
    device = resolve_device(cfg.training.device)
    print(f"Device: {device}")

    tr, va, te, registries, meta = get_dataloaders(cfg)
    n_drugs = len(registries["drug_id_to_idx"])
    n_cells = len(registries["cell_id_to_idx"])
    print(f"n_drugs={n_drugs}, n_cells={n_cells}")

    model = IdBaseline(n_drugs, n_cells, emb_dim=32).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,}")

    trainer = Trainer(
        model=model, cfg=cfg, train_loader=tr, valid_loader=va, device=device,
        label_scaler=registries.get("label_scaler"),
    )
    result = trainer.train()
    print(f"\nBest val RMSE (normalized): {result['best_val_rmse']:.4f}")

    best = min(result.get("history", []),
               key=lambda e: e.get("valid", {}).get("rmse", 1e9))
    v = best.get("valid", {})
    print(f"Best epoch: valid_Pearson={v.get('pearson', 0):.4f} "
          f"valid_R2={v.get('r2', 0):.4f} valid_RMSE={v.get('rmse', 0):.4f}")
    if "rmse_raw" in v:
        print(f"  (raw: RMSE_raw={v['rmse_raw']:.4f}, Pearson_raw={v.get('pearson_raw', 0):.4f})")

    test_metrics = trainer.evaluate(te)
    print(f"\nTest: RMSE={test_metrics.get('rmse', 0):.4f} "
          f"R2={test_metrics.get('r2', 0):.4f} "
          f"Pearson={test_metrics.get('pearson', 0):.4f} "
          f"Spearman={test_metrics.get('spearman', 0):.4f}")
    if "rmse_raw" in test_metrics:
        print(f"  (raw: RMSE_raw={test_metrics['rmse_raw']:.4f}, "
              f"Pearson_raw={test_metrics.get('pearson_raw', 0):.4f})")


if __name__ == "__main__":
    main()
