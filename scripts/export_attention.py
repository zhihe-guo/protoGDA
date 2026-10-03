#!/usr/bin/env python
"""Export cross-attention weights from one evaluation batch.

The formal model returns probe-to-atom weights when called with
``return_attention=True``. This script writes those weights for the first
sample of the first batch and, when Matplotlib is installed, a heatmap.
"""

from __future__ import annotations

import argparse
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


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _sample_weights(attn_layers: list, sample_index: int = 0) -> list[np.ndarray]:
    arrays = []
    for layer in attn_layers:
        weights = layer[sample_index]
        arrays.append(weights.detach().cpu().numpy())
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser(description="Export protoGDA cross-attention weights")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--out", default="results/attention_example")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.training.device)
    train_loader, valid_loader, test_loader, registries, meta = get_dataloaders(cfg)
    loader = valid_loader if args.split == "valid" else test_loader
    batch = _to_device(next(iter(loader)), device)

    model = CellDrugModel.from_config(
        cell_dim=meta["cell_dim"],
        cfg=cfg,
        drug_graphs=registries["drug_graphs"],
        cell_features_table=registries["cell_table"],
        drug_morgan_table=registries.get("drug_morgan_table"),
        drug_chemberta_table=registries.get("drug_chemberta_table"),
    )
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        out = model(batch, return_attention=True)
    arrays = _sample_weights(out["attention_weights"])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {f"layer_{i}": arr for i, arr in enumerate(arrays)}
    np.savez(out_dir / "attention_sample0.npz", **payload)
    print(f"Saved {len(arrays)} layers to {out_dir / 'attention_sample0.npz'}")
    print(f"Prediction for sample 0: {out['prediction'][0].item():.4f}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not installed; skipped heatmap.")
        return

    weights = arrays[-1]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    image = ax.imshow(weights, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Drug atom")
    ax.set_ylabel("Cell probe")
    ax.set_title("Cross-attention, final layer, first sample")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "attention_final_layer.png", dpi=200)
    plt.close(fig)
    print(f"Saved heatmap to {out_dir / 'attention_final_layer.png'}")
    del train_loader


if __name__ == "__main__":
    main()
