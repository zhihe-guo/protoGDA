"""Quick diagnostic: check if MGATAF can overfit a tiny subset."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
import torch.nn.functional as F

from data import get_mgataf_dataloaders
from model import MGATAF


def main():
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_interpolation.yaml")
    cfg = OmegaConf.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ldr, valid_ldr, test_ldr, registries, meta = get_mgataf_dataloaders(cfg)
    print(f"Cell dim: {meta['cell_dim']}")

    model = MGATAF.from_config(
        meta["cell_dim"], cfg,
        drug_graphs=registries["drug_graphs"],
        cell_table=registries["cell_table"],
        drug_fp_table=registries["drug_fp_table"],
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # --- Test 1: Overfit 100 samples ---
    print("\n=== Test 1: Can model overfit 100 samples? ===")
    batch = next(iter(train_ldr))
    small_batch = {
        "drug_ids": batch["drug_ids"][:100].to(device),
        "cell_ids": batch["cell_ids"][:100].to(device),
        "labels": batch["labels"][:100].to(device),
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    model.train()
    for step in range(1000):
        optimizer.zero_grad()
        out = model(small_batch)
        loss = F.mse_loss(out["prediction"].squeeze(-1), small_batch["labels"])
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            with torch.no_grad():
                pred_np = out["prediction"].squeeze(-1).cpu().numpy()
                label_np = small_batch["labels"].cpu().numpy()
                r2 = 1 - np.sum((pred_np - label_np)**2) / max(1e-12, np.sum((label_np - label_np.mean())**2))
                corr = np.corrcoef(pred_np, label_np)[0, 1]
            print(f"  Step {step:3d}: loss={loss.item():.6f}, R2={r2:.4f}, Pearson={corr:.4f}")

    # --- Test 2: Fresh batch prediction diversity (before training) ---
    print("\n=== Test 2: Untrained prediction diversity ===")
    model2 = MGATAF.from_config(
        meta["cell_dim"], cfg,
        drug_graphs=registries["drug_graphs"],
        cell_table=registries["cell_table"],
        drug_fp_table=registries["drug_fp_table"],
    ).to(device)
    model2.eval()
    batch2 = next(iter(valid_ldr))
    big_batch = {
        "drug_ids": batch2["drug_ids"].to(device),
        "cell_ids": batch2["cell_ids"].to(device),
        "labels": batch2["labels"].to(device),
    }
    with torch.no_grad():
        out2 = model2(big_batch)
        p = out2["prediction"].squeeze(-1).cpu().numpy()
        l = big_batch["labels"].cpu().numpy()
        print(f"Predictions: min={p.min():.4f}, max={p.max():.4f}, mean={p.mean():.4f}, std={p.std():.4f}")
        print(f"Labels:      min={l.min():.4f}, max={l.max():.4f}, mean={l.mean():.4f}, std={l.std():.4f}")
        print(f"Predict mean baseline MSE: {np.mean((l.mean() - l)**2):.4f}")
        print(f"Model MSE: {np.mean((p - l)**2):.4f}")

    # --- Test 3: Check cell features after standardization ---
    print("\n=== Test 3: Cell feature analysis ===")
    ct = registries["cell_table"]
    print(f"Cell table stats: min={ct.min():.4f}, max={ct.max():.4f}, "
          f"mean={ct.mean():.4f}, std={ct.std():.4f}")
    # Check what the cell encoder does with all-zero input
    zero_input = torch.zeros(1, meta["cell_dim"], device=device)
    with torch.no_grad():
        zero_emb = model2.cell_encoder(zero_input)
        print(f"Zero-feat embedding: norm={zero_emb.norm():.4f}, values={zero_emb[0, :5].tolist()}")
    # Check what the cell encoder does with random input
    rand_input = torch.randn(1, meta["cell_dim"], device=device)
    with torch.no_grad():
        rand_emb = model2.cell_encoder(rand_input)
        print(f"Random-feat embedding: norm={rand_emb.norm():.4f}, values={rand_emb[0, :5].tolist()}")


if __name__ == "__main__":
    main()
