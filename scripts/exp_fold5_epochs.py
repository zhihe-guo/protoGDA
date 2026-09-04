#!/usr/bin/env python
"""Decisive experiment: is the scaffold_cold fold collapse due to under-training?

Trains one scaffold_cold fold (default: the worst one, fold 5) for the full
configured 60 epochs with early stopping disabled, and records the COLD-START
TEST-set Pearson after EVERY epoch.

This answers directly:
  (a) Does test-set ranking ability improve after the epoch where the
      original run early-stopped?  -> if yes, under-training / checkpoint
      selection is the culprit.
  (b) What does the best achievable test Pearson look like over 60 epochs?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_device
from src.data.dataset import _load_full_tdc_data, _build_registries, DrugCellDataset, collate_drug_cell
from src.models.model import CellDrugModel
from src.training.trainer import Trainer
from src.eval.metrics import compute_metrics
from scripts.cv_runner import make_fold_split, build_scaffold_map
from torch.utils.data import DataLoader


def build_loader(sub_df, drug_id_to_idx, cell_id_to_idx, cfg, shuffle):
    dr = np.array([drug_id_to_idx[str(sub_df.iloc[i]["Drug_ID"])] for i in range(len(sub_df))], dtype=np.int64)
    cl = np.array([cell_id_to_idx[str(sub_df.iloc[i]["Cell_Line_ID"])] for i in range(len(sub_df))], dtype=np.int64)
    lab = sub_df["Y"].values.astype(np.float32)
    ds = DrugCellDataset(dr, cl, lab)
    nw = cfg.data.num_workers
    return DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=shuffle,
                      num_workers=nw, pin_memory=True, collate_fn=collate_drug_cell,
                      persistent_workers=nw > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=5, help="1-based fold index to retrain")
    args = ap.parse_args()

    cfg = load_config(str(ROOT / "config/cellquery_scaffold_cold.yaml"))
    cfg.cv.k = 6  # must match the original 6-fold CV (config default is 5!)
    cfg.training.epochs = 60
    cfg.training.early_stopping_patience = 100  # never early-stop; run the full curve
    cfg.training.checkpoint_dir = str(ROOT / f"checkpoints/exp_fold{args.fold}_60ep")
    device = resolve_device(cfg.training.device)
    fold_idx = args.fold - 1  # 0-based

    df, smiles_series, cell_series = _load_full_tdc_data(cfg.data.dataset_name)
    drug_smiles: dict = {}
    for did, smi in zip(df["Drug_ID"].values, smiles_series.values):
        drug_smiles.setdefault(str(did), str(smi))
    scaffold_map = build_scaffold_map(drug_smiles)
    splits = make_fold_split(df, cfg.cv.k, cfg.cv.split_by, cfg.training.seed, fold_idx,
                             scaffold_map=scaffold_map)
    train_df, valid_df, test_df = splits["train"], splits["valid"], splits["test"]
    print(f"train={len(train_df)} valid={len(valid_df)} test={len(test_df)} "
          f"test_drugs={test_df['Drug_ID'].nunique()}")

    registries = _build_registries(df, smiles_series, cell_series, cfg,
                                   train_cell_ids=set(train_df["Cell_Line_ID"].unique()))
    d2i = registries["drug_id_to_idx"]
    c2i = registries["cell_id_to_idx"]
    cell_dim = registries["cell_table"].shape[1]

    tr = build_loader(train_df, d2i, c2i, cfg, True)
    va = build_loader(valid_df, d2i, c2i, cfg, False)
    te = build_loader(test_df, d2i, c2i, cfg, False)

    model = CellDrugModel.from_config(
        cell_dim=cell_dim, cfg=cfg,
        drug_graphs=registries["drug_graphs"],
        cell_features_table=registries["cell_table"],
        drug_morgan_table=registries.get("drug_morgan_table"),
        drug_chemberta_table=registries.get("drug_chemberta_table"),
    )
    trainer = Trainer(model, cfg, tr, va, device)

    curve = []
    for epoch in range(1, cfg.training.epochs + 1):
        tm = trainer.train_epoch(epoch)
        vm = trainer.evaluate(va)          # warm validation (as original protocol)
        te_m = trainer.evaluate(te)        # COLD-START TEST fold (the key metric)

        # replicate original best.pt selection (min valid RMSE)
        is_best = vm["rmse"] < trainer.best_val_rmse
        if is_best:
            trainer.best_val_rmse = vm["rmse"]
            trainer.best_epoch = epoch
        trainer.save_checkpoint(epoch, is_best=is_best)

        curve.append({
            "epoch": epoch,
            "train_loss": round(float(tm["loss"]), 4),
            "valid_rmse": round(float(vm["rmse"]), 4),
            "valid_pearson": round(float(vm["pearson_r"]), 4),
            "TEST_pearson": round(float(te_m["pearson_r"]), 4),
            "TEST_rmse": round(float(te_m["rmse"]), 4),
        })
        if epoch % 5 == 0 or epoch <= 3:
            print(f"ep{epoch:02d} | train_loss={tm['loss']:.4f} | "
                  f"valid rmse={vm['rmse']:.4f} pear={vm['pearson_r']:.3f} | "
                  f"TEST pear={te_m['pearson_r']:.4f} rmse={te_m['rmse']:.4f}",
                  flush=True)

    # ---- post-hoc selection analysis ----
    best_rmse_ep = min(curve, key=lambda c: c["valid_rmse"])
    best_test_ep = max(curve, key=lambda c: c["TEST_pearson"])
    last = curve[-1]

    # evaluate the RMSE-selected checkpoint (what the original protocol reports)
    ck = torch.load(Path(cfg.training.checkpoint_dir) / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    rmse_sel = trainer.evaluate(te)

    report = {
        "fold": args.fold, "seed": cfg.training.seed, "epochs_run": len(curve),
        "original_reported_test_pearson": -0.0528 if args.fold == 5 else None,
        "rmse_selected_checkpoint": {"epoch": best_rmse_ep["epoch"],
                                     "valid_rmse": best_rmse_ep["valid_rmse"],
                                     "TEST_pearson": round(float(rmse_sel["pearson_r"]), 4)},
        "last_epoch_checkpoint": {"epoch": last["epoch"],
                                  "TEST_pearson": last["TEST_pearson"]},
        "oracle_best_test_pearson": {"epoch": best_test_ep["epoch"],
                                     "TEST_pearson": best_test_ep["TEST_pearson"]},
        "valid_pearson_best_test": max((c["TEST_pearson"] for c in curve)),
        "curve": curve,
    }
    out = ROOT / f"results/exp_fold{args.fold}_60ep.json"
    out.write_text(json.dumps(report, indent=2))
    print("\n=== SUMMARY ===")
    print(f"RMSE-selected (original protocol): ep{best_rmse_ep['epoch']} -> TEST pearson {rmse_sel['pearson_r']:.4f}")
    print(f"Last epoch (ep{last['epoch']}): TEST pearson {last['TEST_pearson']}")
    print(f"Best-ever TEST pearson: ep{best_test_ep['epoch']} = {best_test_ep['TEST_pearson']}")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
