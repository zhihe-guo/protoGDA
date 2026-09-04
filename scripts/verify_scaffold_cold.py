#!/usr/bin/env python
"""Verify why scaffold_cold CV is lower than drug_cold CV.

Experiment A (oracle / sampling-effect test)
--------------------------------------------
Interpolation-fold models have *seen nearly all drugs* during training, so
their per-drug predictive skill is essentially constant. We run these oracle
models on the test folds of drug_cold and scaffold_cold. Any difference
between the two protocols then reflects only *which test drugs were sampled*
into each fold, not the scaffold-grouping logic itself.

Experiment B (seed / fold-partition variance)
---------------------------------------------
Re-partition the entity-level drug_cold split under several seeds and
(re)train one fold to gauge how much of the mean CV Pearson swings purely
from the random 6-fold partition (with identical grouping logic).
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_device
from src.data.dataset import _load_full_tdc_data, _build_registries, DrugCellDataset, collate_drug_cell
from src.eval.metrics import compute_metrics
from src.models.model import CellDrugModel
from scripts.cv_runner import build_scaffold_map, make_fold_split, set_seed


def build_loader(df, drug_id_to_idx, cell_id_to_idx, batch_size=2048):
    dr = np.array([drug_id_to_idx[str(x)] for x in df["Drug_ID"].values], dtype=np.int64)
    cl = np.array([cell_id_to_idx[str(x)] for x in df["Cell_Line_ID"].values], dtype=np.int64)
    lab = df["Y"].values.astype(np.float32)
    ds = DrugCellDataset(dr, cl, lab)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_drug_cell,
    )


def predict(model, loader, device):
    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {kk: v.to(device) for kk, v in batch.items() if isinstance(v, torch.Tensor)}
            p = model(batch)["prediction"].cpu().numpy()
            preds.append(p)
            labels.append(batch["labels"].cpu().numpy())
    return np.concatenate(preds), np.concatenate(labels)


def fold_pearson(y_true, y_pred):
    from scipy.stats import pearsonr
    return float(pearsonr(y_true, y_pred)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-ckpt-dir", default=str(ROOT / "checkpoints" / "cv_exp_interp_egnn_gmm_attn"))
    ap.add_argument("--config", default=str(ROOT / "config" / "exp_interp_egnn_gmm_attn.yaml"))
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "results" / "scaffold_vs_drugcold_oracle.json"))
    args = ap.parse_args()

    device = resolve_device("auto")
    cfg = load_config(args.config)

    print("Loading GDSC2 ...")
    df, smiles_series, cell_series = _load_full_tdc_data(cfg.data.dataset_name)

    # Scaffold map (shared)
    drug_smiles: dict = {}
    for did, smi in zip(df["Drug_ID"].values, smiles_series.values):
        drug_smiles.setdefault(str(did), str(smi))
    scaffold_map = build_scaffold_map(drug_smiles)

    # Registries built from the whole table so every drug has features
    registries = _build_registries(df, smiles_series, cell_series, cfg)
    didx = registries["drug_id_to_idx"]
    cidx = registries["cell_id_to_idx"]

    model = None  # lazy build once
    def get_model():
        nonlocal model
        if model is None:
            model = CellDrugModel.from_config(
                cell_dim=registries["cell_table"].shape[1],
                cfg=cfg,
                drug_graphs=registries["drug_graphs"],
                cell_features_table=registries["cell_table"],
                drug_morgan_table=registries.get("drug_morgan_table"),
                drug_chemberta_table=registries.get("drug_chemberta_table"),
            )
        return model

    results = {"drug_cold": [], "scaffold_cold": []}
    preds_store = {"drug_cold": [], "scaffold_cold": []}
    labels_store = {"drug_cold": [], "scaffold_cold": []}

    for fold_idx in range(args.k):
        oracle_dir = Path(args.oracle_ckpt_dir) / f"fold_{fold_idx+1}"
        ck = torch.load(oracle_dir / "best.pt", map_location=device, weights_only=False)
        m = get_model().to(device)
        m.load_state_dict(ck["model_state_dict"])

        for protocol in ("drug_cold", "scaffold_cold"):
            splits = make_fold_split(
                df, args.k, protocol, args.seed, fold_idx,
                scaffold_map=scaffold_map if protocol == "scaffold_cold" else None,
            )
            test_df = splits["test"]
            loader = build_loader(test_df, didx, cidx)
            preds, labels = predict(m, loader, device)
            preds_store[protocol].append(preds)
            labels_store[protocol].append(labels)
            tm = compute_metrics(labels, preds)
            results[protocol].append({
                "fold": fold_idx + 1,
                "n_samples": int(len(labels)),
                "n_drugs": int(test_df["Drug_ID"].nunique()),
                "n_cells": int(test_df["Cell_Line_ID"].nunique()),
                "y_std": float(np.std(labels)),
                "y_min": float(np.min(labels)),
                "y_max": float(np.max(labels)),
                "pearson": tm["pearson_r"],
                "spearman": tm["spearman_r"],
                "rmse": tm["rmse"],
                "test_drug_ids": [str(x) for x in test_df["Drug_ID"].values],
            })

    summary = {}
    for protocol, folds in results.items():
        pears = [f["pearson"] for f in folds]
        all_p = np.concatenate(preds_store[protocol])
        all_l = np.concatenate(labels_store[protocol])
        from scipy.stats import pearsonr
        pooled_p = pearsonr(all_l, all_p)[0]
        summary[protocol] = {
            "per_fold_pearson": [round(p, 4) for p in pears],
            "mean_pearson": float(np.mean(pears)),
            "std_pearson": float(np.std(pears)),
            "pooled_pearson": float(pooled_p),
            "per_fold_n_drugs": [f["n_drugs"] for f in folds],
            "per_fold_y_std": [round(f["y_std"], 3) for f in folds],
        }
        out_proto = Path(args.out).parent / f"oracle_{protocol}_preds.npz"
        np.savez(out_proto, preds=all_p, labels=all_l)

    print(json.dumps(summary, indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"summary": summary, "detail": results}, fh, indent=2, ensure_ascii=False)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
