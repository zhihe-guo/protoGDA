#!/usr/bin/env python
"""K-fold cross-validation for CellDrugModel on GDSC2.

Config-driven: reads k and split_by from cfg.cv section.
CLI args serve as overrides.
"""

import argparse, copy, json, random, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_device
from src.data.dataset import (
    _load_full_tdc_data, _build_registries, _get_label_column,
    DrugCellDataset, collate_drug_cell,
)
from src.models.model import CellDrugModel
from src.training.trainer import Trainer
from src.eval.metrics import compute_metrics, format_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def kfold_split(entities, k, seed):
    rng = np.random.RandomState(seed)
    ents = sorted(set(entities))
    rng.shuffle(ents)
    return [set(f.tolist()) for f in np.array_split(ents, k)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--k", type=int, default=None, help="Override config cv.k")
    parser.add_argument("--split_by", type=str, default=None,
                        choices=["interpolation", "drug_cold", "cell_cold"],
                        help="Override config cv.split_by")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cv_cfg = cfg.get("cv", {})
    k = args.k if args.k is not None else cv_cfg.get("k", 5)
    split_by = args.split_by if args.split_by is not None else cv_cfg.get("split_by", "interpolation")
    set_seed(cfg.training.seed)
    device = resolve_device(cfg.training.device)
    print(f"Device: {device}, K={k}, split_by={split_by}")

    df, smiles_series, cell_series = _load_full_tdc_data(cfg.data.dataset_name)
    print(f"Full: {len(df):,} pairs, {df['Drug_ID'].nunique()} drugs, {df['Cell_Line_ID'].nunique()} cells")

    # --- Global registries ----------------------------------------------------
    registries = _build_registries(df, smiles_series, cell_series, cfg)
    drug_id_to_idx = registries["drug_id_to_idx"]
    cell_id_to_idx = registries["cell_id_to_idx"]
    cell_dim = registries["cell_table"].shape[1]

    rng = np.random.RandomState(cfg.training.seed)
    label_col = "Y"

    # --- Fold split ----------------------------------------------------------
    if split_by == "drug_cold":
        drug_folds = kfold_split(df["Drug_ID"].unique(), k, cfg.training.seed)
        folds = [df.loc[df["Drug_ID"].isin(s)].copy() for s in drug_folds]
    elif split_by == "cell_cold":
        cell_folds = kfold_split(df["Cell_Line_ID"].unique(), k, cfg.training.seed)
        folds = [df.loc[df["Cell_Line_ID"].isin(s)].copy() for s in cell_folds]
    else:
        indices = rng.permutation(len(df))
        folds = [df.iloc[f].copy() for f in np.array_split(indices, k)]
        # Post-hoc correction for interpolation
        for fi in range(k):
            for _ in range(3):
                changed = False
                train_drugs, train_cells = set(), set()
                for fj in range(k):
                    if fj != fi:
                        train_drugs |= set(folds[fj]["Drug_ID"].values)
                        train_cells |= set(folds[fj]["Cell_Line_ID"].values)
                val_idx = set(folds[fi].index)
                orphan = {i for i in val_idx
                          if df.iloc[i]["Drug_ID"] not in train_drugs
                          or df.iloc[i]["Cell_Line_ID"] not in train_cells}
                if not orphan:
                    break
                target = 0 if fi != 0 else 1
                orphan_mask = folds[fi].index.isin(orphan)
                folds[target] = pd.concat([folds[target], folds[fi][orphan_mask]])
                folds[fi] = folds[fi][~orphan_mask]
                changed = True
                if not changed:
                    break

    for i, f in enumerate(folds):
        print(f"  Fold {i+1}: {len(f):,} pairs, {f['Drug_ID'].nunique()} drugs, {f['Cell_Line_ID'].nunique()} cells")

    # --- CV loop --------------------------------------------------------------
    fold_results, all_preds, all_labels = [], [], []

    for fi in range(k):
        print(f"\n{'='*60}\n  FOLD {fi+1}/{k}\n{'='*60}")
        val_df = folds[fi]
        train_df = pd.concat([folds[i] for i in range(k) if i != fi])
        print(f"  Train={len(train_df):,} + Valid={len(val_df):,}")

        # ID-based datasets
        for name, s in [("train", train_df), ("valid", val_df)]:
            drug_ids = np.array([drug_id_to_idx[str(s.iloc[i]["Drug_ID"])] for i in range(len(s))], dtype=np.int64)
            cell_ids = np.array([cell_id_to_idx[str(s.iloc[i]["Cell_Line_ID"])] for i in range(len(s))], dtype=np.int64)
            labels = s[label_col].values.astype(np.float32)
            ds = DrugCellDataset(drug_ids, cell_ids, labels)
            from torch.utils.data import DataLoader
            nw = cfg.data.num_workers
            loader = DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=(name == "train"),
                                num_workers=nw, pin_memory=True,
                                collate_fn=collate_drug_cell,
                                persistent_workers=nw > 0)
            if name == "train":
                tl = loader
            else:
                vl = loader

        model = CellDrugModel.from_config(
            cell_dim=cell_dim, cfg=cfg,
            drug_graphs=registries["drug_graphs"],
            cell_features_table=registries["cell_table"],
            drug_morgan_table=registries.get("drug_morgan_table"),
            drug_chemberta_table=registries.get("drug_chemberta_table"),
        )
        fold_cfg = copy.deepcopy(cfg)
        fold_cfg.training.checkpoint_dir = str(Path(cfg.training.checkpoint_dir) / f"fold_{fi+1}")
        t = Trainer(model, fold_cfg, tl, vl, device)
        result = t.train()

        best_ckpt = Path(fold_cfg.training.checkpoint_dir) / "best.pt"
        if best_ckpt.exists():
            ck = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(ck["model_state_dict"])
        vm = t.evaluate(vl)
        fold_results.append({"fold": fi + 1, "best_val_rmse": result["best_val_rmse"], "val_metrics": vm})
        print(f"  Fold {fi+1} val: {format_metrics(vm)}")

        model.eval()
        with torch.no_grad():
            for batch in vl:
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                p = model(batch)["prediction"].cpu().numpy()
                all_preds.append(p)
                all_labels.append(batch["labels"].cpu().numpy())

    rmses = [r["best_val_rmse"] for r in fold_results]
    print(f"\n{'='*60}\n  CV SUMMARY ({k}-fold)\n{'='*60}")
    for r in fold_results:
        print(f"  Fold {r['fold']}: RMSE={r['best_val_rmse']:.4f}, R2={r['val_metrics']['r2']:.4f}, Pearson={r['val_metrics']['pearson']:.4f}")

    global_m = compute_metrics(np.concatenate(all_labels), np.concatenate(all_preds))
    print(f"\n  Mean RMSE: {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")
    print(f"  Global: {format_metrics(global_m)}")

    out = Path(cfg.training.checkpoint_dir) / "cv_results.json"
    out.write_text(json.dumps({
        "k": k, "split_by": split_by,
        "fold_results": [{**r, "val_metrics": {k: float(v) for k, v in r["val_metrics"].items()}} for r in fold_results],
        "mean_rmse": float(np.mean(rmses)), "std_rmse": float(np.std(rmses)),
        "global_metrics": {k: float(v) for k, v in global_m.items()},
    }, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
