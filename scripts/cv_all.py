#!/usr/bin/env python
"""6-fold cross-validation for CellQuery / CANDELA / MGATAF on GDSC2.

Each model uses its own data pipeline to build per-fold train/valid loaders:
  - CellQuery: src.data.dataset (TDC, 158-dim atoms + pharm GMM)
  - CANDELA:   baselines/candela (gdsc2.pkl, 78-dim atoms)
  - MGATAF:    baselines/mgataf  (TDC, 78-dim atoms + Morgan, label-scaled)

Fold splits are entity-based (drug_cold / cell_cold) or random (interpolation),
using the same seed across models for comparability.

Usage:
    python scripts/cv_all.py --model cellquery --split_by interpolation --k 6
    python scripts/cv_all.py --model candela --split_by drug_cold --k 6
    python scripts/cv_all.py --model mgataf --split_by cell_cold --k 6
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
from src.eval.metrics import compute_metrics, format_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def kfold_entities(entities, k: int, seed: int):
    rng = np.random.RandomState(seed)
    ents = sorted(set(entities))
    rng.shuffle(ents)
    return [set(f.tolist()) for f in np.array_split(ents, k)]


def make_folds(df: pd.DataFrame, k: int, split_by: str, seed: int) -> list[pd.DataFrame]:
    """Return k fold DataFrames (each is the validation fold of its entities)."""
    if split_by == "drug_cold":
        folds_e = kfold_entities(df["Drug_ID"].unique(), k, seed)
        return [df.loc[df["Drug_ID"].isin(s)].copy() for s in folds_e]
    if split_by == "cell_cold":
        folds_e = kfold_entities(df["Cell_Line_ID"].unique(), k, seed)
        return [df.loc[df["Cell_Line_ID"].isin(s)].copy() for s in folds_e]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
    return [df.iloc[f].copy() for f in np.array_split(idx, k)]


# ---------------------------------------------------------------------------
# Per-model fold dataloader builders
# ---------------------------------------------------------------------------

def _cellquery_fold_loaders(cfg, df, smiles_series, cell_series, train_df, valid_df):
    """Build CellQuery train/valid loaders for a fold using src pipeline internals.

    Registries (drug graphs, cell table, morgan, chemberta, pharm) are rebuilt
    per fold so that cell-feature normalization/PCA fit on this fold's train
    cells only (no leakage across folds).
    """
    from src.data.dataset import _build_registries, _get_label_column, DrugCellDataset, collate_drug_cell
    from torch.utils.data import DataLoader

    train_cell_ids = set(train_df["Cell_Line_ID"].values)
    registries = _build_registries(df, smiles_series, cell_series, cfg, train_cell_ids=train_cell_ids)
    drug_id_to_idx = registries["drug_id_to_idx"]
    cell_id_to_idx = registries["cell_id_to_idx"]
    label_col = _get_label_column(train_df)

    loaders = {}
    for name, s in [("train", train_df), ("valid", valid_df)]:
        drug_ids = np.array([drug_id_to_idx[str(s.iloc[i]["Drug_ID"])] for i in range(len(s))], dtype=np.int64)
        cell_ids = np.array([cell_id_to_idx[str(s.iloc[i]["Cell_Line_ID"])] for i in range(len(s))], dtype=np.int64)
        labels = s[label_col].values.astype(np.float32)
        ds = DrugCellDataset(drug_ids, cell_ids, labels)
        nw = cfg.data.num_workers
        loaders[name] = DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=(name == "train"),
                                   num_workers=nw, pin_memory=True, collate_fn=collate_drug_cell,
                                   persistent_workers=nw > 0)
    return loaders["train"], loaders["valid"], registries


def _candela_fold_loaders(cfg, df, train_df, valid_df):
    """Build CANDELA train/valid loaders for a fold (gdsc2.pkl based)."""
    sys.path.insert(0, str(ROOT / "baselines" / "candela"))
    from candela.data import get_candela_dataloaders
    tr, va, _te, registries, _meta = get_candela_dataloaders(
        cfg, train_df=train_df, valid_df=valid_df,
    )
    return tr, va, registries


def _mgataf_fold_loaders(cfg, df, smiles_series, cell_series, train_df, valid_df):
    """Build MGATAF train/valid loaders for a fold (TDC based)."""
    sys.path.insert(0, str(ROOT / "baselines" / "mgataf"))
    from mgataf.data import get_mgataf_dataloaders
    tr, va, _te, registries, _meta = get_mgataf_dataloaders(
        cfg, train_df=train_df, valid_df=valid_df,
    )
    return tr, va, registries


def _cellquery_model(cfg, registries, meta):
    from src.models.model import CellDrugModel
    return CellDrugModel.from_config(
        cell_dim=meta["cell_dim"], cfg=cfg,
        drug_graphs=registries["drug_graphs"],
        cell_features_table=registries["cell_table"],
        drug_morgan_table=registries.get("drug_morgan_table"),
        drug_chemberta_table=registries.get("drug_chemberta_table"),
    )


def _candela_model(cfg, registries, meta):
    from candela.model import CANDELA
    return CANDELA.from_config(
        cell_dim=meta["cell_dim"], cfg=cfg,
        drug_graphs=registries["drug_graphs"],
        cell_table=registries["cell_table"],
    )


def _mgataf_model(cfg, registries, meta):
    from mgataf.model import MGATAF
    return MGATAF.from_config(
        cell_dim=meta["cell_dim"], cfg=cfg,
        drug_graphs=registries["drug_graphs"],
        cell_table=registries["cell_table"],
        drug_fp_table=registries["drug_fp_table"],
    )


def _cellquery_trainer(model, cfg, tl, vl, device, registries):
    from src.training.trainer import Trainer
    return Trainer(model=model, cfg=cfg, train_loader=tl, valid_loader=vl, device=device)


def _candela_trainer(model, cfg, tl, vl, device, registries):
    from candela.trainer import Trainer
    return Trainer(model=model, cfg=cfg, train_loader=tl, valid_loader=vl, device=device)


def _mgataf_trainer(model, cfg, tl, vl, device, registries):
    from mgataf.trainer import Trainer
    return Trainer(model=model, cfg=cfg, train_loader=tl, valid_loader=vl, device=device,
                   label_scaler=registries.get("label_scaler"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["cellquery", "candela", "mgataf"])
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML. Defaults to cellquery/exp config for CellQuery, "
                             "or baseline config_<split>.yaml.")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--split_by", type=str, required=True,
                        choices=["interpolation", "drug_cold", "cell_cold"])
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs (default: use config value). "
                             "Use to match official-report epoch budgets.")
    args = parser.parse_args()

    set_seed(42)
    device = resolve_device("auto")
    print(f"Device: {device}, Model={args.model}, K={args.k}, split_by={args.split_by}")

    if args.config is None:
        if args.model == "cellquery":
            args.config = str(ROOT / "config" / f"cellquery_{args.split_by}.yaml")
        else:
            args.config = str(ROOT / "baselines" / args.model / f"config_{args.split_by}.yaml")
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
        print(f"Overrode epochs -> {args.epochs}")

    # --- Load full GDSC2 data (per model source) ---
    if args.model in ("cellquery", "mgataf"):
        from tdc.multi_pred import DrugRes
        data_obj = DrugRes(name="GDSC2")
        df = pd.DataFrame({
            "Drug_ID": data_obj.entity1_idx.values,
            "Cell_Line_ID": data_obj.entity2_idx.values,
            "Y": data_obj.y.values,
        })
        smiles_series = data_obj.entity1
        cell_series = data_obj.entity2
    else:  # candela
        raw_df = pd.read_pickle(str(ROOT / "data" / "gdsc2.pkl"))
        raw_df = raw_df.rename(columns={"ID1": "Drug_Name", "ID2": "Cell_Name", "Y": "Y"})
        df = pd.DataFrame({
            "Drug_ID": raw_df["Drug_Name"].values,
            "Cell_Line_ID": raw_df["Cell_Name"].values,
            "Y": raw_df["Y"].values,
        })
        smiles_series = raw_df["X1"]
        cell_series = None
    print(f"Full: {len(df):,} pairs, {df['Drug_ID'].nunique()} drugs, {df['Cell_Line_ID'].nunique()} cells")

    folds = make_folds(df, args.k, args.split_by, seed=42)
    for i, f in enumerate(folds):
        print(f"  Fold {i+1}: {len(f):,} pairs, {f['Drug_ID'].nunique()} drugs, {f['Cell_Line_ID'].nunique()} cells")

    fold_results, all_preds, all_labels = [], [], []
    for fi in range(args.k):
        print(f"\n{'='*60}\n  FOLD {fi+1}/{args.k}\n{'='*60}")
        val_df = folds[fi]
        train_df = pd.concat([folds[i] for i in range(args.k) if i != fi])

        if args.model == "cellquery":
            tl, vl, registries = _cellquery_fold_loaders(
                cfg, df, smiles_series, cell_series, train_df, val_df)
        elif args.model == "candela":
            tl, vl, registries = _candela_fold_loaders(cfg, df, train_df, val_df)
        else:
            tl, vl, registries = _mgataf_fold_loaders(
                cfg, df, smiles_series, cell_series, train_df, val_df)

        cell_dim = registries["cell_table"].shape[1]
        meta = {"cell_dim": cell_dim}

        if args.model == "cellquery":
            model = _cellquery_model(cfg, registries, meta)
        elif args.model == "candela":
            model = _candela_model(cfg, registries, meta)
        else:
            model = _mgataf_model(cfg, registries, meta)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,}")

        fold_cfg = copy.deepcopy(cfg)
        cv_root = ROOT / "checkpoints" / f"cv_{args.model}_{args.split_by}"
        fold_ckpt = cv_root / f"fold_{fi+1}"
        fold_cfg.training.checkpoint_dir = str(fold_ckpt)

        if args.model == "cellquery":
            t = _cellquery_trainer(model, fold_cfg, tl, vl, device, registries)
        elif args.model == "candela":
            t = _candela_trainer(model, fold_cfg, tl, vl, device, registries)
        else:
            t = _mgataf_trainer(model, fold_cfg, tl, vl, device, registries)

        result = t.train()

        best_ckpt = fold_ckpt / "best.pt"
        if best_ckpt.exists():
            ck = torch.load(best_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ck["model_state_dict"])
        vm = t.evaluate(vl)
        fold_results.append({"fold": fi + 1, "best_val_rmse": result["best_val_rmse"], "val_metrics": vm})
        print(f"  Fold {fi+1} val: {format_metrics(vm)}")

        model.eval()
        with torch.no_grad():
            for batch in vl:
                batch = {kk: v.to(device) for kk, v in batch.items() if isinstance(v, torch.Tensor)}
                p = model(batch)["prediction"].cpu().numpy()
                all_preds.append(p)
                all_labels.append(batch["labels"].cpu().numpy())

    rmses = [r["best_val_rmse"] for r in fold_results]
    print(f"\n{'='*60}\n  CV SUMMARY ({args.k}-fold)\n{'='*60}")
    for r in fold_results:
        m = r["val_metrics"]
        print(f"  Fold {r['fold']}: RMSE={m.get('rmse', m.get('best_val_rmse', float('nan'))):.4f}, "
              f"Pearson={m.get('pearson', m.get('pearson_r', float('nan'))):.4f}")

    global_m = compute_metrics(np.concatenate(all_labels), np.concatenate(all_preds))
    print(f"\n  Mean RMSE: {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")
    print(f"  Global: {format_metrics(global_m)}")

    out = cv_root / "cv_results.json"
    out.write_text(json.dumps({
        "model": args.model, "k": args.k, "split_by": args.split_by,
        "fold_results": [{**r, "val_metrics": {k: float(v) for k, v in r["val_metrics"].items()}}
                         for r in fold_results],
        "mean_rmse": float(np.mean(rmses)), "std_rmse": float(np.std(rmses)),
        "global_metrics": {k: float(v) for k, v in global_m.items()},
    }, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
