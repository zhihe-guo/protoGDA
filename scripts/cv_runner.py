#!/usr/bin/env python
"""K-fold cross-validation runner for all report models on GDSC2.

Supports CellQuery (GINE+GMM), CANDELA, and MGATAF under interpolation,
drug_cold, and cell_cold split modes. Reports per-fold metrics plus pooled
(concatenated) metrics.

Usage:
    python scripts/cv_runner.py --model cellquery --k 6 --split_by drug_cold \
        --config config/cellquery_drug_cold.yaml
    python scripts/cv_runner.py --model candela --k 6 --split_by interpolation
    python scripts/cv_runner.py --model mgataf --k 6 --split_by cell_cold
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
from src.training.checkpointing import json_fingerprint, load_checkpoint, resolved_config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def kfold_entities(entities, k, seed):
    rng = np.random.RandomState(seed)
    ents = sorted(set(entities))
    rng.shuffle(ents)
    return [set(f.tolist()) for f in np.array_split(ents, k)]


def fold_manifest(model: str, split_by: str, fold_idx: int, splits: dict) -> dict:
    """Portable identity of a v3 fold, used to reject stale checkpoints."""
    parts = {}
    for name, frame in splits.items():
        records = [
            [int(index), str(row.Drug_ID), str(row.Cell_Line_ID), float(row.Y)]
            for index, row in frame[["Drug_ID", "Cell_Line_ID", "Y"]].iterrows()
        ]
        parts[name] = {
            "row_indices": [int(i) for i in frame.index.tolist()],
            "drug_ids": sorted(map(str, frame["Drug_ID"].unique())),
            "cell_ids": sorted(map(str, frame["Cell_Line_ID"].unique())),
            "n_rows": len(frame),
            "data_fingerprint": json_fingerprint(records),
        }
    return {
        "protocol": "v3",
        "model": model,
        "split_by": split_by,
        "fold": fold_idx + 1,
        "parts": parts,
    }


def build_scaffold_map(drug_smiles: dict) -> dict:
    """Map drug_id -> Murcko scaffold SMILES (transCDR cold-scaffold protocol).

    Uses RDKit MurckoScaffoldSmiles with chirality preserved (matching
    baselines/transcdr/Step1_Data_split.py). Falls back to the input SMILES
    when scaffold extraction fails.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffold_map: dict = {}
    for did, smi in drug_smiles.items():
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                smiles=str(smi), includeChirality=True)
        except Exception:
            scaffold = str(smi)
        scaffold_map[did] = scaffold
    return scaffold_map


def _make_scaffold_fold(full_df: pd.DataFrame, k: int, seed: int, fold_idx: int,
                        scaffold_map: dict) -> dict[str, pd.DataFrame]:
    """Cold-scaffold fold: group drugs by Murcko scaffold, split scaffold groups.

    Test scaffolds are disjoint from train/valid scaffolds (transCDR protocol,
    Sec "Data segmentation strategies"). Drugs sharing a scaffold always travel
    together, so no scaffold leaks between splits.
    """
    if scaffold_map is None:
        raise ValueError("scaffold_cold requires scaffold_map (drug_id -> scaffold).")

    drug_ids = sorted(set(full_df["Drug_ID"].unique()))
    missing = [d for d in drug_ids if str(d) not in scaffold_map]
    if missing:
        raise ValueError(f"scaffold_map missing {len(missing)} drugs, e.g. {missing[:3]}")

    scaffold_to_drugs: dict[str, set] = {}
    for d in drug_ids:
        scaff = scaffold_map[str(d)]
        scaffold_to_drugs.setdefault(scaff, set()).add(str(d))

    scaffolds = sorted(scaffold_to_drugs.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(scaffolds)
    groups = [set(g) for g in np.array_split(scaffolds, k)]
    test_scaffolds = groups[fold_idx]
    # NOTE: groups holds sets; iterate over a DETERMINISTIC order (sorted) so the
    # rest list (and hence the valid/train split) does not depend on Python's
    # string hash randomization (PYTHONHASHSEED), which varies across processes.
    rest = [s for i, g in enumerate(groups) if i != fold_idx for s in sorted(g)]
    rng2 = np.random.RandomState(seed * 1000 + fold_idx)
    rng2.shuffle(rest)
    n_valid = max(int(len(rest) * 0.2), 1)
    valid_scaffolds = set(rest[:n_valid])
    train_scaffolds = set(rest[n_valid:])

    def drugs_of(scaffolds):
        ds: set = set()
        for s in scaffolds:
            ds |= scaffold_to_drugs[s]
        return ds

    return {
        "train": full_df.loc[full_df["Drug_ID"].isin(drugs_of(train_scaffolds))].copy(),
        "valid": full_df.loc[full_df["Drug_ID"].isin(drugs_of(valid_scaffolds))].copy(),
        "test": full_df.loc[full_df["Drug_ID"].isin(drugs_of(test_scaffolds))].copy(),
    }


def make_fold_split(full_df: pd.DataFrame, k: int, split_by: str, seed: int,
                    fold_idx: int, scaffold_map: dict | None = None) -> dict[str, pd.DataFrame]:
    """Return {"train": df, "valid": df, "test": df} for one CV fold.

    Entities (drugs/cells) are split into k disjoint groups. The fold_idx-th
    group is held out as the test split; the remaining k-1 groups are split
    ~80/20 into train/valid (interpolation-like pairing is not enforced here,
    matching the standard k-fold-with-holdout convention).
    """
    rng = np.random.RandomState(seed)
    if split_by == "drug_cold":
        entities = sorted(set(full_df["Drug_ID"].unique()))
        key_col = "Drug_ID"
    elif split_by == "cell_cold":
        entities = sorted(set(full_df["Cell_Line_ID"].unique()))
        key_col = "Cell_Line_ID"
    elif split_by == "scaffold_cold":
        return _make_scaffold_fold(full_df, k, seed, fold_idx, scaffold_map)
    else:
        # interpolation: random row folds
        idx = rng.permutation(len(full_df))
        folds = [set(f.tolist()) for f in np.array_split(idx, k)]
        test_idx = folds[fold_idx]
        other = sorted(set(range(len(full_df))) - test_idx)
        rng2 = np.random.RandomState(seed * 1000 + fold_idx)
        other = rng2.permutation(other)
        n_valid = int(len(other) * 0.2)
        valid_idx = set(other[:n_valid].tolist())
        train_idx = set(other[n_valid:].tolist())
        return {
            "train": full_df.iloc[sorted(train_idx)],
            "valid": full_df.iloc[sorted(valid_idx)],
            "test": full_df.iloc[sorted(test_idx)],
        }

    # Entity-based folds
    rng.shuffle(entities)
    groups = [set(g) for g in np.array_split(entities, k)]
    test_ents = groups[fold_idx]
    # deterministic order: do not rely on set iteration order (string hash seed)
    rest_ents = [e for i, g in enumerate(groups) if i != fold_idx for e in sorted(g)]
    rng2 = np.random.RandomState(seed * 1000 + fold_idx)
    rng2.shuffle(rest_ents)
    n_valid = max(int(len(rest_ents) * 0.2), 1)
    valid_ents = set(rest_ents[:n_valid])
    train_ents = set(rest_ents[n_valid:])

    return {
        "train": full_df.loc[full_df[key_col].isin(train_ents)].copy(),
        "valid": full_df.loc[full_df[key_col].isin(valid_ents)].copy(),
        "test": full_df.loc[full_df[key_col].isin(test_ents)].copy(),
    }


def run_cellquery(cfg, fold_df, fold_idx, device, out_dir, skip_train: bool = False):
    from src.data.dataset import _load_full_tdc_data, _build_registries, DrugCellDataset, collate_drug_cell
    from src.models.model import CellDrugModel
    from src.training.trainer import Trainer
    from torch.utils.data import DataLoader

    df, smiles_series, cell_series = _load_full_tdc_data(cfg.data.dataset_name)
    scaffold_map = None
    if cfg.cv.split_by == "scaffold_cold":
        drug_smiles: dict = {}
        for did, smi in zip(df["Drug_ID"].values, smiles_series.values):
            drug_smiles.setdefault(str(did), str(smi))
        scaffold_map = build_scaffold_map(drug_smiles)
    splits = make_fold_split(df, cfg.cv.k, cfg.cv.split_by, cfg.training.seed, fold_idx,
                             scaffold_map=scaffold_map)
    train_df, valid_df, test_df = splits["train"], splits["valid"], splits["test"]
    manifest = fold_manifest("cellquery", cfg.cv.split_by, fold_idx, splits)

    registries = _build_registries(df, smiles_series, cell_series, cfg,
                                   train_cell_ids=set(train_df["Cell_Line_ID"].unique()))
    drug_id_to_idx = registries["drug_id_to_idx"]
    cell_id_to_idx = registries["cell_id_to_idx"]
    cell_dim = registries["cell_table"].shape[1]

    def build_loader(sub_df, shuffle):
        dr = np.array([drug_id_to_idx[str(sub_df.iloc[i]["Drug_ID"])] for i in range(len(sub_df))], dtype=np.int64)
        cl = np.array([cell_id_to_idx[str(sub_df.iloc[i]["Cell_Line_ID"])] for i in range(len(sub_df))], dtype=np.int64)
        lab = sub_df["Y"].values.astype(np.float32)
        ds = DrugCellDataset(dr, cl, lab)
        nw = cfg.data.num_workers
        return DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=shuffle,
                          num_workers=nw, pin_memory=True, collate_fn=collate_drug_cell,
                          persistent_workers=nw > 0)

    tr = build_loader(train_df, True)
    va = build_loader(valid_df, False)
    te = build_loader(test_df, False)

    model = CellDrugModel.from_config(
        cell_dim=cell_dim, cfg=cfg,
        drug_graphs=registries["drug_graphs"],
        cell_features_table=registries["cell_table"],
        drug_morgan_table=registries.get("drug_morgan_table"),
        drug_chemberta_table=registries.get("drug_chemberta_table"),
    )

    fold_cfg = copy.deepcopy(cfg)
    fold_dir = out_dir / f"fold_{fold_idx+1}"
    fold_cfg.training.checkpoint_dir = str(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    trainer = Trainer(model, fold_cfg, tr, va, device, run_manifest=manifest)
    if skip_train and not (fold_dir / "best.pt").is_file():
        raise FileNotFoundError(f"--skip_train requested but checkpoint is missing: {fold_dir / 'best.pt'}")
    if not skip_train:
        trainer.train()

    # test-set predictions (reload best checkpoint)
    best = fold_dir / "best.pt"
    ck = load_checkpoint(
        best, map_location=device,
        expected_config_fingerprint=json_fingerprint(resolved_config(fold_cfg)),
        expected_manifest_fingerprint=json_fingerprint(manifest),
    )
    model.load_state_dict(ck["model_state_dict"])
    vm = trainer.evaluate(va)
    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in te:
            batch = {kk: v.to(device) for kk, v in batch.items() if isinstance(v, torch.Tensor)}
            p = model(batch)["prediction"].cpu().numpy()
            preds.append(p)
            labels.append(batch["labels"].cpu().numpy())
    tm = compute_metrics(np.concatenate(labels), np.concatenate(preds))
    return {"fold": fold_idx + 1, "val_metrics": vm, "test_metrics": tm,
            "test_preds": np.concatenate(preds), "test_labels": np.concatenate(labels),
            "test_drug_ids": [str(x) for x in test_df["Drug_ID"].values]}


def run_baseline(model_name, cfg, fold_df, fold_idx, device, out_dir, skip_train: bool = False):
    bl_root = ROOT / "baselines"
    sys.path.insert(0, str(bl_root))
    if model_name == "candela":
        sys.path.insert(0, str(ROOT / "baselines" / "candela"))
        import candela.data as data_mod
        import candela.model as model_mod
        import candela.trainer as trainer_mod
    elif model_name == "mgataf":
        sys.path.insert(0, str(ROOT / "baselines" / "mgataf"))
        import mgataf.data as data_mod
        import mgataf.model as model_mod
        import mgataf.trainer as trainer_mod
    elif model_name == "graphdrp":
        sys.path.insert(0, str(ROOT / "baselines" / "graphdrp"))
        import graphdrp.data as data_mod
        import graphdrp.model as model_mod
        import graphdrp.trainer as trainer_mod
    else:
        raise ValueError(f"Unsupported baseline: {model_name}")

    # Build full df the same way each baseline does
    if model_name == "candela":
        raw = pd.read_pickle(str(ROOT / "data" / "gdsc2.pkl"))
        raw = raw.rename(columns={"ID1": "Drug_Name", "ID2": "Cell_Name", "Y": "Y"})
        raw["Drug_ID"] = raw["Drug_Name"]
        raw["Cell_Line_ID"] = raw["Cell_Name"]
        df = raw
        drug_smiles = {}
        for did, smi in zip(raw["Drug_ID"].values, raw["X1"].values):
            drug_smiles.setdefault(str(did), str(smi))
    else:
        from tdc.multi_pred import DrugRes
        dobj = DrugRes(name=cfg.data.dataset_name)
        df = pd.DataFrame({
            "Drug_ID": dobj.entity1_idx.values,
            "Cell_Line_ID": dobj.entity2_idx.values,
            "Y": dobj.y.values,
        })
        drug_smiles = {}
        for did, smi in zip(dobj.entity1_idx.values, dobj.entity1.values):
            drug_smiles.setdefault(str(did), str(smi))

    scaffold_map = build_scaffold_map(drug_smiles) if cfg.cv.split_by == "scaffold_cold" else None
    splits = make_fold_split(df, cfg.cv.k, cfg.cv.split_by, cfg.training.seed, fold_idx,
                             scaffold_map=scaffold_map)
    manifest = fold_manifest(model_name, cfg.cv.split_by, fold_idx, splits)

    if model_name == "candela":
        tr, va, te, registries, meta = data_mod.get_candela_dataloaders(
            cfg, train_df=splits["train"], valid_df=splits["valid"], test_df=splits["test"])
    elif model_name == "mgataf":
        tr, va, te, registries, meta = data_mod.get_mgataf_dataloaders(
            cfg, train_df=splits["train"], valid_df=splits["valid"], test_df=splits["test"])
    else:
        tr, va, te, registries, meta = data_mod.get_graphdrp_dataloaders(
            cfg, train_df=splits["train"], valid_df=splits["valid"], test_df=splits["test"])

    cell_dim = meta["cell_dim"]
    if model_name == "candela":
        model = model_mod.CANDELA.from_config(
            cell_dim=cell_dim, cfg=cfg,
            drug_graphs=registries["drug_graphs"],
            cell_table=registries["cell_table"],
        )
    elif model_name == "mgataf":
        model = model_mod.MGATAF.from_config(
            cell_dim=cell_dim, cfg=cfg,
            drug_graphs=registries["drug_graphs"],
            cell_table=registries["cell_table"],
            drug_fp_table=registries["drug_fp_table"],
        )
    else:
        model = model_mod.GraphDRP.from_config(
            cell_dim=cell_dim, cfg=cfg,
            drug_graphs=registries["drug_graphs"],
            cell_table=registries["cell_table"],
        )

    fold_cfg = copy.deepcopy(cfg)
    fold_dir = out_dir / f"fold_{fold_idx+1}"
    fold_cfg.training.checkpoint_dir = str(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if model_name in {"candela", "graphdrp"}:
        trainer = trainer_mod.Trainer(model, fold_cfg, tr, va, device, run_manifest=manifest)
    else:
        trainer = trainer_mod.Trainer(model, fold_cfg, tr, va, device,
                                      label_scaler=registries.get("label_scaler"),
                                      run_manifest=manifest)
    if skip_train and not (fold_dir / "best.pt").is_file():
        raise FileNotFoundError(f"--skip_train requested but checkpoint is missing: {fold_dir / 'best.pt'}")
    if not skip_train:
        trainer.train()
    best = fold_dir / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training did not produce best checkpoint: {best}")
    ck = load_checkpoint(
        best, map_location=device,
        expected_config_fingerprint=json_fingerprint(resolved_config(fold_cfg)),
        expected_manifest_fingerprint=json_fingerprint(manifest),
    )
    model.load_state_dict(ck["model_state_dict"])
    vm = trainer.evaluate(va)
    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in te:
            batch = {kk: v.to(device) for kk, v in batch.items() if isinstance(v, torch.Tensor)}
            p = model(batch)["prediction"].cpu().numpy()
            preds.append(p)
            labels.append(batch["labels"].cpu().numpy())
    # MGATAF: inverse-transform predictions from normalized to raw scale
    y = np.concatenate(labels)
    if model_name == "mgataf" and registries.get("label_scaler"):
        y = y * registries["label_scaler"]["span"] + registries["label_scaler"]["min"]
        p = np.concatenate(preds) * registries["label_scaler"]["span"] + registries["label_scaler"]["min"]
    else:
        p = np.concatenate(preds)
    tm = compute_metrics(y, p)
    return {"fold": fold_idx + 1, "val_metrics": vm, "test_metrics": tm,
            "test_preds": p, "test_labels": y,
            "test_drug_ids": [str(x) for x in splits["test"]["Drug_ID"].values]}


def compute_drug_level_metrics(drug_ids, preds, labels):
    """Aggregate pair-level predictions to drug level, then compute global
    Pearson/Spearman across all drugs (concatenated folds).

    Each drug's potency is averaged over all its cell lines, yielding one
    (label, pred) point per drug. This is robust to per-fold drug sampling
    and directly measures drug-level ranking ability.
    """
    df = pd.DataFrame({
        "drug": np.asarray(drug_ids).ravel(),
        "pred": np.asarray(preds).ravel(),
        "label": np.asarray(labels).ravel(),
    })
    g = df.groupby("drug")[["pred", "label"]].mean()
    return compute_metrics(g["label"].values, g["pred"].values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["cellquery", "candela", "mgataf", "graphdrp"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--split_by", type=str, required=True,
                        choices=["interpolation", "drug_cold", "cell_cold", "scaffold_cold"])
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs (default: 200 interp / 60 drug_cold / 100 cell_cold / 60 scaffold_cold)")
    parser.add_argument("--patience", type=int, default=None,
                        help="Override early_stopping_patience (use a large value e.g. 999 "
                             "to disable early stopping / run full epochs)")
    parser.add_argument("--checkpoint-metric", type=str, default=None,
                        choices=["rmse", "pearson_r"],
                        help="Validation metric used to select best.pt (default: config or rmse)")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training; load existing best.pt per fold and re-run inference only")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if args.config is None:
        if args.model == "cellquery":
            args.config = str(ROOT / "config" / f"cellquery_{args.split_by}.yaml")
        else:
            args.config = str(ROOT / "baselines" / args.model / f"config_{args.split_by}.yaml")

    cfg = load_config(args.config)
    if "cv" not in cfg:
        from omegaconf import OmegaConf
        cfg.cv = OmegaConf.create({})
    cfg.cv.k = args.k
    cfg.cv.split_by = args.split_by
    # Match the official full-run epochs per split mode
    default_epochs = {"interpolation": 200, "drug_cold": 60, "cell_cold": 100, "scaffold_cold": 60}
    cfg.training.epochs = args.epochs if args.epochs is not None else default_epochs[args.split_by]
    if args.patience is not None:
        cfg.training.early_stopping_patience = args.patience
    if args.checkpoint_metric is not None:
        cfg.training.checkpoint_metric = args.checkpoint_metric
    set_seed(cfg.training.seed)
    device = resolve_device(cfg.training.device)

    out_root = Path(args.out) if args.out else Path(cfg.training.checkpoint_dir)
    out_root = out_root if out_root.is_absolute() else ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Model={args.model}, K={args.k}, split_by={args.split_by}, "
          f"epochs={cfg.training.epochs}, out={out_root}")

    results = []
    all_preds, all_labels, all_drug_ids = [], [], []
    for fi in range(args.k):
        print(f"\n{'='*60}\n  FOLD {fi+1}/{args.k}\n{'='*60}")
        # Deterministic per-fold reproducibility: fold uses seed = base + fold_idx
        # (matches cv_resume v3 protocol).
        set_seed(cfg.training.seed + fi)
        if args.model == "cellquery":
            r = run_cellquery(cfg, None, fi, device, out_root, skip_train=args.skip_train)
        else:
            r = run_baseline(args.model, cfg, None, fi, device, out_root,
                             skip_train=args.skip_train)
        results.append(r)
        all_preds.append(r["test_preds"])
        all_labels.append(r["test_labels"])
        all_drug_ids.append(np.asarray(r["test_drug_ids"]))
        print(f"  Fold {fi+1} test: {format_metrics(r['test_metrics'])}")

    print(f"\n{'='*60}\n  CV SUMMARY ({args.k}-fold, split_by={args.split_by})\n{'='*60}")
    keys = ["pearson_r", "rmse", "mae", "r2", "spearman_r"]
    for key in keys:
        vals = [r["test_metrics"].get(key, float("nan")) for r in results]
        print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    pooled = compute_metrics(np.concatenate(all_labels), np.concatenate(all_preds))
    print(f"\n  Pooled: {format_metrics(pooled)}")

    drug_level = compute_drug_level_metrics(
        np.concatenate(all_drug_ids),
        np.concatenate(all_preds),
        np.concatenate(all_labels),
    )
    print(f"\n  Drug-level (per-drug aggregated): {format_metrics(drug_level)}")

    np.savez(
        out_root / "cv_predictions.npz",
        drug_ids=np.concatenate(all_drug_ids),
        preds=np.concatenate(all_preds),
        labels=np.concatenate(all_labels),
    )

    out = out_root / "cv_results.json"
    out.write_text(json.dumps({
        "model": args.model, "k": args.k, "split_by": args.split_by,
        "fold_results": [{
            "fold": r["fold"],
            "test_metrics": {kk: float(v) for kk, v in r["test_metrics"].items()},
        } for r in results],
        "mean_std": {key: {"mean": float(np.mean([r["test_metrics"].get(key, float("nan")) for r in results])),
                           "std": float(np.std([r["test_metrics"].get(key, float("nan")) for r in results]))}
                     for key in keys},
        "pooled_metrics": {k: float(v) for k, v in pooled.items()},
        "drug_level_metrics": {k: float(v) for k, v in drug_level.items()},
    }, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
