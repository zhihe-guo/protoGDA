#!/usr/bin/env python
"""External validation of protoGDA / CANDELA / MGATAF on GDSC1 (zero-shot).

Design
------
GDSC1 (data/gdsc1.pkl) shares 62 drugs (matched by canonical SMILES) and
~803 cell lines with GDSC2 (TDC DrugRes). In the v3 drug_cold 6-fold
protocol each drug is the *test* drug of exactly one fold. For every shared
drug d we predict its GDSC1 responses using *only* the checkpoint of the fold
where d was held out as a test drug — that model never saw d in training.
This yields a strict zero-shot, cross-dataset evaluation on an independently
measured assay. All three models use the same seed (42), so d is held out in
the same fold for each model.

Note on label scale
-------------------
GDSC1 and GDSC2 responses share the same (higher = more resistant) direction
but differ in absolute assay scale, so the primary metrics are scale-free
Pearson / Spearman correlations. RMSE is reported for reference only.

Usage
-----
    python scripts/external_validate_gdsc1.py --model cellquery|candela|mgataf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rdkit import Chem
from scipy.stats import pearsonr, spearmanr

from src.config import load_config, resolve_device
from scripts.cv_runner import make_fold_split

CHECKPOINT_DIRS = {
    "cellquery": "checkpoints/cv_drug_cold_v3",
    "candela": "checkpoints/cv_candela_drug_cold_v3",
    "mgataf": "checkpoints/cv_mgataf_drug_cold_v3",
    "graphdrp": "checkpoints/cv_graphdrp_drug_cold_v3",
}


def _canon(smi: str) -> str | None:
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m) if m is not None else None
    except Exception:
        return None


def load_gdsc2(model: str, cfg):
    """Return (df, smiles_series, cell_series) in the model's native namespace."""
    if model == "candela":
        raw = pd.read_pickle(str(ROOT / "data" / "gdsc2.pkl"))
        raw = raw.rename(columns={"ID1": "Drug_Name", "ID2": "Cell_Name", "Y": "Y"})
        raw["Drug_ID"] = raw["Drug_Name"]
        raw["Cell_Line_ID"] = raw["Cell_Name"]
        return raw, raw["X1"], raw["X2"]
    from src.data.dataset import _load_full_tdc_data
    return _load_full_tdc_data(cfg.data.dataset_name)


def build_test_df(model: str, part: pd.DataFrame, did2idx, cid2idx,
                  registries) -> pd.DataFrame:
    """Convert GDSC1 rows into the model-native lookup columns."""
    return part[["Drug2", "Cell_Name", "Y"]].rename(
        columns={"Drug2": "Drug_ID", "Cell_Name": "Cell_Line_ID"})


def run_fold_cellquery(cfg, full, part, fi, device, ckpt_dir):
    from src.data.dataset import _build_registries
    from src.models.model import CellDrugModel

    df, smiles_series, cell_series = full
    splits = make_fold_split(df, 6, "drug_cold", cfg.training.seed, fi)
    registries = _build_registries(
        df, smiles_series, cell_series, cfg,
        train_cell_ids=set(splits["train"]["Cell_Line_ID"].unique()),
    )
    model = CellDrugModel.from_config(
        cell_dim=registries["cell_table"].shape[1], cfg=cfg,
        drug_graphs=registries["drug_graphs"],
        cell_features_table=registries["cell_table"],
        drug_morgan_table=registries.get("drug_morgan_table"),
        drug_chemberta_table=registries.get("drug_chemberta_table"),
    ).to(device)
    did2idx, cid2idx = registries["drug_id_to_idx"], registries["cell_id_to_idx"]

    part = part[part["Drug2"].isin(did2idx) & part["Cell_Name"].isin(cid2idx)]
    drug_idx = part["Drug2"].map(lambda d: did2idx[str(d)]).values.astype(np.int64)
    cell_idx = part["Cell_Name"].map(lambda c: cid2idx[str(c)]).values.astype(np.int64)
    labels = part["Y"].values.astype(np.float32)
    return model, drug_idx, cell_idx, labels


def run_fold_baseline(model_name, cfg, full, part, fi, device, ckpt_dir):
    bl_root = ROOT / "baselines"
    sys.path.insert(0, str(bl_root))
    if model_name == "candela":
        sys.path.insert(0, str(ROOT / "baselines" / "candela"))
        import candela.data as data_mod
        import candela.model as model_mod
    elif model_name == "mgataf":
        sys.path.insert(0, str(ROOT / "baselines" / "mgataf"))
        import mgataf.data as data_mod
        import mgataf.model as model_mod
    elif model_name == "graphdrp":
        sys.path.insert(0, str(ROOT / "baselines" / "graphdrp"))
        import graphdrp.data as data_mod
        import graphdrp.model as model_mod
    else:
        raise ValueError(f"Unsupported baseline: {model_name}")

    df, _, _ = full
    splits = make_fold_split(df, 6, "drug_cold", cfg.training.seed, fi)

    test_df = build_test_df(model_name, part, None, None, None)
    if model_name == "candela":
        _, _, te, registries, meta = data_mod.get_candela_dataloaders(
            cfg, train_df=splits["train"], valid_df=splits["valid"], test_df=test_df)
    elif model_name == "mgataf":
        _, _, te, registries, meta = data_mod.get_mgataf_dataloaders(
            cfg, train_df=splits["train"], valid_df=splits["valid"], test_df=test_df)
    else:
        _, _, te, registries, meta = data_mod.get_graphdrp_dataloaders(
            cfg, train_df=splits["train"], valid_df=splits["valid"], test_df=test_df)
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
    model.to(device)
    return model, te, registries


def collect_stats(all_preds, all_labels, drug_pred_mean, drug_label_mean,
                  per_drug_pearson, part, preds, labels):
    all_preds.append(preds)
    all_labels.append(labels)
    part_df = part.copy()
    part_df["pred"] = preds
    part_df = part_df.reset_index(drop=True)
    for d, g in part_df.groupby("Drug2"):
        drug_pred_mean[d] = g["pred"].mean()
        drug_label_mean[d] = g["Y"].mean()
    for d, g in part_df.groupby("Drug2"):
        if len(g) >= 5:
            r, _ = pearsonr(g["pred"], g["Y"])
            per_drug_pearson.append(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(CHECKPOINT_DIRS), required=True)
    ap.add_argument("--folds", type=str, default="all",
                    help="Comma list of 0-based fold indices to run (default: all)")
    args = ap.parse_args()
    model = args.model
    device = resolve_device("auto")
    print(f"Model={model} | Device: {device}", flush=True)

    cfg = load_config(ROOT / "config" / "cellquery_drug_cold.yaml" if model == "cellquery"
                      else ROOT / "baselines" / model / "config_drug_cold.yaml")

    # --- GDSC2 full data (the model's training universe) ---------------------
    df2, smiles2, cell2 = load_gdsc2(model, cfg)
    canon2drug = {}
    for did in df2["Drug_ID"].unique():
        smi = str(smiles2[df2[df2["Drug_ID"] == did].index[0]])
        c = _canon(smi)
        if c:
            canon2drug.setdefault(c, did)
    print(f"GDSC2: {len(df2):,} pairs, {df2['Drug_ID'].nunique()} drugs", flush=True)

    # --- GDSC1 ---------------------------------------------------------------
    g1 = pd.read_pickle(str(ROOT / "data" / "gdsc1.pkl"))
    g1 = g1.rename(columns={"ID1": "Drug_Name", "ID2": "Cell_Name", "Y": "Y"})
    g1["canon"] = g1["X1"].map(_canon)

    cells2 = set(df2["Cell_Line_ID"].unique())
    shared_mask = g1["canon"].isin(canon2drug) & g1["Cell_Name"].isin(cells2)
    shared = g1[shared_mask].copy()
    shared["Drug2"] = shared["canon"].map(canon2drug)
    print(f"GDSC1 shared pairs: {len(shared):,} "
          f"({shared['Drug2'].nunique()} drugs, {shared['Cell_Name'].nunique()} cells)",
          flush=True)

    # --- Per-drug v3 fold assignment (which fold holds it out as test) -------
    fold_test_drugs = {}
    for fi in range(6):
        splits = make_fold_split(df2, 6, "drug_cold", cfg.training.seed, fi)
        fold_test_drugs[fi] = set(splits["test"]["Drug_ID"].unique())
    shared["fold"] = shared["Drug2"].map(
        lambda d: next(fi for fi, drugs in fold_test_drugs.items() if d in drugs))
    assert shared["fold"].notna().all()
    by_fold = {fi: shared[shared["fold"] == fi] for fi in range(6)}

    # --- Per-fold zero-shot prediction ---------------------------------------
    all_preds, all_labels = [], []
    drug_pred_mean, drug_label_mean = {}, {}
    per_drug_pearson: list[float] = []

    for fi in range(6):
        if args.folds != "all" and fi not in {int(x) for x in args.folds.split(",")}:
            continue
        part = by_fold[fi]
        print(f"\n=== fold {fi+1} ({len(part):,} GDSC1 pairs) ===", flush=True)
        ckpt_path = ROOT / CHECKPOINT_DIRS[model] / f"fold_{fi+1}" / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)

        if model == "cellquery":
            mdl, drug_idx, cell_idx, labels = run_fold_cellquery(
                cfg, (df2, smiles2, cell2), part, fi, device, ckpt_path)
            sd = torch.load(ckpt_path, map_location=device, weights_only=False)
            mdl.load_state_dict(sd["model_state_dict"])
            mdl.eval()
            preds = []
            bs = cfg.data.batch_size
            with torch.inference_mode():
                for s in range(0, len(part), bs):
                    b = slice(s, s + bs)
                    batch = {
                        "drug_ids": torch.from_numpy(drug_idx[b]).to(device),
                        "cell_ids": torch.from_numpy(cell_idx[b]).to(device),
                    }
                    preds.append(mdl(batch)["prediction"].cpu().numpy())
            preds = np.concatenate(preds).reshape(-1)
            part_used = part.reset_index(drop=True)
            labels_cpu = labels
        else:
            mdl, te, registries = run_fold_baseline(
                model, cfg, (df2, smiles2, cell2), part, fi, device, ckpt_path)
            sd = torch.load(ckpt_path, map_location=device, weights_only=False)
            mdl.load_state_dict(sd["model_state_dict"])
            mdl.eval()
            labels_raw, preds = [], []
            with torch.inference_mode():
                for batch in te:
                    bb = {k: v.to(device) for k, v in batch.items()
                          if isinstance(v, torch.Tensor)}
                    preds.append(mdl(bb)["prediction"].cpu().numpy())
                    labels_raw.append(bb["labels"].cpu().numpy())
            labels_raw = np.concatenate(labels_raw).reshape(-1)
            preds = np.concatenate(preds).reshape(-1)
            if model == "mgataf" and registries.get("label_scaler"):
                sc = registries["label_scaler"]
                labels_raw = labels_raw * sc["span"] + sc["min"]   # GDSC1 scale
                preds = preds * sc["span"] + sc["min"]             # GDSC2 scale
            part_used = part.reset_index(drop=True)   # te is shuffle=False
            labels_cpu = labels_raw

        collect_stats(all_preds, all_labels, drug_pred_mean, drug_label_mean,
                      per_drug_pearson, part_used, preds, labels_cpu)
        if len(preds) != len(labels_cpu) or len(labels_cpu) != len(part):
            print(f"!! len mismatch: preds={len(preds)} labels={len(labels_cpu)} "
                  f"part={len(part)}", flush=True)
        assert len(preds) == len(labels_cpu) == len(part), "length mismatch"
        if model == "cellquery":
            del mdl
        else:
            del mdl, registries
        torch.cuda.empty_cache()

    # --- Aggregate metrics ----------------------------------------------------
    print("PERFOLD:", flush=True)
    for i, (a, b) in enumerate(zip(all_preds, all_labels)):
        print(f"  fold {i+1}: pred {a.shape} label {b.shape} "
              f"n_pred={len(a)} n_label={len(b)}", flush=True)
    pred = np.concatenate(all_preds)
    label = np.concatenate(all_labels)
    r_p, _ = pearsonr(pred, label)
    r_s, _ = spearmanr(pred, label)
    rmse = float(np.sqrt(np.mean((pred - label) ** 2)))
    print("\n===== GDSC1 zero-shot external validation =====", flush=True)
    print(f"n pairs         : {len(label):,}")
    print(f"pooled Pearson  : {r_p:.4f}")
    print(f"pooled Spearman : {r_s:.4f}")
    print(f"RMSE (ref only) : {rmse:.4f}   (GDSC1/GDSC2 assay scales differ)")

    keys = sorted(set(drug_pred_mean) & set(drug_label_mean))
    dp = np.array([drug_pred_mean[k] for k in keys])
    dl = np.array([drug_label_mean[k] for k in keys])
    r_dp, _ = pearsonr(dp, dl)
    r_ds, _ = spearmanr(dp, dl)
    print(f"n drugs (aggregated): {len(keys)}")
    print(f"drug-level Pearson  : {r_dp:.4f}")
    print(f"drug-level Spearman : {r_ds:.4f}")

    pr = np.array(per_drug_pearson)
    print(f"per-drug Pearson (within-drug, n>=5): mean {pr.mean():.4f} +/- {pr.std():.4f} "
          f"(n={len(pr)})")

    out = {
        "model": model,
        "n_pairs": int(len(label)),
        "pooled_pearson": float(r_p),
        "pooled_spearman": float(r_s),
        "rmse_ref_only": rmse,
        "n_drugs_aggregated": len(keys),
        "drug_level_pearson": float(r_dp),
        "drug_level_spearman": float(r_ds),
        "per_drug_pearson_mean": float(pr.mean()),
        "per_drug_pearson_std": float(pr.std()),
        "per_drug_pearson_n": int(len(pr)),
        "per_drug_pearson": [float(x) for x in pr],
    }
    dest = ROOT / "checkpoints" / f"external_gdsc1_{model}_results.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {dest}")


if __name__ == "__main__":
    main()
