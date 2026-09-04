#!/usr/bin/env python
"""Chemical-distance analysis: is scaffold_cold's test set structurally more
remote from its training drugs than drug_cold's?

For each fold under each protocol, compute the mean nearest-neighbour Tanimoto
distance (Morgan r=2, 1024 bits) from every test drug to the training drugs.
If scaffold_cold folds are NOT more remote, the lower CV Pearson cannot be
attributed to an intrinsically harder generalization gap.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.dataset import _load_full_tdc_data
from scripts.cv_runner import build_scaffold_map, make_fold_split


def morgan_fp(smi, radius=2, nbits=1024):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)


def nn_distance(fp, fps):
    sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
    return 1.0 - max(sims)


def main():
    cfg = load_config(str(ROOT / "config" / "exp_interp_egnn_gmm_attn.yaml"))
    df, smiles_series, cell_series = _load_full_tdc_data(cfg.data.dataset_name)
    k, seed = 6, 42

    drug_smiles: dict = {}
    for did, smi in zip(df["Drug_ID"].values, smiles_series.values):
        drug_smiles.setdefault(str(did), str(smi))
    scaffold_map = build_scaffold_map(drug_smiles)

    fp_cache = {}
    for did, smi in drug_smiles.items():
        fp = morgan_fp(smi)
        if fp is not None:
            fp_cache[did] = fp

    out = {}
    for protocol in ("drug_cold", "scaffold_cold"):
        rows = []
        for fold_idx in range(k):
            splits = make_fold_split(df, k, protocol, seed, fold_idx,
                                     scaffold_map=scaffold_map if protocol == "scaffold_cold" else None)
            tr_drugs = sorted(set(splits["train"]["Drug_ID"].astype(str)))
            te_drugs = sorted(set(splits["test"]["Drug_ID"].astype(str)))
            tr_fps = [fp_cache[d] for d in tr_drugs if d in fp_cache]
            per_drug = []
            for d in te_drugs:
                fp = fp_cache.get(d)
                if fp is None or not tr_fps:
                    continue
                per_drug.append(nn_distance(fp, tr_fps))
            rows.append({
                "fold": fold_idx + 1,
                "n_test_drugs": len(te_drugs),
                "mean_nn_dist": float(np.mean(per_drug)),
                "min_nn_dist": float(np.min(per_drug)),
                "p90_nn_dist": float(np.percentile(per_drug, 90)),
            })
        out[protocol] = rows

    print(f"{'protocol':<14}{'fold':<6}{'n_test':<8}{'mean_nn':<9}{'min_nn':<8}{'p90_nn':<8}")
    for protocol, rows in out.items():
        for r in rows:
            print(f"{protocol:<14}{r['fold']:<6}{r['n_test_drugs']:<8}"
                  f"{r['mean_nn_dist']:<9.4f}{r['min_nn_dist']:<8.4f}{r['p90_nn_dist']:<8.4f}")
        means = [r["mean_nn_dist"] for r in rows]
        print(f"  -> mean over folds: {np.mean(means):.4f}  (min per fold: "
              f"{np.mean([r['min_nn_dist'] for r in rows]):.4f})")

    out_path = ROOT / "results" / "scaffold_vs_drugcold_chemdist.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
