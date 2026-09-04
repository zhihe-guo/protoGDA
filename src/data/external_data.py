"""External drug-cell data loader.

Reads precomputed registries produced by scripts/precompute_external.py.
Performs ZERO on-the-fly computation — all drug graphs, Morgan fingerprints,
conformer coordinates, ChemBERTa embeddings, and cell features are loaded
from disk via torch.load / pickle.load.

Expected directory layout (data/external/):

    drug_graphs.pt       list[PyG Data] — one per unique SMILES
    drug_smiles.pkl       list[str]      — SMILES strings, same order
    drug_morgan.pt        Tensor (N_drugs, n_bits)  (optional)
    drug_chemberta.pt     Tensor (N_drugs, 768)     (optional)
    conformer_cache.pt    {SMILES: Tensor(N_atoms, 3)} (optional)
    cell_table.pt         Tensor (N_cells, pca_dim)
    drug_id_to_idx.pkl    {SMILES: int}
    cell_id_to_idx.pkl    {ModelID: int}
    meta.json             cell_dim, drug_count, cell_count, ...
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
#  ID-based Dataset (lightweight — only three integers per sample)
# ---------------------------------------------------------------------------

class DrugCellDataset(Dataset):
    def __init__(self, drug_ids: np.ndarray, cell_ids: np.ndarray, labels: np.ndarray):
        self.drug_ids = drug_ids.astype(np.int64)
        self.cell_ids = cell_ids.astype(np.int64)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.drug_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "drug_id": int(self.drug_ids[idx]),
            "cell_id": int(self.cell_ids[idx]),
            "label":  float(self.labels[idx]),
        }


def collate_drug_cell(batch: list[dict]) -> dict[str, Any]:
    return {
        "drug_ids": torch.tensor([b["drug_id"] for b in batch], dtype=torch.long),
        "cell_ids": torch.tensor([b["cell_id"] for b in batch], dtype=torch.long),
        "labels":   torch.tensor([b["label"]  for b in batch], dtype=torch.float),
    }


# ---------------------------------------------------------------------------
#  Splits (same logic as dataset.py)
# ---------------------------------------------------------------------------

def _custom_split(df: pd.DataFrame, split_mode: str, seed: int,
                  frac: tuple[float, float, float] = (0.7, 0.1, 0.2),
                  ) -> dict[str, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    drugs = df["Drug_ID"].unique()
    cells = df["Cell_Line_ID"].unique()

    if split_mode == "interpolation":
        idx = rng.permutation(len(df))
        n_tr = int(len(df) * frac[0])
        n_va = int(len(df) * frac[1])
        tr_set = set(idx[:n_tr].tolist())
        va_set = set(idx[n_tr:n_tr + n_va].tolist())
        te_set = set(idx[n_tr + n_va:].tolist())
        tr_d = set(df.iloc[list(tr_set)]["Drug_ID"].values)
        tr_c = set(df.iloc[list(tr_set)]["Cell_Line_ID"].values)
        for s in (va_set, te_set):
            for i in list(s):
                r = df.iloc[i]
                if r["Drug_ID"] not in tr_d or r["Cell_Line_ID"] not in tr_c:
                    s.discard(i)
                    tr_set.add(i)
                    tr_d.add(r["Drug_ID"])
                    tr_c.add(r["Cell_Line_ID"])
        return {
            "train": df.iloc[list(tr_set)],
            "valid": df.iloc[list(va_set)],
            "test":  df.iloc[list(te_set)],
        }

    if split_mode == "cell_cold":
        cs = cells.copy()
        rng.shuffle(cs)
        nt = max(int(len(cs) * frac[0]), 1)
        nv = max(int(len(cs) * frac[1]), 1)
        sets = {
            "train": set(cs[:nt]),
            "valid": set(cs[nt:nt + nv]),
            "test":  set(cs[nt + nv:]),
        }
        return {k: df[df["Cell_Line_ID"].isin(s)] for k, s in sets.items()}

    if split_mode == "drug_cold":
        ds = drugs.copy()
        rng.shuffle(ds)
        nt = max(int(len(ds) * frac[0]), 1)
        nv = max(int(len(ds) * frac[1]), 1)
        sets = {
            "train": set(ds[:nt]),
            "valid": set(ds[nt:nt + nv]),
            "test":  set(ds[nt + nv:]),
        }
        return {k: df[df["Drug_ID"].isin(s)] for k, s in sets.items()}

    raise ValueError(f"Unknown split_mode: {split_mode}")


# ---------------------------------------------------------------------------
#  Registry loader — zero on-the-fly computation
# ---------------------------------------------------------------------------

_CACHE_DIR = "data/external"

def _load_registries() -> dict[str, Any]:
    """Load ALL precomputed registries from disk.

    Falls back to the root 'data/' directory if data/external/ is empty.
    Returns None for optional tables that don't exist on disk.
    """
    cache_dir = _CACHE_DIR
    if not os.path.isdir(cache_dir):
        # Fallback: look for files directly in data/
        cache_dir = "data"

    print(f"Loading precomputed registries from {cache_dir}/ ...")

    # ---- Required ----
    drug_graphs = torch.load(os.path.join(cache_dir, "drug_graphs.pt"), map_location="cpu",
                             weights_only=False)
    with open(os.path.join(cache_dir, "drug_smiles.pkl"), "rb") as f:
        drug_smiles = pickle.load(f)
    cell_table = torch.load(os.path.join(cache_dir, "cell_table.pt"), map_location="cpu",
                            weights_only=True)
    with open(os.path.join(cache_dir, "drug_id_to_idx.pkl"), "rb") as f:
        drug_id_to_idx = pickle.load(f)
    with open(os.path.join(cache_dir, "cell_id_to_idx.pkl"), "rb") as f:
        cell_id_to_idx = pickle.load(f)

    print(f"  drug_graphs: {len(drug_graphs)}, cell_table: {cell_table.shape}")
    print(f"  drugs: {len(drug_id_to_idx)}, cells: {len(cell_id_to_idx)}")

    # ---- Optional ----
    mp = os.path.join(cache_dir, "drug_morgan.pt")
    drug_morgan_table = torch.load(mp, map_location="cpu", weights_only=True) if os.path.exists(mp) else None
    if drug_morgan_table is not None:
        print(f"  drug_morgan: {drug_morgan_table.shape}")

    cp = os.path.join(cache_dir, "drug_chemberta.pt")
    drug_chemberta_table = torch.load(cp, map_location="cpu", weights_only=True) if os.path.exists(cp) else None
    if drug_chemberta_table is not None:
        print(f"  drug_chemberta: {drug_chemberta_table.shape}")

    ccp = os.path.join(cache_dir, "conformer_cache.pt")
    conformer_cache: dict | None = None
    if os.path.exists(ccp):
        conformer_cache = torch.load(ccp, map_location="cpu", weights_only=False)
        print(f"  conformer_cache: {len(conformer_cache)} entries")
    else:
        print("  conformer_cache: not found (EGNN will use zero coords)")

    return {
        "drug_graphs": drug_graphs,
        "drug_smiles": drug_smiles,
        "cell_table": cell_table,
        "drug_id_to_idx": drug_id_to_idx,
        "cell_id_to_idx": cell_id_to_idx,
        "drug_morgan_table": drug_morgan_table,
        "drug_chemberta_table": drug_chemberta_table,
        "conformer_cache": conformer_cache,
    }


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def get_external_dataloaders(cfg: DictConfig):
    """Return (train_ldr, valid_ldr, test_ldr, registries, meta).

    All heavy computation (drug graphs, Morgan, ChemBERTa, conformers,
    cell PCA) is done offline by precompute_external.py. This function
    only performs fast disk I/O.
    """
    ext_cfg = cfg.data.get("external", {})
    parquet_path = ext_cfg.get("pairs_parquet", "data/dti_pairs.parquet")
    split_mode = cfg.data.get("split_mode", "interpolation")
    seed = cfg.training.seed

    print(f"External dataset | split={split_mode} | parquet={parquet_path}")

    # ---- 1. Load registries (all precomputed, fast disk reads) ----
    registries = _load_registries()
    did2idx = registries["drug_id_to_idx"]
    cid2idx = registries["cell_id_to_idx"]
    cell_dim = registries["cell_table"].shape[1]

    # ---- 2. Load pair table ----
    print(f"Loading {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    df["Drug_ID"] = df["Unified_ID"].astype(str).str.strip()
    df["Cell_Line_ID"] = df["ModelID"].astype(str).str.strip()
    df["Y"] = df["IC50_uM"].astype(np.float32)
    df["Y"] = np.log10(np.maximum(df["Y"].values, 1e-6))  # log10(IC50_uM), match GDSC2 scale
    df = df[["Drug_ID", "Cell_Line_ID", "Y", "gene_idx", "conformer_sdf"]]

    print(f"  Total: {len(df):,} pairs, {df.Drug_ID.nunique()} drugs, "
          f"{df.Cell_Line_ID.nunique()} cells")

    # ---- 3. Split ----
    splits = _custom_split(df, split_mode, seed)
    for n in ("train", "valid", "test"):
        s = splits[n]
        print(f"  {n}: {len(s):,} pairs, {s.Drug_ID.nunique()} drugs, "
              f"{s.Cell_Line_ID.nunique()} cells")

    # ---- 4. Build ID-based DataLoaders ----
    normalize_label = cfg.data.get("normalize_label", False)
    label_scaler = None
    if normalize_label:
        train_labels = splits["train"]["Y"].values.astype(np.float32)
        y_min = float(train_labels.min())
        y_max = float(train_labels.max())
        span = y_max - y_min
        if span < 1e-12:
            span = 1.0
        label_scaler = {"min": y_min, "max": y_max, "span": span}
        print(f"Label normalization: y' = (y - {y_min:.4f}) / {span:.4f}  "
              f"(train range [{y_min:.4f}, {y_max:.4f}])")
        registries["label_scaler"] = label_scaler

    datasets: dict[str, DrugCellDataset] = {}
    for n in ("train", "valid", "test"):
        s = splits[n]
        dids = np.array([did2idx[str(s.iloc[i]["Drug_ID"])] for i in range(len(s))],
                        dtype=np.int64)
        cids = np.array([cid2idx[str(s.iloc[i]["Cell_Line_ID"])] for i in range(len(s))],
                        dtype=np.int64)
        ys = s["Y"].values.astype(np.float32)
        if label_scaler is not None:
            ys = (ys - label_scaler["min"]) / label_scaler["span"]
        datasets[n] = DrugCellDataset(dids, cids, ys)

    nw = cfg.data.num_workers
    kw: dict[str, Any] = {
        "batch_size": cfg.data.batch_size,
        "num_workers": nw,
        "pin_memory": True,
        "collate_fn": collate_drug_cell,
        "persistent_workers": nw > 0,
    }
    tr = DataLoader(datasets["train"], shuffle=True, **kw)
    va = DataLoader(datasets["valid"], shuffle=False, **kw)
    te = DataLoader(datasets["test"], shuffle=False, **kw)

    registries["train_df"] = splits["train"]
    meta = {"cell_dim": cell_dim, "label_col": "Y", "dataset_source": "external"}
    return tr, va, te, registries, meta
