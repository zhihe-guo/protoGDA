"""GDSC2 drug-cell dataset via TDC with custom cold-start split modes.

DataLoader returns lightweight integer IDs only.
Global drug graphs, cell feature table, Morgan fingerprints, and ChemBERTa
embeddings are built once and held by the model.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from src.data.preprocessing import compute_morgan_fingerprint, normalize_features, smiles_to_graph


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------

def _load_full_tdc_data(
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load the full GDSC2 pair table, SMILES series, and cell-feature series."""
    from tdc.multi_pred import DrugRes

    data = DrugRes(name=dataset_name)
    df = pd.DataFrame(
        {
            "Drug_ID": data.entity1_idx.values,
            "Cell_Line_ID": data.entity2_idx.values,
            "Y": data.y.values,
        }
    )
    smiles_series = data.entity1.copy()
    cell_series = data.entity2.copy()
    return df, smiles_series, cell_series


# ---------------------------------------------------------------------------
#  Custom dataset splits
# ---------------------------------------------------------------------------

def _custom_split(
    df: pd.DataFrame,
    split_mode: str,
    seed: int,
    frac: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> dict[str, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    drugs = df["Drug_ID"].unique()
    cells = df["Cell_Line_ID"].unique()

    if split_mode == "interpolation":
        indices = rng.permutation(len(df))
        n_train = int(len(df) * frac[0])
        n_valid = int(len(df) * frac[1])
        train_idx = set(indices[:n_train].tolist())
        valid_idx = set(indices[n_train : n_train + n_valid].tolist())
        test_idx = set(indices[n_train + n_valid :].tolist())
        train_drugs = set(df.iloc[list(train_idx)]["Drug_ID"].values)
        train_cells = set(df.iloc[list(train_idx)]["Cell_Line_ID"].values)
        for sidx in (valid_idx, test_idx):
            for i in list(sidx):
                row = df.iloc[i]
                if row["Drug_ID"] not in train_drugs or row["Cell_Line_ID"] not in train_cells:
                    sidx.discard(i)
                    train_idx.add(i)
                    train_drugs.add(row["Drug_ID"])
                    train_cells.add(row["Cell_Line_ID"])
        splits = {
            "train": df.iloc[list(train_idx)],
            "valid": df.iloc[list(valid_idx)],
            "test": df.iloc[list(test_idx)],
        }

    elif split_mode == "cell_cold":
        cells_shuffled = cells.copy(); rng.shuffle(cells_shuffled)
        n_c_train = max(int(len(cells) * frac[0]), 1)
        n_c_valid = max(int(len(cells) * frac[1]), 1)
        train_cells = set(cells_shuffled[:n_c_train].tolist())
        valid_cells = set(cells_shuffled[n_c_train : n_c_train + n_c_valid].tolist())
        test_cells = set(cells_shuffled[n_c_train + n_c_valid :].tolist())
        splits = {}
        for name, cell_set in [("train", train_cells), ("valid", valid_cells), ("test", test_cells)]:
            mask = df["Cell_Line_ID"].isin(cell_set)
            splits[name] = df.loc[mask]

    elif split_mode == "drug_cold":
        drugs_shuffled = drugs.copy(); rng.shuffle(drugs_shuffled)
        n_d_train = max(int(len(drugs) * frac[0]), 1)
        n_d_valid = max(int(len(drugs) * frac[1]), 1)
        train_drugs = set(drugs_shuffled[:n_d_train].tolist())
        valid_drugs = set(drugs_shuffled[n_d_train : n_d_train + n_d_valid].tolist())
        test_drugs = set(drugs_shuffled[n_d_train + n_d_valid :].tolist())
        splits = {}
        for name, drug_set in [("train", train_drugs), ("valid", valid_drugs), ("test", test_drugs)]:
            mask = df["Drug_ID"].isin(drug_set)
            splits[name] = df.loc[mask]
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}. "
                         f"Use one of: interpolation, cell_cold, drug_cold")
    return splits


def _get_label_column(df: pd.DataFrame) -> str:
    for name in ("Y", "LN_IC50", "IC50", "Label", "y"):
        if name in df.columns:
            return name
    raise ValueError("Label column not found in dataset.")


# ---------------------------------------------------------------------------
#  ID-based Dataset (lightweight – no graphs, no large vectors)
# ---------------------------------------------------------------------------

class DrugCellDataset(Dataset):
    """Each sample is just three numbers: drug_id, cell_id, label."""

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
#  Global registries (built once, held by CellDrugModel)
# ---------------------------------------------------------------------------

def _build_registries(
    df: pd.DataFrame,
    smiles_series: pd.Series,
    cell_series: pd.Series,
    cfg: DictConfig,
    train_cell_ids: set | None = None,
) -> dict:
    """Build all global lookup tables.

    Returns a dict with keys:
        drug_graphs           – list[Data]  of length N_drugs
        cell_table            – Tensor (N_cells, feat_dim)  on CPU
        drug_id_to_idx        – dict[str, int]
        cell_id_to_idx        – dict[str, int]
        drug_morgan_table     – Tensor (N_drugs, n_bits) or None
        drug_chemberta_table  – Tensor (N_drugs, 768) or None
        pharmacophore_cache   – dict[str, Tensor(N_atoms, P)] or None
    """
    gnn_type = cfg.model.gnn_type.lower()
    with_conformers = (gnn_type == "egnn")
    conformer_cache: dict[str, torch.Tensor] | None = None

    if with_conformers:
        conformer_cache_path = cfg.data.get("conformer_cache_path", "data/conformers.pt")
        if not os.path.exists(conformer_cache_path):
            print(f"Conformer cache not found at {conformer_cache_path}. Auto-generating...")
            script = Path(__file__).parents[2] / "scripts" / "precompute_conformers.py"
            subprocess.run(
                [sys.executable, str(script), "--output", conformer_cache_path, "--workers", "8"],
                check=True,
            )
        print(f"Loading conformer cache from {conformer_cache_path} ...")
        conformer_cache = torch.load(conformer_cache_path, map_location="cpu")
        print(f"  Loaded {len(conformer_cache)} entries")

    # --- Pharmacophore covariance-eigenvalue feature cache --------------------
    use_pharmacophore = cfg.model.get("use_pharmacophore", False)
    pharmacophore_cache: dict[str, torch.Tensor] | None = None
    pharm_feat_dim = cfg.model.get("pharm_feat_dim", 18)
    if use_pharmacophore:
        pharm_cache_path = cfg.data.get("pharmacophore_cache_path", "data/pharmacophores.pt")
        if not os.path.exists(pharm_cache_path):
            print(f"Pharmacophore cache not found at {pharm_cache_path}. Auto-generating...")
            script = Path(__file__).parents[2] / "scripts" / "precompute_pharmacophores.py"
            cmd = [
                sys.executable, str(script),
                "--output", pharm_cache_path,
                "--workers", "8",
                "--num-conformers", str(cfg.data.get("pharm_num_conformers", 30)),
                "--agg", str(cfg.model.get("pharm_agg", "max")),
            ]
            if cfg.model.get("pharm_use_gmm", False):
                cmd.append("--gmm")
            subprocess.run(cmd, check=True)
        print(f"Loading pharmacophore cache from {pharm_cache_path} ...")
        pharmacophore_cache = torch.load(pharm_cache_path, map_location="cpu", weights_only=False)
        print(f"  Loaded {len(pharmacophore_cache)} entries")

    drug_ids = sorted(df["Drug_ID"].unique())
    cell_ids = sorted(df["Cell_Line_ID"].unique())
    drug_id_to_idx = {did: i for i, did in enumerate(drug_ids)}
    cell_id_to_idx = {cid: i for i, cid in enumerate(cell_ids)}

    # --- Drug graphs -----------------------------------------------------------
    print(f"Building drug graphs ({len(drug_ids)} drugs)...")
    drug_graphs: list = []
    drug_smiles: list[str] = []  # collect SMILES for ChemBERTa
    for did in drug_ids:
        first_row = df[df["Drug_ID"] == did].iloc[0]
        smi = str(smiles_series.iloc[int(first_row.name)])
        drug_smiles.append(smi)
        g = smiles_to_graph(
            smi,
            with_conformers=with_conformers,
            conformer_cache=conformer_cache,
            pharmacophore_cache=pharmacophore_cache,
            pharm_feat_dim=pharm_feat_dim,
        )
        if g is None:
            raise ValueError(f"Invalid SMILES for drug {did}: {smi}")
        drug_graphs.append(g)

    # --- Cell feature table ----------------------------------------------------
    print(f"Building cell feature table ({len(cell_ids)} cells)...")
    cell_rows = []
    for cid in cell_ids:
        first_row = df[df["Cell_Line_ID"] == cid].iloc[0]
        feat = np.asarray(cell_series.iloc[int(first_row.name)], dtype=np.float32)
        cell_rows.append(feat)
    cell_table = np.stack(cell_rows, axis=0)  # (N_cells, feat_dim)

    # --- Fit normalization and PCA on train cells only -------------------------
    if train_cell_ids is not None:
        train_mask = np.array([cid in train_cell_ids for cid in cell_ids], dtype=bool)
    else:
        train_mask = np.ones(len(cell_ids), dtype=bool)  # fallback: all cells

    if cfg.data.get("normalize_cell_features", True):
        if train_mask.sum() > 0:
            cell_table_train = cell_table[train_mask]
            cell_table_train, mean, std = normalize_features(cell_table_train)
            cell_table[train_mask] = cell_table_train
            cell_table[~train_mask] = (cell_table[~train_mask] - mean) / std
        else:
            cell_table, _, _ = normalize_features(cell_table)

    cell_pca_dim = cfg.data.get("cell_pca_dim", 0)
    if cell_pca_dim > 0 and cell_pca_dim < cell_table.shape[1]:
        print(f"Reducing cell features from {cell_table.shape[1]} to {cell_pca_dim} via PCA ...")
        if train_mask.sum() > 0:
            cell_table_train = cell_table[train_mask]
            cell_table = _pca_reduce_fit_transform(
                cell_table_train, cell_table, n_components=cell_pca_dim,
            )
        else:
            cell_table = _pca_reduce(cell_table, n_components=cell_pca_dim)
        if train_mask.sum() > 0:
            print(f"  Cell table after PCA: {cell_table.shape}  (fitted on {train_mask.sum()} train cells)")
        else:
            print(f"  Cell table after PCA: {cell_table.shape}")

    cell_table = torch.from_numpy(cell_table)  # stays on CPU, model moves to GPU

    # --- Morgan fingerprint table ----------------------------------------------
    use_morgan = cfg.model.get("use_morgan", False)
    drug_morgan_table: torch.Tensor | None = None
    if use_morgan:
        n_bits = cfg.model.get("morgan_n_bits", 1024)
        radius = cfg.model.get("morgan_radius", 2)
        print(f"Building Morgan fingerprint table ({len(drug_ids)} drugs, "
              f"radius={radius}, n_bits={n_bits})...")
        morgan_rows = []
        for smi in drug_smiles:
            fp = compute_morgan_fingerprint(smi, radius=radius, n_bits=n_bits)
            morgan_rows.append(fp)
        drug_morgan_table = torch.from_numpy(np.stack(morgan_rows, axis=0))
        print(f"  Morgan table shape: {drug_morgan_table.shape}")

    # --- ChemBERTa embedding table ---------------------------------------------
    use_chemberta = cfg.model.get("use_chemberta", False)
    drug_chemberta_table: torch.Tensor | None = None
    if use_chemberta:
        chemberta_cache_path = cfg.data.get("chemberta_cache_path",
                                             "data/chemberta_embeddings.pt")
        if not os.path.exists(chemberta_cache_path):
            print(f"ChemBERTa cache not found at {chemberta_cache_path}. Auto-generating...")
            script = Path(__file__).parents[2] / "scripts" / "precompute_chemberta.py"
            subprocess.run(
                [sys.executable, str(script),
                 "--output", chemberta_cache_path,
                 "--device", "cuda:0" if torch.cuda.is_available() else "cpu"],
                check=True,
            )
        print(f"Loading ChemBERTa cache from {chemberta_cache_path} ...")
        chemberta_cache = torch.load(chemberta_cache_path, map_location="cpu")
        print(f"  Loaded {len(chemberta_cache)} entries")
        bert_rows = []
        for smi in drug_smiles:
            emb = chemberta_cache.get(smi)
            if emb is None:
                emb = torch.zeros(768, dtype=torch.float)
            bert_rows.append(emb)
        drug_chemberta_table = torch.stack(bert_rows, dim=0)  # (N_drugs, 768)
        print(f"  ChemBERTa table shape: {drug_chemberta_table.shape}")

    return {
        "drug_graphs": drug_graphs,
        "cell_table": cell_table,
        "drug_id_to_idx": drug_id_to_idx,
        "cell_id_to_idx": cell_id_to_idx,
        "drug_morgan_table": drug_morgan_table,
        "drug_chemberta_table": drug_chemberta_table,
        "pharmacophore_cache": pharmacophore_cache,
    }


def _pca_reduce(features: np.ndarray, n_components: int) -> np.ndarray:
    """Reduce feature dimensionality via PCA.

    Tries sklearn PCA first, falls back to PyTorch SVD.
    """
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components, random_state=42)
        reduced = pca.fit_transform(features)
        explained = pca.explained_variance_ratio_.sum()
        print(f"  PCA explained variance: {explained:.3f}")
        return reduced.astype(np.float32)
    except ImportError:
        pass

    # Fallback: truncated SVD via PyTorch
    print("  sklearn not available, using PyTorch SVD fallback ...")
    t = torch.from_numpy(features.astype(np.float32))
    U, S, V = torch.linalg.svd(t - t.mean(dim=0), full_matrices=False)
    reduced = (U[:, :n_components] * S[:n_components]).numpy().astype(np.float32)
    explained = (S[:n_components] ** 2).sum() / (S ** 2).sum()
    print(f"  SVD explained variance: {explained:.3f}")
    return reduced


def _pca_reduce_fit_transform(
    train: np.ndarray,
    all_features: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """Fit PCA on *train* cell features, then transform *all* cells.

    This prevents data leakage from validation/test cells into the PCA
    components — the transformation is purely determined by training data.
    """
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(train)
        reduced = pca.transform(all_features)
        explained = pca.explained_variance_ratio_.sum()
        print(f"  PCA explained variance: {explained:.3f}  (fit on {len(train)} train cells)")
        return reduced.astype(np.float32)
    except ImportError:
        pass

    # Fallback: truncated SVD via PyTorch
    print("  sklearn not available, using PyTorch SVD fallback ...")
    t_train = torch.from_numpy(train.astype(np.float32))
    t_mean = t_train.mean(dim=0)
    U, S, V = torch.linalg.svd(t_train - t_mean, full_matrices=False)
    # Transform all
    t_all = torch.from_numpy(all_features.astype(np.float32))
    reduced = ((t_all - t_mean) @ V.T[:, :n_components]).numpy().astype(np.float32)
    explained = (S[:n_components] ** 2).sum() / (S ** 2).sum()
    print(f"  SVD explained variance: {explained:.3f}  (fit on {len(train)} train cells)")
    return reduced


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def get_dataloaders(
    cfg: DictConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, dict, dict]:
    """Return (train_ldr, valid_ldr, test_ldr, registries, meta)."""
    split_mode = cfg.data.get("split_mode", "interpolation")
    seed = cfg.training.seed

    print(f"Split mode: {split_mode}")

    df, smiles_series, cell_series = _load_full_tdc_data(cfg.data.dataset_name)
    print(f"  Full dataset: {len(df):,} pairs, "
          f"{df['Drug_ID'].nunique()} drugs, "
          f"{df['Cell_Line_ID'].nunique()} cell lines")

    # --- Split first, then fit normalizer + PCA on train cells only -----------
    splits = _custom_split(df, split_mode=split_mode, seed=seed)
    train_cell_ids = set(splits["train"]["Cell_Line_ID"].unique())

    # --- Global registries ----------------------------------------------------
    registries = _build_registries(df, smiles_series, cell_series, cfg,
                                   train_cell_ids=train_cell_ids)
    drug_id_to_idx = registries["drug_id_to_idx"]
    cell_id_to_idx = registries["cell_id_to_idx"]
    cell_dim = registries["cell_table"].shape[1]

    label_col = _get_label_column(splits["train"])

    for name in ("train", "valid", "test"):
        s = splits[name]
        print(f"  {name}: {len(s):,} pairs, "
              f"{s['Drug_ID'].nunique()} drugs, "
              f"{s['Cell_Line_ID'].nunique()} cell lines")

    # --- Build ID-based datasets ----------------------------------------------
    datasets: dict[str, DrugCellDataset] = {}
    for name in ("train", "valid", "test"):
        s = splits[name]
        drug_ids = np.array([drug_id_to_idx[str(s.iloc[i]["Drug_ID"])] for i in range(len(s))], dtype=np.int64)
        cell_ids = np.array([cell_id_to_idx[str(s.iloc[i]["Cell_Line_ID"])] for i in range(len(s))], dtype=np.int64)
        labels = s[label_col].values.astype(np.float32)
        datasets[name] = DrugCellDataset(drug_ids, cell_ids, labels)

    # --- DataLoaders ----------------------------------------------------------
    nw = cfg.data.num_workers
    loader_kwargs = {
        "batch_size": cfg.data.batch_size,
        "num_workers": nw,
        "pin_memory": True,
        "collate_fn": collate_drug_cell,
        "persistent_workers": nw > 0,
    }
    train_loader = DataLoader(datasets["train"], shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(datasets["valid"], shuffle=False, **loader_kwargs)
    test_loader = DataLoader(datasets["test"], shuffle=False, **loader_kwargs)

    registries["train_df"] = splits["train"]  # raw DataFrame for meta-learning

    meta = {"cell_dim": cell_dim, "label_col": label_col}
    return train_loader, valid_loader, test_loader, registries, meta
