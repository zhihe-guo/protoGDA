"""Independent data pipeline for GraphDRP baseline.

Fully self-contained: 78-dim atom features, GDSC2 data loading via TDC,
raw gene expression cell features (no PCA), no label normalization,
train/valid/test splits.

Zero dependencies on src/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# 78-dim atom features (RDKit-based, matching original GraphDRP paper)
# ---------------------------------------------------------------------------

_ATOM_TYPES = [
    1, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22,
    23, 24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 37, 38, 39, 42, 47,
    48, 50, 51, 52, 53, 55, 78, 79, 82, 83,
]


def _one_hot_78(value, choices) -> list[float]:
    encoding = [0.0] * len(choices)
    try:
        idx = choices.index(value)
        encoding[idx] = 1.0
    except ValueError:
        pass
    return encoding


def _one_hot_range(value, max_val: int) -> list[float]:
    encoding = [0.0] * max_val
    v = min(max(int(value), 0), max_val - 1)
    encoding[v] = 1.0
    return encoding


def atom_features_78(atom, mol):
    """Compute 78-dim atom features: type(44) + degree(11) + Hs(11) + valence(11) + aromatic(1)."""
    feats: list[float] = []
    feats += _one_hot_78(atom.GetAtomicNum(), _ATOM_TYPES)
    feats += _one_hot_range(atom.GetTotalDegree(), 11)
    feats += _one_hot_range(atom.GetTotalNumHs(), 11)
    feats += _one_hot_range(atom.GetImplicitValence(), 11)
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)
    return feats


def smiles_to_graph_78(smiles: str) -> "Data | None":
    """Convert SMILES to PyG Data with 78-dim atom features."""
    from rdkit import Chem
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    x_list = [atom_features_78(atom, mol) for atom in mol.GetAtoms()]
    x = torch.tensor(x_list, dtype=torch.float)

    edge_index: list[list[int]] = [[], []]
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index[0] += [i, j]
        edge_index[1] += [j, i]

    if len(edge_index[0]) == 0:
        edge_index_t = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index_t = torch.tensor(edge_index, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index_t)
    data.num_nodes = x.size(0)
    return data


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DrugCellDataset(Dataset):
    """Lightweight ID-based dataset: (drug_id, cell_id, label)."""

    def __init__(self, drug_ids: np.ndarray, cell_ids: np.ndarray, labels: np.ndarray):
        self.drug_ids = drug_ids.astype(np.int64)
        self.cell_ids = cell_ids.astype(np.int64)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.drug_ids)

    def __getitem__(self, idx: int) -> dict:
        return {
            "drug_id": int(self.drug_ids[idx]),
            "cell_id": int(self.cell_ids[idx]),
            "label":  float(self.labels[idx]),
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "drug_ids": torch.tensor([b["drug_id"] for b in batch], dtype=torch.long),
        "cell_ids": torch.tensor([b["cell_id"] for b in batch], dtype=torch.long),
        "labels":   torch.tensor([b["label"]  for b in batch], dtype=torch.float),
    }


# ---------------------------------------------------------------------------
# Custom splits
# ---------------------------------------------------------------------------

def _custom_split(df: pd.DataFrame, split_mode: str, seed: int,
                  frac: tuple = (0.7, 0.1, 0.2)) -> dict[str, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    drugs = df["Drug_ID"].unique()
    cells = df["Cell_Line_ID"].unique()

    if split_mode == "interpolation":
        indices = rng.permutation(len(df))
        n_train = int(len(df) * frac[0])
        n_valid = int(len(df) * frac[1])
        train_idx = set(indices[:n_train].tolist())
        valid_idx = set(indices[n_train:n_train + n_valid].tolist())
        test_idx = set(indices[n_train + n_valid:].tolist())
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
        valid_cells = set(cells_shuffled[n_c_train:n_c_train + n_c_valid].tolist())
        test_cells = set(cells_shuffled[n_c_train + n_c_valid:].tolist())
        splits = {}
        for name, cell_set in [("train", train_cells), ("valid", valid_cells), ("test", test_cells)]:
            mask = df["Cell_Line_ID"].isin(cell_set)
            splits[name] = df.loc[mask]
    elif split_mode == "drug_cold":
        drugs_shuffled = drugs.copy(); rng.shuffle(drugs_shuffled)
        n_d_train = max(int(len(drugs) * frac[0]), 1)
        n_d_valid = max(int(len(drugs) * frac[1]), 1)
        train_drugs = set(drugs_shuffled[:n_d_train].tolist())
        valid_drugs = set(drugs_shuffled[n_d_train:n_d_train + n_d_valid].tolist())
        test_drugs = set(drugs_shuffled[n_d_train + n_d_valid:].tolist())
        splits = {}
        for name, drug_set in [("train", train_drugs), ("valid", valid_drugs), ("test", test_drugs)]:
            mask = df["Drug_ID"].isin(drug_set)
            splits[name] = df.loc[mask]
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")
    return splits


def _get_label_column(df: pd.DataFrame) -> str:
    for name in ("Y", "LN_IC50", "IC50", "Label", "y"):
        if name in df.columns:
            return name
    raise ValueError("Label column not found in dataset.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_graphdrp_dataloaders(
    cfg: DictConfig,
    train_df: pd.DataFrame | None = None,
    valid_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict, dict]:
    """Return (train_ldr, valid_ldr, test_ldr, registries, meta).

    No label normalization. Raw gene expression (no PCA). 78-dim atoms.
    """
    from tdc.multi_pred import DrugRes

    split_mode = cfg.data.get("split_mode", "interpolation")
    seed = cfg.training.seed

    print(f"Split mode: {split_mode}")
    data_obj = DrugRes(name=cfg.data.dataset_name)
    df = pd.DataFrame({
        "Drug_ID": data_obj.entity1_idx.values,
        "Cell_Line_ID": data_obj.entity2_idx.values,
        "Y": data_obj.y.values,
    })
    smiles_series = data_obj.entity1.copy()
    cell_series = data_obj.entity2.copy()
    print(f"  Full dataset: {len(df):,} pairs, "
          f"{df['Drug_ID'].nunique()} drugs, "
          f"{df['Cell_Line_ID'].nunique()} cell lines")

    # --- Split (or inject the canonical v3 fold manifest) ---
    injected = (train_df, valid_df, test_df)
    if any(part is not None for part in injected):
        if not all(part is not None for part in injected):
            raise ValueError("train_df, valid_df, and test_df must be provided together.")
        splits = {"train": train_df.copy(), "valid": valid_df.copy(), "test": test_df.copy()}
    else:
        splits = _custom_split(df, split_mode=split_mode, seed=seed)
    for name in ("train", "valid", "test"):
        s = splits[name]
        print(f"  {name}: {len(s):,} pairs, "
              f"{s['Drug_ID'].nunique()} drugs, "
              f"{s['Cell_Line_ID'].nunique()} cell lines")

    # --- Build ID mappings ---
    drug_ids = sorted(df["Drug_ID"].unique())
    cell_ids = sorted(df["Cell_Line_ID"].unique())
    drug_id_to_idx = {did: i for i, did in enumerate(drug_ids)}
    cell_id_to_idx = {cid: i for i, cid in enumerate(cell_ids)}

    # --- Drug graphs (78-dim) ---
    print(f"Building drug graphs ({len(drug_ids)} drugs, 78-dim atoms)...")
    drug_graphs: list = []
    for did in drug_ids:
        first_row = df[df["Drug_ID"] == did].iloc[0]
        smi = str(smiles_series.iloc[int(first_row.name)])
        g = smiles_to_graph_78(smi)
        if g is None:
            raise ValueError(f"Invalid SMILES for drug {did}: {smi}")
        drug_graphs.append(g)

    # --- Cell feature table (raw gene expression, no PCA) ---
    print(f"Building cell feature table ({len(cell_ids)} cells, raw expression)...")
    cell_rows = []
    for cid in cell_ids:
        first_row = df[df["Cell_Line_ID"] == cid].iloc[0]
        feat = np.asarray(cell_series.iloc[int(first_row.name)], dtype=np.float32)
        cell_rows.append(feat)
    cell_features = np.stack(cell_rows, axis=0)

    # Fit normalization only on training cells. This is material in cell-cold CV.
    train_cells = set(splits["train"]["Cell_Line_ID"].unique())
    train_mask = np.array([cid in train_cells for cid in cell_ids], dtype=bool)
    if not train_mask.any():
        raise ValueError("Training split contains no cells for feature normalization.")
    mean = cell_features[train_mask].mean(axis=0)
    std = cell_features[train_mask].std(axis=0)
    std[std < 1e-8] = 1.0
    cell_features = ((cell_features - mean) / std).astype(np.float32)
    print(f"  Cell table shape: {cell_features.shape}")

    cell_table = torch.from_numpy(cell_features)

    # --- Build datasets (NO label normalization) ---
    label_col = _get_label_column(splits["train"])
    datasets: dict[str, DrugCellDataset] = {}
    for name in ("train", "valid", "test"):
        s = splits[name]
        dr_ids = np.array([drug_id_to_idx[str(s.iloc[i]["Drug_ID"])] for i in range(len(s))], dtype=np.int64)
        cl_ids = np.array([cell_id_to_idx[str(s.iloc[i]["Cell_Line_ID"])] for i in range(len(s))], dtype=np.int64)
        labs = s[label_col].values.astype(np.float32)
        datasets[name] = DrugCellDataset(dr_ids, cl_ids, labs)

    # --- DataLoaders ---
    nw = cfg.data.num_workers
    loader_kwargs = {
        "batch_size": cfg.data.batch_size,
        "num_workers": nw,
        "pin_memory": True,
        "collate_fn": collate_fn,
        "persistent_workers": nw > 0,
    }
    train_loader = DataLoader(datasets["train"], shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(datasets["valid"], shuffle=False, **loader_kwargs)
    test_loader = DataLoader(datasets["test"], shuffle=False, **loader_kwargs)

    registries = {
        "drug_graphs": drug_graphs,
        "cell_table": cell_table,
        "drug_id_to_idx": drug_id_to_idx,
        "cell_id_to_idx": cell_id_to_idx,
        "train_df": splits["train"],
        "cell_preprocessing": {"mean": mean, "std": std, "fit_cell_ids": sorted(map(str, train_cells))},
    }
    meta = {"cell_dim": int(cell_table.shape[1]), "label_col": label_col}
    return train_loader, valid_loader, test_loader, registries, meta
