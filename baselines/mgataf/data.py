"""Independent data pipeline for MGATAF baseline.

Fully self-contained: 78-dim atom features (paper spec), GDSC2 data loading
via TDC, 735-dim binary genomic cell features (with gene-expression fallback),
min-max label normalization, and train/valid/test splits.

Zero dependencies on src/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# 78-dim atom features — faithfully matching Saeed et al. (2025) Table 1
#
#   Feature          Dim   Description
#   ───────────────  ────  ────────────────────────────────────────────
#   Atom type         44   44 common elements (one-hot)
#   Degree            11   0–10 (one-hot)
#   Implicit valence   7   0–6  (one-hot)
#   Formal charge      1   integer scalar
#   Radical electrons  1   integer scalar
#   Hybridization      5   SP, SP2, SP3, SP3D, SP3D2 (one-hot or null)
#   Aromatic           1   binary
#   Hydrogens          5   0–4 (one-hot)
#   Ring               1   binary
#   Chirality          2   R, S (one-hot or null)
#   ───────────────  ────
#   Total             78
# ---------------------------------------------------------------------------

# Atom types commonly found in drug-like molecules (44 types)
_ATOM_TYPES = [
    1, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22,
    23, 24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 37, 38, 39, 42, 47,
    48, 50, 51, 52, 53, 55, 78, 79, 82, 83,
]

# Hybridization mapping: RDKit → paper’s 5 categories
_HYBRIDIZATION_MAP = {
    "SP":    0,
    "SP2":   1,
    "SP3":   2,
    "SP3D":  3,
    "SP3D2": 4,
}

_ATOM_DIM = 78


def _one_hot_list(value, choices: list) -> list[float]:
    """One-hot encode value into the given choices list; all zeros if missing."""
    encoding = [0.0] * len(choices)
    try:
        encoding[choices.index(value)] = 1.0
    except ValueError:
        pass
    return encoding


def _one_hot_range(value, max_val: int) -> list[float]:
    """One-hot encode value into [0, max_val-1], clipped."""
    encoding = [0.0] * max_val
    v = max(0, min(int(value), max_val - 1))
    encoding[v] = 1.0
    return encoding


def atom_features_78(atom, mol) -> list[float]:
    """Compute 78-dim atom features exactly matching MGATAF paper Table 1."""
    feats: list[float] = []

    # 1. Atom type — 44 dim
    feats += _one_hot_list(atom.GetAtomicNum(), _ATOM_TYPES)

    # 2. Degree — 11 dim (0–10)
    feats += _one_hot_range(atom.GetTotalDegree(), 11)

    # 3. Implicit valence — 7 dim (0–6)
    feats += _one_hot_range(atom.GetImplicitValence(), 7)

    # 4. Formal charge — 1 dim (integer scalar, clipped to [-6, 6])
    fc = max(-6, min(int(atom.GetFormalCharge()), 6))
    feats.append(float(fc))

    # 5. Radical electrons — 1 dim (integer scalar)
    feats.append(float(atom.GetNumRadicalElectrons()))

    # 6. Hybridization — 5 dim (SP/SP2/SP3/SP3D/SP3D2, one-hot or null)
    hyb_encoding = [0.0] * 5
    hyb_str = str(atom.GetHybridization())
    if hyb_str in _HYBRIDIZATION_MAP:
        hyb_encoding[_HYBRIDIZATION_MAP[hyb_str]] = 1.0
    feats += hyb_encoding

    # 7. Aromatic — 1 dim (binary)
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)

    # 8. Hydrogens — 5 dim (0–4, one-hot)
    feats += _one_hot_range(atom.GetTotalNumHs(), 5)

    # 9. Ring — 1 dim (binary)
    feats.append(1.0 if atom.IsInRing() else 0.0)

    # 10. Chirality — 2 dim (R/S, one-hot or null)
    chiral_encoding = [0.0] * 2
    tag = atom.GetChiralTag()
    from rdkit.Chem import ChiralType
    if tag == ChiralType.CHI_TETRAHEDRAL_CW:
        chiral_encoding[0] = 1.0  # R (clockwise)
    elif tag == ChiralType.CHI_TETRAHEDRAL_CCW:
        chiral_encoding[1] = 1.0  # S (counter-clockwise)
    feats += chiral_encoding

    assert len(feats) == 78, f"Expected 78 features, got {len(feats)}"
    return feats


def smiles_to_graph_78(smiles: str) -> "Data | None":
    """Convert SMILES to PyG Data with 78-dim atom features."""
    from rdkit import Chem
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
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
# Morgan fingerprint
# ---------------------------------------------------------------------------

def compute_morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """Compute ECFP fingerprint as a fixed-length bit vector."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    for i in range(n_bits):
        if fp[i]:
            arr[i] = 1.0
    return arr


# ---------------------------------------------------------------------------
# Genomic cell features (735-dim binary aberrations)
# ---------------------------------------------------------------------------

def _load_genomic_features(genomic_path: str, cell_ids: list) -> np.ndarray | None:
    """Load 735-dim binary genomic aberrations from a pre-built CSV.

    The CSV should have a 'Cell_Line_ID' column and 735 gene columns with
    binary (0/1) values. Returns (N_cells, 735) array or None if the file
    is unavailable.
    """
    path = Path(genomic_path)
    if not path.exists():
        print(f"  Genomic features not found at {genomic_path}. "
              f"Falling back to gene expression.")
        return None
    df = pd.read_csv(path)
    # Build mapping from Cell_Line_ID to row
    gene_cols = [c for c in df.columns if c != "Cell_Line_ID"]
    print(f"  Loaded genomic features: {len(df)} cell lines, {len(gene_cols)} genes")
    id_to_row = {str(row["Cell_Line_ID"]): i for i, row in df.iterrows()}
    features = np.zeros((len(cell_ids), len(gene_cols)), dtype=np.float32)
    missing = 0
    for i, cid in enumerate(cell_ids):
        row_idx = id_to_row.get(str(cid))
        if row_idx is not None:
            features[i] = df.iloc[row_idx][gene_cols].values.astype(np.float32)
        else:
            missing += 1
    if missing > 0:
        print(f"  Warning: {missing}/{len(cell_ids)} cell lines missing from genomic data")
    return features


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


def _pca_reduce(features: np.ndarray, n_components: int) -> np.ndarray:
    """Reduce feature dimensionality via PCA.

    Tries sklearn PCA first, falls back to PyTorch SVD.
    """
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components, random_state=42)
        reduced = pca.fit_transform(features)
        explained = pca.explained_variance_ratio_.sum()
        print(f"    PCA explained variance: {explained:.3f}")
        return reduced.astype(np.float32)
    except ImportError:
        pass

    # Fallback: truncated SVD via PyTorch
    print("    sklearn not available, using PyTorch SVD fallback ...")
    t = torch.from_numpy(features.astype(np.float32))
    U, S, V = torch.linalg.svd(t - t.mean(dim=0), full_matrices=False)
    reduced = (U[:, :n_components] * S[:n_components]).numpy().astype(np.float32)
    explained = (S[:n_components] ** 2).sum() / (S ** 2).sum()
    print(f"    SVD explained variance: {explained:.3f}")
    return reduced


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_mgataf_dataloaders(cfg: DictConfig,
                           train_df: pd.DataFrame | None = None,
                           valid_df: pd.DataFrame | None = None,
                           test_df: pd.DataFrame | None = None
                           ) -> tuple[DataLoader, DataLoader, DataLoader, dict, dict]:
    """Return (train_ldr, valid_ldr, test_ldr, registries, meta).

    registries contains:
        drug_graphs: list[Data]  (78-dim atom features)
        cell_table:  Tensor (N_cells, cell_dim)
        drug_id_to_idx: dict
        cell_id_to_idx: dict
        drug_fp_table: Tensor (N_drugs, 1024)
        label_scaler:  dict or None  (for inverse-transform during eval)

    When train_df / valid_df / test_df are given, the internal _custom_split is
    skipped and those exact splits are used instead (for K-fold CV).
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

    # --- Split ---
    if train_df is not None and valid_df is not None:
        splits = {"train": train_df, "valid": valid_df, "test": test_df if test_df is not None else valid_df}
    else:
        splits = _custom_split(df, split_mode=split_mode, seed=seed, frac=(0.8, 0.1, 0.1))
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
    drug_smiles: list[str] = []
    for did in drug_ids:
        first_row = df[df["Drug_ID"] == did].iloc[0]
        smi = str(smiles_series.iloc[int(first_row.name)])
        drug_smiles.append(smi)
        g = smiles_to_graph_78(smi)
        if g is None:
            raise ValueError(f"Invalid SMILES for drug {did}: {smi}")
        drug_graphs.append(g)

    # --- Morgan fingerprint table ---
    n_bits = 1024
    print(f"Building Morgan fingerprint table ({len(drug_ids)} drugs, {n_bits} bits)...")
    morgan_rows = [compute_morgan_fingerprint(smi, n_bits=n_bits) for smi in drug_smiles]
    drug_fp_table = torch.from_numpy(np.stack(morgan_rows, axis=0))

    # --- Cell feature table ---
    print(f"Building cell feature table ({len(cell_ids)} cells)...")
    genomic_path = cfg.data.get("genomic_path", "data/gdsc_genomic.csv")
    cell_features = _load_genomic_features(genomic_path, cell_ids)

    if cell_features is None:
        # Fallback: use gene expression from TDC (raw, no PCA)
        print(f"  Using raw gene expression as fallback ({cell_series.iloc[0].shape[0]} dim)...")
        cell_rows = []
        for cid in cell_ids:
            first_row = df[df["Cell_Line_ID"] == cid].iloc[0]
            feat = np.asarray(cell_series.iloc[int(first_row.name)], dtype=np.float32)
            cell_rows.append(feat)
        cell_features = np.stack(cell_rows, axis=0)

    # Z-score normalize cell features
    mean = cell_features.mean(axis=0)
    std = cell_features.std(axis=0)
    std[std < 1e-8] = 1.0
    cell_features = ((cell_features - mean) / std).astype(np.float32)
    print(f"  Cell table shape: {cell_features.shape}")

    # --- Optional PCA reduction (for high-dim gene expression fallback) ---
    cell_pca_dim = cfg.data.get("cell_pca_dim", 0)
    if cell_pca_dim > 0 and cell_pca_dim < cell_features.shape[1]:
        print(f"  Applying PCA: {cell_features.shape[1]} -> {cell_pca_dim} ...")
        cell_features = _pca_reduce(cell_features, n_components=cell_pca_dim)
        print(f"  Cell table shape after PCA: {cell_features.shape}")

    cell_table = torch.from_numpy(cell_features)

    # --- Label normalization (min-max to [0,1]) ---
    label_col = _get_label_column(splits["train"])
    train_labels = splits["train"][label_col].values.astype(np.float32)
    y_min = float(train_labels.min())
    y_max = float(train_labels.max())
    span = y_max - y_min
    if span < 1e-12:
        span = 1.0
    label_scaler = {"min": y_min, "max": y_max, "span": span}
    print(f"Label normalization: y' = (y - {y_min:.4f}) / {span:.4f}  "
          f"(train range [{y_min:.4f}, {y_max:.4f}])")

    # --- Build datasets ---
    datasets: dict[str, DrugCellDataset] = {}
    for name in ("train", "valid", "test"):
        s = splits[name]
        dr_ids = np.array([drug_id_to_idx[str(s.iloc[i]["Drug_ID"])] for i in range(len(s))], dtype=np.int64)
        cl_ids = np.array([cell_id_to_idx[str(s.iloc[i]["Cell_Line_ID"])] for i in range(len(s))], dtype=np.int64)
        labs = s[label_col].values.astype(np.float32)
        labs = (labs - label_scaler["min"]) / label_scaler["span"]
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
        "drug_fp_table": drug_fp_table,
        "label_scaler": label_scaler,
        "train_df": splits["train"],
    }
    meta = {"cell_dim": int(cell_table.shape[1]), "label_col": label_col}
    return train_loader, valid_loader, test_loader, registries, meta
