"""SMILES to molecular graph conversion and cell feature utilities."""

from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

# Atom feature dimensions (matches common OGB / PyG molecular encodings)
ATOM_FEATURES = {
    "atomic_num": list(range(1, 119)),
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-2, -1, 0, 1, 2],
    "chiral_tag": [0, 1, 2, 3],
    "num_Hs": [0, 1, 2, 3, 4],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}

# SMARTS patterns for substructure matching (pre-compiled)
_PURINE_SMARTS = Chem.MolFromSmarts("c1ncnc2[nH]cnc12")
_PYRIMIDINE_SMARTS = Chem.MolFromSmarts("c1cncnc1")


def _one_hot(value, choices: list) -> list[float]:
    encoding = [0.0] * len(choices)
    try:
        idx = choices.index(value)
        encoding[idx] = 1.0
    except ValueError:
        pass
    return encoding


def _atom_extra_features(atom: Chem.Atom, mol: Chem.Mol) -> list[float]:
    """Compute ~14 additional atom-level chemical descriptors (binary flags).

    Uses RDKit atom API and pre-compiled SMARTS for purine/pyrimidine detection.
    """
    anum = atom.GetAtomicNum()
    idx = atom.GetIdx()
    ring_info = mol.GetRingInfo()

    # 1. H-bond donor: N or O with attached H
    hbd = 1.0 if (anum in (7, 8) and atom.GetTotalNumHs() > 0) else 0.0

    # 2. H-bond acceptor: N, O, S
    hba = 1.0 if anum in (7, 8, 16) else 0.0

    # 3. Is heteroatom (non C, non H)
    hetero = 1.0 if anum not in (1, 6) else 0.0

    # 4. In carbonyl group: C or O participates in C=O
    in_carbonyl = 0.0
    for nb in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(idx, nb.GetIdx())
        if bond is not None and bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            paired_nums = {anum, nb.GetAtomicNum()}
            if paired_nums == {6, 8}:  # C=O
                in_carbonyl = 1.0
                break

    # 5. In double bond: any bond is double
    in_double = 0.0
    # 6. In triple bond: any bond is triple
    in_triple = 0.0
    for bond in atom.GetBonds():
        bt = bond.GetBondType()
        if bt == Chem.rdchem.BondType.DOUBLE:
            in_double = 1.0
        elif bt == Chem.rdchem.BondType.TRIPLE:
            in_triple = 1.0

    # 7. In heterocycle: atom is in a ring containing N/O/S
    in_heterocycle = 0.0
    atom_rings = ring_info.AtomRings()
    for ring_atoms in atom_rings:
        if idx in ring_atoms:
            ring_nums = {mol.GetAtomWithIdx(ra).GetAtomicNum() for ra in ring_atoms}
            if ring_nums.intersection({7, 8, 16}):  # N, O, S present
                in_heterocycle = 1.0
                break

    # 8. In 5-membered ring
    in_ring5 = 1.0 if ring_info.IsAtomInRingOfSize(idx, 5) else 0.0

    # 9. In 6-membered ring
    in_ring6 = 1.0 if ring_info.IsAtomInRingOfSize(idx, 6) else 0.0

    # 10. In fused ring: atom belongs to multiple distinct rings
    ring_count = sum(1 for ra in ring_info.AtomRings() if idx in ra)
    in_fused = 1.0 if ring_count >= 2 else 0.0

    # 11. In purine scaffold (SMARTS match)
    in_purine = 1.0 if mol.HasSubstructMatch(_PURINE_SMARTS) and idx in {
        a for match in mol.GetSubstructMatches(_PURINE_SMARTS) for a in match
    } else 0.0

    # 12. In pyrimidine (SMARTS match)
    in_pyrimidine = 1.0 if mol.HasSubstructMatch(_PYRIMIDINE_SMARTS) and idx in {
        a for match in mol.GetSubstructMatches(_PYRIMIDINE_SMARTS) for a in match
    } else 0.0

    # 13. Has explicit H
    has_explicit_h = 1.0 if atom.GetNumExplicitHs() > 0 else 0.0

    # 14. Is in any ring (non-aromatic ring detection; aromaticity already captured)
    in_ring = 1.0 if atom.IsInRing() else 0.0

    return [
        hbd,
        hba,
        hetero,
        in_carbonyl,
        in_double,
        in_triple,
        in_heterocycle,
        in_ring5,
        in_ring6,
        in_fused,
        in_purine,
        in_pyrimidine,
        has_explicit_h,
        in_ring,
    ]


def atom_features(atom: Chem.Atom, mol: Chem.Mol | None = None) -> list[float]:
    """Encode a single atom as a fixed-length feature vector (144 + 14 = 158 dims)."""
    feats = []
    feats += _one_hot(atom.GetAtomicNum(), ATOM_FEATURES["atomic_num"])
    feats += _one_hot(atom.GetTotalDegree(), ATOM_FEATURES["degree"])
    feats += _one_hot(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
    feats += _one_hot(int(atom.GetChiralTag()), ATOM_FEATURES["chiral_tag"])
    feats += _one_hot(atom.GetTotalNumHs(), ATOM_FEATURES["num_Hs"])
    feats += _one_hot(atom.GetHybridization(), ATOM_FEATURES["hybridization"])
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)
    if mol is not None:
        feats += _atom_extra_features(atom, mol)
    return feats


def bond_features(bond: Chem.Bond) -> list[float]:
    bt = bond.GetBondType()
    return [
        float(bt == Chem.rdchem.BondType.SINGLE),
        float(bt == Chem.rdchem.BondType.DOUBLE),
        float(bt == Chem.rdchem.BondType.TRIPLE),
        float(bt == Chem.rdchem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def get_atom_feature_dim() -> int:
    """Return atom feature dimension for a dummy atom."""
    mol = Chem.MolFromSmiles("C")
    mol = Chem.AddHs(mol)
    return len(atom_features(mol.GetAtomWithIdx(0), mol))


def get_bond_feature_dim() -> int:
    mol = Chem.MolFromSmiles("C=C")
    return len(bond_features(mol.GetBondBetweenAtoms(0, 1)))


def smiles_to_graph(
    smiles: str,
    mol: Chem.Mol | None = None,
    with_conformers: bool = False,
    conformer_cache: dict[str, torch.Tensor] | None = None,
    pharmacophore_cache: dict[str, torch.Tensor] | None = None,
    pharm_feat_dim: int = 18,
) -> Data | None:
    """
    Convert SMILES string to PyG Data object.

    Accepts an optional pre-built mol (e.g., from SMILES augmentation)
    to avoid redundant RDKit parsing. Returns None if SMILES is invalid.

    Args:
        smiles: SMILES string.
        mol: Optional pre-built RDKit Mol (from augmentation).
        with_conformers: If True, generate or load a 3D conformer.
        conformer_cache: Pre-computed dict mapping SMILES -> (N,3) tensor.
            When provided and with_conformers=True, uses the cache instead
            of on-the-fly ETKDG+MMFF.
        pharmacophore_cache: Pre-computed dict mapping SMILES -> (N, P) tensor
            of per-atom pharmacophore covariance-eigenvalue features.
        pharm_feat_dim: Feature width P used when zero-filling on cache miss.
    """
    if mol is None:
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # --- 3D conformer (EGNN) ---
    pos = None
    if with_conformers:
        if conformer_cache is not None and smiles in conformer_cache:
            pos = conformer_cache[smiles]
        else:
            # Fallback: generate on the fly (slow)
            try:
                from rdkit.Chem import AllChem
                status = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                if status == 0:
                    AllChem.MMFFOptimizeMolecule(mol)
                    conf = mol.GetConformer()
                    pos = torch.tensor(
                        [[conf.GetAtomPosition(i).x,
                          conf.GetAtomPosition(i).y,
                          conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())],
                        dtype=torch.float,
                    )
            except Exception:
                pass
            if pos is None:
                pos = torch.zeros(mol.GetNumAtoms(), 3)

    x_list = [atom_features(atom, mol) for atom in mol.GetAtoms()]
    x = torch.tensor(x_list, dtype=torch.float)

    edge_index: list[list[int]] = [[], []]
    edge_attr_list: list[list[float]] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index[0] += [i, j]
        edge_index[1] += [j, i]
        edge_attr_list.append(bf)
        edge_attr_list.append(bf)

    if len(edge_index[0]) == 0:
        # Single atom molecule
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, get_bond_feature_dim()), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = x.size(0)
    if pos is not None:
        data.pos = pos

    # --- Pharmacophore covariance-eigenvalue features -----------------------
    # Every graph gets a data.pharm attribute so PyG Batch keeps it uniform:
    # cached features on hit, zero-filled on miss / shape mismatch.
    if pharmacophore_cache is not None:
        pharm = pharmacophore_cache.get(smiles)
        if pharm is not None and pharm.shape[0] == x.size(0):
            data.pharm = pharm
        else:
            if pharm is not None:
                print(f"[warn] pharmacophore shape mismatch for {smiles}: "
                      f"{pharm.shape[0]} != {x.size(0)}; zero-filling")
            data.pharm = x.new_zeros(x.size(0), pharm_feat_dim)
    return data


def compute_morgan_fingerprint(
    smiles: str,
    radius: int = 2,
    n_bits: int = 1024,
) -> np.ndarray:
    """Compute ECFP (Morgan) circular fingerprint as a fixed-length bit vector.

    Returns a float32 array of shape (n_bits,). Invalid SMILES produces all-zero.
    """
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


def normalize_features(matrix: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    """Z-score normalize feature matrix along columns."""
    if mean is None:
        mean = matrix.mean(axis=0)
    if std is None:
        std = matrix.std(axis=0)
        std[std < 1e-8] = 1.0
    normalized = (matrix - mean) / std
    return normalized.astype(np.float32), mean, std
