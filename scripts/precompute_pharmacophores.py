#!/usr/bin/env python
"""Precompute per-atom pharmacophore "Gaussian-mixture covariance eigenvalue" features.

For each SMILES:
  1. Sample `--num-conformers` 3D conformers via ETKDG + MMFF optimization.
  2. Align all conformers to the reference via AlignMolConformers (heavy-atom
     pruned-RMSD iterative alignment), so flexible tail atoms cannot skew the
     rigid scaffold.
  3. Detect pharmacophore features with RDKit FeatureFactory (BaseFeatures.fdef),
     normalizing to 6 families: Donor / Acceptor / Aromatic / Hydrophobe /
     PosIonizable / NegIonizable.
  4. For each feature instance, collect its centroid position across all
     conformers and compute the empirical 3x3 covariance — the MLE of a Gaussian
     component given a fixed assignment — plus a small ridge term.
  5. Sort the covariance eigenvalues descending and normalize with
     f = log(1 + sqrt(max(lambda, 0))) to keep values in ~[0, 3].
  6. Map each instance back to its (heavy) atom ids, aggregating per-family
     blocks with max (default) or mean.

Output: {smiles: torch.Tensor(N_atoms, 18)} saved to data/pharmacophores.pt
where N_atoms is the number of atoms AFTER AddHs, matching the node count of
`src/data/preprocessing.smiles_to_graph`.

Usage:
    python scripts/precompute_pharmacophores.py \\
        --output data/pharmacophores.pt \\
        --num-conformers 30 \\
        --workers 8
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Pharmacophore families we keep, in fixed order. Each contributes 3 slots
# (the 3 sorted covariance eigenvalues) -> total feature dim = 6 * 3 = 18.
TARGET_FAMILIES = [
    "Donor",
    "Acceptor",
    "Aromatic",
    "Hydrophobe",
    "PosIonizable",
    "NegIonizable",
]

# Map RDKit BaseFeatures.fdef family names onto the target families.
FAMILY_ALIASES = {
    "Donor": "Donor",
    "Acceptor": "Acceptor",
    "Aromatic": "Aromatic",
    "Hydrophobe": "Hydrophobe",
    "LumpedHydrophobe": "Hydrophobe",
    "PosIonizable": "PosIonizable",
    "NegIonizable": "NegIonizable",
    # uncommon aliases, mapped for robustness
    "HBD": "Donor",
    "HBA": "Acceptor",
    "PiRings": "Aromatic",
    "Base": "PosIonizable",
    "Acid": "NegIonizable",
}

NUM_SLOTS = len(TARGET_FAMILIES) * 3  # 18

_RIDGE = 1e-6
_GMM_REG_COVAR = 1e-4


def _build_feature_factory():
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures

    fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    return ChemicalFeatures.BuildFeatureFactory(fdef)


_FACTORY = None


def _get_factory():
    global _FACTORY
    if _FACTORY is None:
        _FACTORY = _build_feature_factory()
    return _FACTORY


def _heavy_atom_ids(atom_ids: tuple[int, ...], anum: list[int], n_atoms: int) -> list[int]:
    """Filter feature atom ids to heavy atoms within bounds (defense, pitfall #1)."""
    return [a for a in atom_ids if a < n_atoms and anum[a] != 1]


def _normalize(eigvals: np.ndarray) -> np.ndarray:
    """f = log(1 + sqrt(max(lambda, 0))) — scale to ~[0, 3] (pitfall #4)."""
    return np.log1p(np.sqrt(np.maximum(eigvals, 0.0)))


def _instance_points(
    coords: np.ndarray,
    atom_ids: tuple[int, ...],
    anum: list[int],
    n_atoms: int,
) -> np.ndarray | None:
    """Centroid of a feature instance's heavy atoms across all conformers.

    coords: (n_conf, N, 3). Returns (n_conf, 3) or None if no heavy atoms.
    """
    ids = _heavy_atom_ids(atom_ids, anum, n_atoms)
    if not ids:
        return None
    return coords[:, ids, :].mean(axis=1)


def _empirical_eigvals(pts: np.ndarray) -> np.ndarray | None:
    """Empirical covariance of the instance centroid (Gaussian-component MLE).

    Returns the 3 normalized eigenvalues, or None when there are too few
    conformers to estimate a covariance (pitfall #2).
    """
    n_conf = pts.shape[0]
    if n_conf < 2:
        # Degenerate: nothing moves. Report a tiny constant (ridge only).
        return np.full(3, np.log1p(np.sqrt(_RIDGE)), dtype=np.float32)
    cov = np.cov(pts, rowvar=False) + np.eye(3) * _RIDGE
    eig = np.linalg.eigvalsh(cov)[::-1]  # descending λ1 >= λ2 >= λ3
    return _normalize(eig)


def _gmm_eigvals(instances_pts: list[np.ndarray]) -> list[np.ndarray] | None:
    """Optional sklearn GMM path (off by default).

    Fits one GaussianMixture per family over all instance centroids, with
    means_init = per-instance reference centroids, then greedily matches
    components back to instances by nearest initialized mean to avoid the
    EM component-reordering risk.
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        return None

    K = len(instances_pts)
    X = np.concatenate(instances_pts, axis=0)  # (K * n_conf, 3)
    means_init = np.stack([p.mean(axis=0) for p in instances_pts])  # (K, 3)

    gmm = GaussianMixture(
        n_components=K,
        covariance_type="full",
        means_init=means_init,
        reg_covar=_GMM_REG_COVAR,
        random_state=42,
        n_init=1,
        max_iter=300,
    )
    gmm.fit(X)

    # Greedy component -> instance assignment by nearest initialized mean.
    dist = ((means_init[:, None, :] - gmm.means_[None, :, :]) ** 2).sum(-1)  # (K, K)
    used: set[int] = set()
    assign: dict[int, int] = {}
    for i in range(K):
        for j in np.argsort(dist[i]):
            jj = int(j)
            if jj not in used:
                assign[i] = jj
                used.add(jj)
                break
    out: list[np.ndarray] = []
    for i in range(K):
        eig = np.linalg.eigvalsh(gmm.covariances_[assign[i]])[::-1]
        out.append(_normalize(eig))
    return out


def _feature_table(
    mol,
    num_conformers: int,
    use_gmm: bool,
    agg: str,
) -> np.ndarray:
    """Compute the (N_atoms, 18) per-atom pharmacophore feature table."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    n_atoms = mol.GetNumAtoms()
    anum = [a.GetAtomicNum() for a in mol.GetAtoms()]

    # --- 1. Conformer sampling ------------------------------------------------
    params = AllChem.ETKDG()
    params.randomSeed = 0xF00D
    try:
        AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)
    except Exception:
        return np.zeros((n_atoms, NUM_SLOTS), dtype=np.float32)

    n_conf = mol.GetNumConformers()
    if n_conf == 0:
        return np.zeros((n_atoms, NUM_SLOTS), dtype=np.float32)

    # --- 2. Force-field relaxation -------------------------------------------
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1)
    except Exception:
        try:
            AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=1)
        except Exception:
            pass

    # --- 3. Alignment to the reference conformer (heavy-atom based) ----------
    heavy_ids = [i for i, z in enumerate(anum) if z != 1]
    try:
        if heavy_ids:
            AllChem.AlignMolConformers(mol, atomIds=heavy_ids, pruneRmsThresh=0.0)
        else:
            AllChem.AlignMolConformers(mol, pruneRmsThresh=0.0)
    except Exception:
        pass

    # --- 4. Collect per-conformer coordinates ---------------------------------
    confs = list(mol.GetConformers())
    n_conf = len(confs)
    coords = np.stack([np.asarray(c.GetPositions()) for c in confs], axis=0)  # (n_conf, N, 3)

    # --- 5. Pharmacophore detection + covariance eigenvalues ------------------
    feats = _get_factory().GetFeaturesForMol(mol)

    table = np.zeros((n_atoms, NUM_SLOTS), dtype=np.float32)
    counts = np.zeros((n_atoms, NUM_SLOTS), dtype=np.float32)

    for fam_idx, family in enumerate(TARGET_FAMILIES):
        instances = [f for f in feats if FAMILY_ALIASES.get(f.GetFamily()) == family]
        if not instances:
            continue
        block = slice(fam_idx * 3, fam_idx * 3 + 3)

        instance_points = []
        valid_instances = []
        for inst in instances:
            pts = _instance_points(coords, inst.GetAtomIds(), anum, n_atoms)
            if pts is None:
                continue
            instance_points.append(pts)
            valid_instances.append(inst)
        if not valid_instances:
            continue

        if use_gmm:
            eig_list = _gmm_eigvals(instance_points)
            if eig_list is None:
                eig_list = [_empirical_eigvals(p) for p in instance_points]
        else:
            eig_list = [_empirical_eigvals(p) for p in instance_points]

        for inst, eig in zip(valid_instances, eig_list):
            if eig is None:
                continue
            atom_ids = _heavy_atom_ids(inst.GetAtomIds(), anum, n_atoms)
            for aid in atom_ids:
                if agg == "max":
                    table[aid, block] = np.maximum(table[aid, block], eig)
                else:  # mean
                    table[aid, block] += eig
                    counts[aid, block] += 1

    if agg == "mean":
        mask = counts > 0
        table[mask] /= counts[mask]

    return table


def compute_one(
    smiles: str,
    num_conformers: int,
    use_gmm: bool,
    agg: str,
) -> tuple[str, torch.Tensor | None]:
    """Compute per-atom pharmacophore features for a single SMILES."""
    try:
        from rdkit import Chem

        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        if mol is None:
            return smiles, None
        table = _feature_table(mol, num_conformers, use_gmm, agg)
        return smiles, torch.from_numpy(table)
    except Exception:
        return smiles, None


def load_smiles_from_tdc() -> list[str]:
    """Load all unique SMILES from the GDSC2 dataset."""
    from tdc.multi_pred import DrugRes

    data = DrugRes(name="GDSC2")
    smiles_series = data.entity1.copy()
    unique = sorted(set(str(s) for s in smiles_series))
    print(f"Loaded {len(unique)} unique SMILES from GDSC2")
    return unique


def load_smiles_from_file(path: str) -> list[str]:
    """Load SMILES from a text file, one per line."""
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    unique = sorted(set(lines))
    print(f"Loaded {len(unique)} unique SMILES from {path}")
    return unique


def main():
    parser = argparse.ArgumentParser(description="Precompute pharmacophore covariance-eigenvalue features")
    parser.add_argument("--input", type=str, default=None,
                        help="File with one SMILES per line. Default: auto-load from GDSC2")
    parser.add_argument("--output", type=str, default="data/pharmacophores.pt",
                        help="Output .pt file path")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers")
    parser.add_argument("--num-conformers", type=int, default=30,
                        help="Number of 3D conformers to sample per SMILES")
    parser.add_argument("--agg", type=str, choices=["max", "mean"], default="max",
                        help="Aggregation for atoms shared by multiple instances of the same family")
    parser.add_argument("--gmm", action="store_true",
                        help="Use sklearn GaussianMixture instead of per-instance empirical covariance")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="SMILES per worker batch (for progress reporting)")
    args = parser.parse_args()

    # --- Load SMILES -----------------------------------------------------------
    if args.input:
        all_smiles = load_smiles_from_file(args.input)
    else:
        all_smiles = load_smiles_from_tdc()

    # --- Load existing cache ----------------------------------------------------
    cache: dict[str, torch.Tensor] = {}
    output_path = Path(args.output)
    if output_path.exists():
        print(f"Loading existing cache from {output_path} ...")
        cache = torch.load(output_path, map_location="cpu", weights_only=False)
        print(f"  Found {len(cache)} existing entries")

    pending = [s for s in all_smiles if s not in cache]
    if not pending:
        print("All SMILES already cached. Done.")
        return

    print(f"Need to compute pharmacophore features for {len(pending)} SMILES "
          f"({len(cache)} already cached, {len(all_smiles)} total)")
    print(f"Using {args.workers} workers, {args.num_conformers} conformers, "
          f"agg={args.agg}, gmm={args.gmm}")

    completed = 0
    failed = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(compute_one, s, args.num_conformers, args.gmm, args.agg): s
            for s in pending
        }
        for future in as_completed(futures):
            smi, table = future.result()
            if table is not None:
                cache[smi] = table
                completed += 1
            else:
                failed += 1
            if (completed + failed) % 10 == 0 or (completed + failed) == len(pending):
                print(f"  Progress: {completed + failed}/{len(pending)} "
                      f"(ok={completed}, fail={failed})", flush=True)

    torch.save(cache, str(output_path))
    print(f"\nSaved {len(cache)} entries to {output_path}")
    print(f"  Completed: {completed}, Failed: {failed}")

    # --- Sanity summary ---------------------------------------------------------
    n_nonzero = sum(1 for t in cache.values() if t.abs().sum().item() > 0)
    shapes = set(t.shape for t in cache.values())
    print(f"  Entries with non-zero features: {n_nonzero}/{len(cache)}")
    print(f"  Feature shapes seen: {shapes}")


if __name__ == "__main__":
    main()
