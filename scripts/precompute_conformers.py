#!/usr/bin/env python
"""Precompute 3D conformers for all SMILES in GDSC2 (or a user-provided list).

Usage:
    python scripts/precompute_conformers.py \\
        --output data/conformers.pt \\
        --workers 8

If --input is not given, auto-loads SMILES from GDSC2 via TDC.

The output is a torch dict {smiles: tensor(N,3)} that can be loaded at
training time with torch.load() and passed as conformer_cache.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def generate_one(smiles: str) -> tuple[str, torch.Tensor | None]:
    """Generate a single 3D conformer. Returns (smiles, pos) or (smiles, None)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles, None
        mol = Chem.AddHs(mol)
        status = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if status != 0:
            # fallback: zero coords
            num_atoms = mol.GetNumAtoms()
            return smiles, torch.zeros(num_atoms, 3)
        AllChem.MMFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        pos = torch.tensor(
            [[conf.GetAtomPosition(i).x,
              conf.GetAtomPosition(i).y,
              conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())],
            dtype=torch.float,
        )
        return smiles, pos
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
    parser = argparse.ArgumentParser(description="Precompute molecular 3D conformers")
    parser.add_argument("--input", type=str, default=None,
                        help="File with one SMILES per line. Default: auto-load from GDSC2")
    parser.add_argument("--output", type=str, default="data/conformers.pt",
                        help="Output .pt file path")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers")
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
        cache = torch.load(output_path, map_location="cpu")
        print(f"  Found {len(cache)} existing entries")

    # --- Determine which SMILES need processing ---------------------------------
    pending = [s for s in all_smiles if s not in cache]
    if not pending:
        print("All SMILES already cached. Done.")
        return

    print(f"Need to compute conformers for {len(pending)} SMILES "
          f"({len(cache)} already cached, {len(all_smiles)} total)")
    print(f"Using {args.workers} workers ...")

    # --- Process in parallel ----------------------------------------------------
    completed = 0
    failed = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(generate_one, s): s for s in pending}
        for future in as_completed(futures):
            smi, pos = future.result()
            if pos is not None:
                cache[smi] = pos
                completed += 1
            else:
                failed += 1
            if (completed + failed) % 10 == 0 or (completed + failed) == len(pending):
                print(f"  Progress: {completed + failed}/{len(pending)} "
                      f"(ok={completed}, fail={failed})", flush=True)

    # --- Save -------------------------------------------------------------------
    torch.save(cache, str(output_path))
    print(f"\nSaved {len(cache)} entries to {output_path}")
    print(f"  Completed: {completed}, Failed: {failed}")


if __name__ == "__main__":
    main()
