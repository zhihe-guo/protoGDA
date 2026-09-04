"""
Feature extraction pipeline: DTI data -> cell-line mapping -> transcriptomics -> 3D conformers.

Stages:
  1. Load Final_DTI_Statistics_Strict1.csv, deduplicate by (SMILES, Cell_Line_Name)
  2. Map Cell_Line_Name -> RRID -> ModelID via Model.csv
  3. Extract gene-expression features from OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv
  4. Multi-process generation of 3D SDF conformers for every unique SMILES
  5. Write Parquet output

Output files:
    data/dti_pairs.parquet        Per (drug, cell) pair: SMILES, IC50_uM, ModelID, conformer_sdf
    data/cell_gene_expression.npz Gene expression matrix (N_cells x N_genes) + ModelID index

Usage:
    python scripts/extract_features.py
    python scripts/extract_features.py --max-workers 8
    python scripts/extract_features.py --sample                 # target 20:1 drug:cell ratio, 100k pairs
    python scripts/extract_features.py --sample --target-ratio 10 --total-limit 50000
    python scripts/extract_features.py --skip-conformers
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths (relative to the repository root)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
DATA_DIR = ROOT / "data"

DTI_CSV = DATASET_DIR / "Final_DTI_Statistics_Strict1.csv"
MODEL_CSV = DATASET_DIR / "Model.csv"
OMICS_CSV = DATASET_DIR / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"

CONFORMER_DIR = DATA_DIR / "conformers"
OUTPUT_PAIRS = DATA_DIR / "dti_pairs.parquet"
OUTPUT_CELL_FEATURES = DATA_DIR / "cell_gene_expression.npz"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smiles_hash(smiles: str) -> str:
    """Return a deterministic hex digest for a SMILES string."""
    return hashlib.sha256(smiles.encode("utf-8")).hexdigest()


def _timed_stage(name: str):
    """Decorator that prints stage timing."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            print("=" * 60)
            print(f"[Stage] {name}")
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            print(f"  Done in {elapsed:.1f}s")
            return result
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Stage 1: Load & deduplicate DTI data
# ---------------------------------------------------------------------------

@_timed_stage("1/5: Load & deduplicate DTI data")
def load_dedupe_dti(dti_path: Path) -> pd.DataFrame:
    """Read the DTI CSV, keep needed columns, and deduplicate."""
    usecols = ["Cell_Line_Name", "Unified_ID", "IC50_uM"]
    dti = pd.read_csv(dti_path, usecols=usecols, low_memory=False)
    print(f"  Raw rows: {len(dti):,}")

    before = len(dti)
    dti = dti.dropna(subset=["Unified_ID", "Cell_Line_Name", "IC50_uM"])
    if len(dti) < before:
        print(f"  Dropped {before - len(dti):,} rows with missing values")

    dti = dti.drop_duplicates(subset=["Unified_ID", "Cell_Line_Name"], keep="first")
    dti = dti.reset_index(drop=True)
    print(f"  Unique (SMILES, Cell_Line_Name) pairs: {len(dti):,}")
    print(f"  Unique SMILES: {dti['Unified_ID'].nunique():,}")
    print(f"  Unique Cell_Line_Names: {dti['Cell_Line_Name'].nunique():,}")
    return dti


# ---------------------------------------------------------------------------
# Stage 2: Cell_Line_Name -> RRID -> ModelID
# ---------------------------------------------------------------------------

@_timed_stage("2/5: Map Cell_Line_Name -> ModelID via Model.csv")
def map_cellname_to_modelid(dti: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    """Map DTI Cell_Line_Name through Model.csv RRID to obtain ModelID."""
    model_df = pd.read_csv(model_path, usecols=["ModelID", "RRID"], low_memory=False)
    model_df = model_df.dropna(subset=["RRID"])
    model_df["RRID"] = model_df["RRID"].str.strip()

    rrid_to_modelid = dict(zip(model_df["RRID"], model_df["ModelID"]))
    print(f"  Unique RRID entries in Model.csv: {len(rrid_to_modelid):,}")

    before = len(dti)
    dti["ModelID"] = dti["Cell_Line_Name"].str.strip().map(rrid_to_modelid)

    unmapped = dti["ModelID"].isna().sum()
    if unmapped:
        unmapped_names = dti.loc[dti["ModelID"].isna(), "Cell_Line_Name"].unique()
        print(f"  WARNING: {unmapped:,} rows ({len(unmapped_names)} unique cell lines) "
              f"could NOT be mapped (dropping)")
        dti = dti.dropna(subset=["ModelID"]).reset_index(drop=True)

    print(f"  Rows after mapping: {len(dti):,} (dropped {before - len(dti):,})")
    return dti


# ---------------------------------------------------------------------------
# Stage 3: Extract transcriptomic features
# ---------------------------------------------------------------------------

@_timed_stage("3/5: Extract transcriptomic features")
def extract_transcriptomics(
    dti: pd.DataFrame, omics_path: Path
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    """Join gene-expression features from the omics matrix.

    Returns:
        dti            DataFrame with an added `gene_idx` column
        gene_matrix    float32 ndarray of shape (N_unique_cells, N_genes)
        gene_cols      list of gene column names
        cell_modelids  list of ModelIDs aligned with gene_matrix rows
    """
    print("  Reading omics matrix (this may take a moment) ...")
    omics = pd.read_csv(omics_path, low_memory=False)

    # Identify gene columns (everything after the metadata columns)
    META_NAMES = {"SequencingID", "ModelConditionID", "ModelID",
                  "IsDefaultEntryForMC", "IsDefaultEntryForModel", "Unnamed: 0"}
    gene_cols = [c for c in omics.columns if c not in META_NAMES]
    print(f"  Omics matrix: {len(omics):,} rows x {len(gene_cols):,} genes")

    # Some ModelIDs have multiple entries; keep only the default entry
    before_dedup = len(omics)
    if "IsDefaultEntryForModel" in omics.columns:
        omics = omics.sort_values("IsDefaultEntryForModel", ascending=False)
        omics = omics.drop_duplicates(subset=["ModelID"], keep="first")
    else:
        omics = omics.drop_duplicates(subset=["ModelID"], keep="first")
    if len(omics) < before_dedup:
        print(f"  Dropped {before_dedup - len(omics):,} duplicate ModelID entries")

    # Build gene matrix and ModelID index using vectorised extraction
    omics_modelids = omics["ModelID"].astype(str).str.strip().values
    gene_matrix = omics[gene_cols].values.astype(np.float32)

    # Create a fast lookup: ModelID -> row index in gene_matrix
    mid_to_idx = {mid: i for i, mid in enumerate(omics_modelids)}
    print(f"  Unique ModelIDs with omics data: {len(mid_to_idx):,}")

    # Map each DTI row to its gene expression row index
    before = len(dti)
    gene_indices: list[int] = []
    keep_mask: list[bool] = []

    for mid in dti["ModelID"].values:
        idx = mid_to_idx.get(str(mid).strip())
        if idx is None:
            keep_mask.append(False)
        else:
            gene_indices.append(idx)
            keep_mask.append(True)

    dropped = before - sum(keep_mask)
    if dropped:
        print(f"  WARNING: {dropped:,} rows have no omics data (dropping)")
        dti = dti.loc[keep_mask].reset_index(drop=True)

    dti["gene_idx"] = gene_indices
    print(f"  Rows after omics join: {len(dti):,}")

    return dti, gene_matrix, gene_cols, omics_modelids.tolist()


# ---------------------------------------------------------------------------
# Stage 3.5: Stratified sampling (per-cell cap + total limit)
# ---------------------------------------------------------------------------

def sample_cell_drug_pairs(
    dti: pd.DataFrame,
    target_ratio: int = 20,
    total_limit: int = 100_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample DTI pairs so that unique_drugs : unique_cells ≈ target_ratio.

    Iterates over randomly-shuffled cells. For each cell, picks drugs
    **preferring novel drugs** (not yet used by any earlier cell) to drive
    the ratio toward target_ratio.  The per-cell drug count is
    auto-tuned.  Stops when total pair count reaches total_limit.
    """
    rng = np.random.default_rng(seed)

    # Pre-group to avoid repeated boolean indexing
    groups = {cell: grp for cell, grp in dti.groupby("ModelID", sort=False)}
    cell_ids = list(groups)
    rng.shuffle(cell_ids)

    used_drugs: set[str] = set()
    used_cells: set[str] = set()
    sampled_frames: list[pd.DataFrame] = []

    for cell in cell_ids:
        grp = groups[cell]
        drugs = grp["Unified_ID"].unique()

        # Split into novel vs already-seen drugs
        novel = [d for d in drugs if d not in used_drugs]
        seen  = [d for d in drugs if d in used_drugs]
        rng.shuffle(novel)
        rng.shuffle(seen)

        # How many drugs to pick for this cell?
        # We want unique_drugs ≈ target_ratio × len(used_cells).
        current_drugs = len(used_drugs)
        target_drugs = target_ratio * (len(used_cells) + 1)
        nov_missing = max(0, target_drugs - current_drugs)

        # Pick enough novel drugs to stay on target; no hard cap
        max_pick = min(len(drugs), nov_missing + target_ratio)
        max_pick = max(1, max_pick)

        picked_drugs = novel[:max_pick]
        if len(picked_drugs) < max_pick:
            # Fill remainder with seen drugs (each drug appears at most once per cell)
            need = max_pick - len(picked_drugs)
            for d in seen:
                if d not in picked_drugs:
                    picked_drugs.append(d)
                if len(picked_drugs) >= max_pick:
                    break

        # Collect one row per (cell, drug) pair
        rows = grp[grp["Unified_ID"].isin(set(picked_drugs))].iloc[:len(picked_drugs)]
        sampled_frames.append(rows)
        used_drugs.update(picked_drugs)
        used_cells.add(cell)

        if sum(len(f) for f in sampled_frames) >= total_limit:
            break

    sampled = pd.concat(sampled_frames, ignore_index=True)
    # Trim to exact limit if overshot slightly
    if len(sampled) > total_limit:
        sampled = sampled.iloc[:total_limit]

    ratio_actual = len(used_drugs) / max(len(used_cells), 1)
    print(f"  Sampled: {len(sampled):,} pairs  |  "
          f"{len(used_cells):,} unique cells  |  "
          f"{len(used_drugs):,} unique drugs  |  "
          f"ratio = {ratio_actual:.1f}:1")
    return sampled


# ---------------------------------------------------------------------------
# Stage 4: Multi-process 3D conformer generation
# ---------------------------------------------------------------------------

def _conformer_worker(smiles, out_dir):
    """Generate one 3D SDF conformer in this thread.

    RDKit conformer generation releases the GIL during C++ computation
    (ETKDG embedding + MMFF optimisation), so ThreadPoolExecutor achieves
    real parallelism without the subprocess-spawn DLL conflicts that hit
    ProcessPoolExecutor on Windows.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDistGeom

    h = _smiles_hash(smiles)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return h, None, "Invalid SMILES"
        mol = Chem.AddHs(mol)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = 42
        if rdDistGeom.EmbedMolecule(mol, params) != 0:
            return h, None, "Embedding failed"
        AllChem.MMFFOptimizeMolecule(mol)
        sdf_path = os.path.join(out_dir, f"{h}.sdf")
        writer = Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()
        return h, sdf_path, None
    except Exception:
        return h, None, "Unexpected error"


@_timed_stage("4/5: Generate 3D conformers (multi-threaded)")
def generate_conformers(
    smiles_list: list[str],
    output_dir: Path,
    max_workers: int | None = None,
) -> dict[str, str | None]:
    """Generate 3D SDF conformers using ThreadPoolExecutor.

    RDKit releases the GIL during conformer generation (ETKDG embedding +
    MMFF optimisation runs in C++), so multi-threading provides real
    parallelism. Threads avoid the subprocess-spawn DLL conflicts
    (``ImportError: cannot load module more than once per process``)
    that occur with ProcessPoolExecutor on Windows.

    Returns dict: {smiles_hash: sdf_path | None}.
    """
    print(f"  Unique SMILES to process: {len(smiles_list):,}")
    print(f"  Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    CONFORMER_TIMEOUT = 60  # seconds per molecule

    results: dict[str, str | None] = {}
    succeeded = 0
    failed = 0
    invalid_smiles = 0
    embed_failed = 0
    timed_out = 0

    # Resume: skip SMILES whose SDF already exists
    out_dir_str = str(output_dir)
    pending: list[str] = []
    already_done = 0
    for smi in smiles_list:
        h = _smiles_hash(smi)
        sdf_path = os.path.join(out_dir_str, f"{h}.sdf")
        if os.path.isfile(sdf_path):
            results[h] = sdf_path
            succeeded += 1
            already_done += 1
        else:
            pending.append(smi)

    if already_done:
        print(f"  Already generated (skipped): {already_done:,}")
    print(f"  Remaining to process: {len(pending):,}")

    if not pending:
        print(f"  Success: {succeeded:,}  |  Failed: {failed:,}")
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_conformer_worker, smi, out_dir_str): smi
                   for smi in pending}

        with tqdm(total=len(futures), desc="  Generating conformers",
                  unit="mol") as pbar:
            for future in as_completed(futures):
                try:
                    h, sdf_path, error = future.result(timeout=CONFORMER_TIMEOUT)
                    results[h] = sdf_path
                    if sdf_path is not None:
                        succeeded += 1
                    else:
                        failed += 1
                        if error == "Invalid SMILES":
                            invalid_smiles += 1
                        elif error == "Embedding failed":
                            embed_failed += 1
                except TimeoutError:
                    smi = futures[future]
                    h = _smiles_hash(smi)
                    results[h] = None
                    timed_out += 1
                pbar.update(1)

    print(f"  Success: {succeeded:,}  |  Failed: {failed:,}  "
          f"(invalid SMILES: {invalid_smiles},  embedding failed: {embed_failed})")
    if timed_out:
        print(f"  Timed out (> {CONFORMER_TIMEOUT}s): {timed_out:,}")
    return results


# ---------------------------------------------------------------------------
# Stage 5: Write output files
# ---------------------------------------------------------------------------

@_timed_stage("5/5: Write output files")
def write_outputs(
    dti: pd.DataFrame,
    gene_matrix: np.ndarray,
    gene_cols: list[str],
    cell_modelids: list[str],
    conformer_results: dict[str, str | None],
    output_pairs: Path,
    output_cell_features: Path,
):
    """Write the per-pair Parquet and the cell-level gene expression .npz."""
    # ---- 5a: Build conformer path column and write pairs Parquet ----
    conformer_paths = [conformer_results.get(_smiles_hash(smi))
                       for smi in dti["Unified_ID"]]

    pairs_df = dti[["Unified_ID", "IC50_uM", "ModelID", "gene_idx"]].copy()
    pairs_df["conformer_sdf"] = conformer_paths
    pairs_df["IC50_uM"] = pairs_df["IC50_uM"].astype(np.float32)

    has_conf = pairs_df["conformer_sdf"].notna().sum()
    missing_conf = len(pairs_df) - has_conf
    print(f"  Rows with valid conformer: {has_conf:,}")
    if missing_conf:
        print(f"  WARNING: {missing_conf:,} rows missing conformer (dropping)")
        pairs_df = pairs_df.dropna(subset=["conformer_sdf"]).reset_index(drop=True)

    pairs_df.to_parquet(output_pairs, index=False, engine="pyarrow",
                        compression="snappy")
    size_mb = output_pairs.stat().st_size / 1e6
    print(f"  Written: {output_pairs} ({size_mb:.1f} MB, {len(pairs_df):,} rows)")

    # ---- 5b: Write gene expression matrix with ModelID index ----
    np.savez_compressed(
        output_cell_features,
        gene_matrix=gene_matrix,
        gene_columns=np.array(gene_cols, dtype=object),
        model_ids=np.array(cell_modelids, dtype=object),
    )
    size_mb = output_cell_features.stat().st_size / 1e6
    print(f"  Written: {output_cell_features} ({size_mb:.1f} MB, "
          f"{gene_matrix.shape[0]:,} cells x {gene_matrix.shape[1]:,} genes)")

    # ---- 5c: Summary ----
    print(f"  Final pair count: {len(pairs_df):,}")
    print(f"  Unique SMILES in output: {pairs_df['Unified_ID'].nunique():,}")
    print(f"  Unique cells in output: {pairs_df['gene_idx'].nunique():,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract DTI features: SMILES, IC50, cell gene expression, 3D conformers"
    )
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Number of threads for conformer generation "
                             "(default: min(32, CPU count + 4))")
    parser.add_argument("--skip-conformers", action="store_true",
                        help="Skip conformer generation (use if already generated)")
    parser.add_argument("--conformer-dir", type=Path, default=CONFORMER_DIR,
                        help="Directory for SDF conformer files")
    parser.add_argument("--sample", action="store_true",
                        help="Stratified sampling with target drug:cell ratio")
    parser.add_argument("--target-ratio", type=int, default=20,
                        help="Target unique_drugs : unique_cells ratio (default: 20)")
    parser.add_argument("--total-limit", type=int, default=100_000,
                        help="Total pair limit when --sample is set (default: 100000)")
    args = parser.parse_args()

    # 1. Load & deduplicate
    dti = load_dedupe_dti(DTI_CSV)

    # 2. Map Cell_Line_Name -> ModelID
    dti = map_cellname_to_modelid(dti, MODEL_CSV)

    # 3. Extract transcriptomic features
    dti, gene_matrix, gene_cols, cell_modelids = extract_transcriptomics(dti, OMICS_CSV)

    # 3.5 Optional stratified sampling
    if args.sample:
        before = len(dti)
        dti = sample_cell_drug_pairs(
            dti,
            target_ratio=args.target_ratio,
            total_limit=args.total_limit,
        )
        n_cells = dti["ModelID"].nunique()
        n_smiles = dti["Unified_ID"].nunique()
        print("=" * 60)
        print(f"[Sampling] Kept {len(dti):,} pairs (from {before:,}), "
              f"{n_cells:,} cells, {n_smiles:,} SMILES")

    # 4. Generate 3D conformers
    if args.skip_conformers:
        print("=" * 60)
        print("[Stage] 4/5: SKIPPED conformer generation (--skip-conformers)")
        # Build conformer_results from existing files
        conformer_results: dict[str, str | None] = {}
        unique_smiles = dti["Unified_ID"].unique().tolist()
        for smi in unique_smiles:
            h = _smiles_hash(smi)
            sdf_path = args.conformer_dir / f"{h}.sdf"
            if sdf_path.exists():
                conformer_results[h] = str(sdf_path)
            else:
                conformer_results[h] = None
        existing = sum(1 for v in conformer_results.values() if v is not None)
        print(f"  Found {existing:,} existing conformer files")
    else:
        unique_smiles = sorted(dti["Unified_ID"].unique().tolist())
        conformer_results = generate_conformers(unique_smiles, args.conformer_dir,
                                                max_workers=args.max_workers)

    # 5. Write outputs
    write_outputs(dti, gene_matrix, gene_cols, cell_modelids,
                  conformer_results, OUTPUT_PAIRS, OUTPUT_CELL_FEATURES)

    print("=" * 60)
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
