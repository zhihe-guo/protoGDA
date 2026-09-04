"""Precompute all external dataset registries offline.

Reads dti_pairs.parquet + cell_gene_expression.npz + conformers/*.sdf
and produces:

    data/external/
        drug_graphs.pt           list[PyG Data] — one per unique SMILES
        drug_smiles.pkl           list[str]      — SMILES strings, same order
        drug_morgan.pt           Tensor (N_drugs, n_bits)
        drug_chemberta.pt        Tensor (N_drugs, 768)
        conformer_cache.pt       {SMILES: Tensor(N_atoms, 3)}
        cell_table.pt            Tensor (N_cells, pca_dim)
        drug_id_to_idx.pkl       {SMILES: int}
        cell_id_to_idx.pkl       {ModelID: int}
        meta.json                cell_dim, drug_count, cell_count, ...

Usage:
    python scripts/precompute_external.py
    python scripts/precompute_external.py --gnn-type gcn    # skip conformers
    python scripts/precompute_external.py --no-chemberta    # skip ChemBERTa
    python scripts/precompute_external.py --morgan-radius 2 --morgan-bits 2048
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.preprocessing import compute_morgan_fingerprint, normalize_features, smiles_to_graph

OUT_DIR = ROOT / "data" / "external"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _smiles_hash(smiles: str) -> str:
    return hashlib.sha256(smiles.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
#  Stage 1: Load pair table
# ---------------------------------------------------------------------------

def load_pair_table(parquet_path: str) -> pd.DataFrame:
    print(f"[1/6] Loading pairs from {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    df["Drug_ID"] = df["Unified_ID"].astype(str).str.strip()
    df["Cell_Line_ID"] = df["ModelID"].astype(str).str.strip()
    df["Y"] = df["IC50_uM"].astype(np.float32)
    df = df[["Drug_ID", "Cell_Line_ID", "Y", "gene_idx", "conformer_sdf"]]
    print(f"  {len(df):,} pairs, {df.Drug_ID.nunique()} drugs, {df.Cell_Line_ID.nunique()} cells")
    return df


# ---------------------------------------------------------------------------
#  Stage 2: Drug graphs (multi-threaded)
# ---------------------------------------------------------------------------

def _graph_worker(smi: str, with_conformers: bool, conformer_cache: dict):
    return smiles_to_graph(smi, with_conformers=with_conformers, conformer_cache=conformer_cache)


def build_drug_graphs(
    drug_ids: list[str],
    num_workers: int = 8,
    with_conformers: bool = False,
    conformer_cache: dict | None = None,
) -> tuple[list[Any], list[str]]:
    print(f"[2/6] Building drug graphs for {len(drug_ids)} SMILES "
          f"(with_conformers={with_conformers}, workers={num_workers}) ...")
    t0 = time.perf_counter()

    smi_graphs: dict[str, Any] = {}
    failed = 0
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {
            ex.submit(_graph_worker, smi, with_conformers, conformer_cache or {}): smi
            for smi in drug_ids
        }
        for f in tqdm(as_completed(futures), total=len(futures), desc="  Drug graphs"):
            smi = futures[f]
            g = f.result()
            if g is None:
                g = smiles_to_graph("C", with_conformers=False)
                failed += 1
            smi_graphs[smi] = g
    if failed:
        print(f"  Warning: {failed}/{len(drug_ids)} invalid SMILES replaced with dummy")

    drug_graphs = [smi_graphs[smi] for smi in drug_ids]
    drug_smiles = list(drug_ids)

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s")
    return drug_graphs, drug_smiles


# ---------------------------------------------------------------------------
#  Stage 3: Morgan fingerprints
# ---------------------------------------------------------------------------

def build_morgan_table(
    drug_smiles: list[str],
    radius: int = 2,
    n_bits: int = 1024,
    num_workers: int = 8,
) -> torch.Tensor:
    print(f"[3/6] Building Morgan FPs for {len(drug_smiles)} SMILES "
          f"(radius={radius}, bits={n_bits}) ...")
    t0 = time.perf_counter()
    fp_map: dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {
            ex.submit(compute_morgan_fingerprint, s, radius, n_bits): s
            for s in drug_smiles
        }
        for f in tqdm(as_completed(futures), total=len(futures), desc="  Morgan FPs"):
            fp_map[futures[f]] = f.result()
    rows = [fp_map[s] for s in drug_smiles]
    table = torch.from_numpy(np.stack(rows, axis=0))
    elapsed = time.perf_counter() - t0
    print(f"  Morgan table shape: {table.shape} ({elapsed:.1f}s)")
    return table


# ---------------------------------------------------------------------------
#  Stage 4: Conformer cache from SDF files
# ---------------------------------------------------------------------------

def build_conformer_cache(
    pairs_df: pd.DataFrame,
) -> dict[str, torch.Tensor]:
    print("[4/6] Building conformer cache from SDF files ...")
    t0 = time.perf_counter()

    smi_to_sdf: dict[str, str] = {}
    for _, row in pairs_df.iterrows():
        smi = str(row["Drug_ID"])
        if smi not in smi_to_sdf:
            sdf_path = row.get("conformer_sdf")
            if isinstance(sdf_path, str) and os.path.isfile(sdf_path):
                smi_to_sdf[smi] = sdf_path

    print(f"  {len(smi_to_sdf)} SMILES with SDF conformers")
    cache: dict[str, torch.Tensor] = {}
    failed = 0

    for smi, sdf_path in tqdm(smi_to_sdf.items(), desc="  Conformers"):
        try:
            mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
            if mol is None:
                failed += 1
                continue
            conf = mol.GetConformer()
            n = mol.GetNumAtoms()
            pos = torch.tensor(
                [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                 for i in range(n)],
                dtype=torch.float32,
            )
            cache[smi] = pos
        except Exception:
            failed += 1
            continue

    if failed:
        print(f"  Warning: {failed} SDF files failed to load")
    elapsed = time.perf_counter() - t0
    print(f"  {len(cache)} conformers in {elapsed:.1f}s")
    return cache


# ---------------------------------------------------------------------------
#  Stage 5: Cell features from npz
# ---------------------------------------------------------------------------

def build_cell_table(
    cell_ids: list[str],
    cell_npz_path: str,
    pca_dim: int = 256,
    normalize: bool = True,
) -> torch.Tensor:
    print(f"[5/6] Building cell table for {len(cell_ids)} cells ...")
    t0 = time.perf_counter()

    data = np.load(cell_npz_path, allow_pickle=True)
    raw_matrix: np.ndarray = data["gene_matrix"]
    model_ids: np.ndarray = data["model_ids"]
    print(f"  Gene matrix: {raw_matrix.shape}")

    mid_to_row = {str(mid).strip(): i for i, mid in enumerate(model_ids)}
    cell_rows = np.zeros((len(cell_ids), raw_matrix.shape[1]), dtype=np.float32)
    for i, cid in enumerate(cell_ids):
        ri = mid_to_row.get(str(cid).strip())
        if ri is not None:
            cell_rows[i] = raw_matrix[ri]

    if normalize:
        cell_rows, _, _ = normalize_features(cell_rows)

    if 0 < pca_dim < cell_rows.shape[1]:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_dim, random_state=42)
        cell_rows = pca.fit_transform(cell_rows).astype(np.float32)
        print(f"  PCA: {raw_matrix.shape[1]} → {pca_dim}, "
              f"explained var: {pca.explained_variance_ratio_.sum():.3f}")

    table = torch.from_numpy(cell_rows)
    elapsed = time.perf_counter() - t0
    print(f"  Cell table shape: {table.shape} ({elapsed:.1f}s)")
    return table


# ---------------------------------------------------------------------------
#  Stage 6: ChemBERTa embeddings
# ---------------------------------------------------------------------------

def build_chemberta_table(
    drug_smiles: list[str],
    batch_size: int = 1024,
) -> torch.Tensor | None:
    print(f"[6/6] Building ChemBERTa embeddings for {len(drug_smiles)} SMILES ...")
    t0 = time.perf_counter()

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {dev}, batch_size={batch_size}")

    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").to(dev).eval()

    embeddings: list[torch.Tensor] = []
    for i in tqdm(range(0, len(drug_smiles), batch_size), desc="  ChemBERTa"):
        batch_smiles = drug_smiles[i:i + batch_size]
        tok = tokenizer(batch_smiles, return_tensors="pt", truncation=True,
                        max_length=512, padding=True)
        tok = {k: v.to(dev) for k, v in tok.items()}
        with torch.no_grad():
            out = model(**tok)
        # [CLS] token = last_hidden[:, 0, :]   (batch_size, 768)
        embeddings.append(out.last_hidden_state[:, 0, :].cpu())

    table = torch.cat(embeddings, dim=0)
    elapsed = time.perf_counter() - t0
    print(f"  ChemBERTa table shape: {table.shape} ({elapsed:.1f}s)")
    return table


# ---------------------------------------------------------------------------
#  Save
# ---------------------------------------------------------------------------

def save_outputs(
    drug_graphs: list,
    drug_smiles: list[str],
    drug_morgan: torch.Tensor | None,
    chemberta: torch.Tensor | None,
    conformer_cache: dict | None,
    cell_table: torch.Tensor,
    drug_id_to_idx: dict[str, int],
    cell_id_to_idx: dict[str, int],
    cell_dim: int,
):
    print("=" * 60)
    print(f"Saving to {OUT_DIR} ...")
    t0 = time.perf_counter()

    torch.save(drug_graphs, OUT_DIR / "drug_graphs.pt")
    with open(OUT_DIR / "drug_smiles.pkl", "wb") as f:
        pickle.dump(drug_smiles, f)

    if drug_morgan is not None:
        torch.save(drug_morgan, OUT_DIR / "drug_morgan.pt")
    if chemberta is not None:
        torch.save(chemberta, OUT_DIR / "drug_chemberta.pt")
    if conformer_cache is not None and len(conformer_cache) > 0:
        torch.save(conformer_cache, OUT_DIR / "conformer_cache.pt")

    torch.save(cell_table, OUT_DIR / "cell_table.pt")
    with open(OUT_DIR / "drug_id_to_idx.pkl", "wb") as f:
        pickle.dump(drug_id_to_idx, f)
    with open(OUT_DIR / "cell_id_to_idx.pkl", "wb") as f:
        pickle.dump(cell_id_to_idx, f)

    meta = {
        "cell_dim": cell_dim,
        "drug_count": len(drug_graphs),
        "cell_count": cell_table.shape[0],
        "has_morgan": drug_morgan is not None,
        "has_chemberta": chemberta is not None,
        "has_conformers": conformer_cache is not None and len(conformer_cache) > 0,
    }
    with open(OUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f)

    elapsed = time.perf_counter() - t0
    total_size = sum(
        (OUT_DIR / fn).stat().st_size
        for fn in os.listdir(OUT_DIR)
        if os.path.isfile(OUT_DIR / fn)
    )
    print(f"  Saved {len(os.listdir(OUT_DIR))} files ({total_size/1e6:.1f} MB) in {elapsed:.1f}s")
    print("Done!")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Precompute external dataset registries")
    parser.add_argument("--pairs", default="data/dti_pairs.parquet")
    parser.add_argument("--cell-npz", default="data/cell_gene_expression.npz")
    parser.add_argument("--gnn-type", default="egnn", choices=["gat", "gcn", "gine", "egconv", "egnn"])
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pca-dim", type=int, default=256)
    parser.add_argument("--morgan-radius", type=int, default=2)
    parser.add_argument("--morgan-bits", type=int, default=1024)
    parser.add_argument("--no-morgan", action="store_true")
    parser.add_argument("--no-chemberta", action="store_true")
    parser.add_argument("--chemberta-batch", type=int, default=1024)
    args = parser.parse_args()

    overall_start = time.perf_counter()
    with_conformers = (args.gnn_type == "egnn")

    # 1. Load pair table (always needed for drug/cell IDs)
    df = load_pair_table(args.pairs)
    drug_ids = sorted(df["Drug_ID"].unique())
    cell_ids = sorted(df["Cell_Line_ID"].unique())

    drug_id_to_idx = {did: i for i, did in enumerate(drug_ids)}
    cell_id_to_idx = {cid: i for i, cid in enumerate(cell_ids)}

    # --- Auto-skip: each stage checks if output file already exists ---
    def _exists(fname: str) -> bool:
        return os.path.isfile(OUT_DIR / fname)

    # 2. Conformer cache
    conformer_cache = None
    if with_conformers:
        if _exists("conformer_cache.pt"):
            print("[2/6] Conformer cache already exists, loading ...")
            conformer_cache = torch.load(OUT_DIR / "conformer_cache.pt", map_location="cpu",
                                         weights_only=False)
            print(f"  Loaded {len(conformer_cache)} entries")
        else:
            conformer_cache = build_conformer_cache(df)

    # 3. Drug graphs (need drug_smiles for Morgan / ChemBERTa)
    gs = None
    smiles = None
    if _exists("drug_graphs.pt"):
        print("[3/6] Drug graphs already exist, loading drug_smiles.pkl only ...")
        with open(OUT_DIR / "drug_smiles.pkl", "rb") as f:
            smiles = pickle.load(f)
        # Load graphs only if we need to re-save everything (for consistency)
        gs = torch.load(OUT_DIR / "drug_graphs.pt", map_location="cpu", weights_only=False)
    else:
        gs, smiles = build_drug_graphs(drug_ids, args.num_workers,
                                       with_conformers=with_conformers,
                                       conformer_cache=conformer_cache)

    # 4. Morgan FPs
    morgan_table: torch.Tensor | None = None
    if not args.no_morgan:
        if _exists("drug_morgan.pt"):
            print("[4/6] Morgan table already exists, skipping ...")
            morgan_table = torch.load(OUT_DIR / "drug_morgan.pt", map_location="cpu",
                                      weights_only=True)
        else:
            morgan_table = build_morgan_table(smiles, args.morgan_radius,
                                              args.morgan_bits, args.num_workers)

    # 5. Cell table
    if _exists("cell_table.pt"):
        print("[5/6] Cell table already exists, loading ...")
        cell_table = torch.load(OUT_DIR / "cell_table.pt", map_location="cpu",
                                weights_only=True)
    else:
        cell_table = build_cell_table(cell_ids, args.cell_npz, pca_dim=args.pca_dim)
    cell_dim = cell_table.shape[1]

    # 6. ChemBERTa
    chemberta_table: torch.Tensor | None = None
    if not args.no_chemberta:
        if _exists("drug_chemberta.pt"):
            print("[6/6] ChemBERTa table already exists, skipping ...")
            chemberta_table = torch.load(OUT_DIR / "drug_chemberta.pt", map_location="cpu",
                                         weights_only=True)
        else:
            chemberta_table = build_chemberta_table(smiles, args.chemberta_batch)

    # Save only newly generated files (respect existing ones)
    final_gs = gs or torch.load(OUT_DIR / "drug_graphs.pt", map_location="cpu", weights_only=False)
    final_morgan = morgan_table
    if final_morgan is None and _exists("drug_morgan.pt"):
        final_morgan = torch.load(OUT_DIR / "drug_morgan.pt", map_location="cpu", weights_only=True)
    final_chemberta = chemberta_table
    if final_chemberta is None and _exists("drug_chemberta.pt"):
        final_chemberta = torch.load(OUT_DIR / "drug_chemberta.pt", map_location="cpu", weights_only=True)
    final_conformer = conformer_cache
    if final_conformer is None and _exists("conformer_cache.pt"):
        final_conformer = torch.load(OUT_DIR / "conformer_cache.pt", map_location="cpu", weights_only=False)

    save_outputs(final_gs, smiles, final_morgan, final_chemberta, final_conformer,
                 cell_table, drug_id_to_idx, cell_id_to_idx, cell_dim)

    overall_elapsed = time.perf_counter() - overall_start
    print(f"Total time: {overall_elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
