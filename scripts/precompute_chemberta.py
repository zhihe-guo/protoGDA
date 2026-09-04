#!/usr/bin/env python
"""Precompute ChemBERTa embeddings for all SMILES in GDSC2.

Usage:
    python scripts/precompute_chemberta.py \
        --output data/chemberta_embeddings.pt

If --input is not given, auto-loads SMILES from GDSC2 via TDC.

The output is a torch dict {smiles: tensor(768,)} that can be loaded at
training time with torch.load() and passed as chemberta_cache.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def encode_one(smiles: str, model, tokenizer, device: str) -> tuple[str, torch.Tensor | None]:
    """Encode a single SMILES with ChemBERTa. Returns (smiles, embedding) or (smiles, None)."""
    try:
        inputs = tokenizer(smiles, return_tensors="pt", truncation=True,
                           max_length=512, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()  # [CLS] token, no random pooler
        return smiles, emb
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
    parser = argparse.ArgumentParser(description="Precompute ChemBERTa drug embeddings")
    parser.add_argument("--input", type=str, default=None,
                        help="File with one SMILES per line. Default: auto-load from GDSC2")
    parser.add_argument("--output", type=str, default="data/chemberta_embeddings.pt",
                        help="Output .pt file path")
    parser.add_argument("--model-name", type=str,
                        default="seyonec/ChemBERTa-zinc-base-v1",
                        help="HuggingFace model name")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or cuda:0")
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

    pending = [s for s in all_smiles if s not in cache]
    if not pending:
        print("All SMILES already cached. Done.")
        return

    print(f"Need to encode {len(pending)} SMILES "
          f"({len(cache)} already cached, {len(all_smiles)} total)")

    # --- Load ChemBERTa ---------------------------------------------------------
    print(f"Loading ChemBERTa model: {args.model_name} ...")
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.to(args.device)
    model.eval()
    print(f"  Model loaded on {args.device}")

    # --- Encode ----------------------------------------------------------------
    completed = 0
    failed = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for i, smi in enumerate(pending):
        smi, emb = encode_one(smi, model, tokenizer, args.device)
        if emb is not None:
            cache[smi] = emb
            completed += 1
        else:
            failed += 1
        if (i + 1) % 10 == 0 or (i + 1) == len(pending):
            print(f"  Progress: {i + 1}/{len(pending)} "
                  f"(ok={completed}, fail={failed})", flush=True)

    # --- Save -------------------------------------------------------------------
    torch.save(cache, str(output_path))
    print(f"\nSaved {len(cache)} entries to {output_path}")
    print(f"  Completed: {completed}, Failed: {failed}")


if __name__ == "__main__":
    main()
