#!/usr/bin/env python
"""Compute-cost comparison: protoGDA vs CANDELA vs MGATAF.

Usage:
  python scripts/measure_compute_cost.py params          # trainable-parameter counts (CPU)
  python scripts/measure_compute_cost.py latency         # single-forward latency on real data + ckpts (GPU)

All models are built from their official drug_cold configs so the numbers
match the paper's experimental setup.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config


# ---------------------------------------------------------------------------
# Model construction (parameter-count path; no data loading needed)
# ---------------------------------------------------------------------------

def _mgataf_cell_dim() -> int:
    """MGATAF cell dim = # genomic columns actually loaded at runtime
    (gdsc_genomic.csv, no PCA since cell_pca_dim=0)."""
    df = pd.read_csv(ROOT / "data" / "gdsc_genomic.csv", nrows=1)
    return len([c for c in df.columns if c != "Cell_Line_ID"])


def build_protogda(cfg):
    from src.models.model import CellDrugModel
    cell_dim = 256  # data.cell_pca_dim
    n_cells = 8
    return CellDrugModel.from_config(
        cell_dim=cell_dim,
        cfg=cfg,
        drug_graphs=None,
        cell_features_table=torch.zeros(n_cells, cell_dim),
        drug_morgan_table=torch.zeros(n_cells, cfg.model.get("morgan_n_bits", 1024))
        if cfg.model.get("use_morgan", False) else None,
        drug_chemberta_table=torch.zeros(n_cells, 768)
        if cfg.model.get("use_chemberta", False) else None,
    )


def build_candela(cfg):
    from baselines.candela.model import CANDELA
    cell_dim = cfg.data.pca_dim  # 256
    n_cells = 8
    return CANDELA.from_config(
        cell_dim=cell_dim,
        cfg=cfg,
        drug_graphs=None,
        cell_table=torch.zeros(n_cells, cell_dim),
    )


def build_mgataf(cfg):
    from baselines.mgataf.model import MGATAF
    cell_dim = _mgataf_cell_dim()   # 677 genomic columns
    n_cells = 8
    return MGATAF.from_config(
        cell_dim=cell_dim,
        cfg=cfg,
        drug_graphs=None,
        cell_table=torch.zeros(n_cells, cell_dim),
        drug_fp_table=torch.zeros(n_cells, 1024),
    )


def count_params(model, name: str):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    n_buffers = sum(b.numel() for b in model.buffers())
    print(f"{name:<10} trainable={trainable:>11,}  total_params={total:>11,}  "
          f"buffers={n_buffers:>11,}  est_MB(4B)={4 * trainable / 1e6:.2f}")
    return {"model": name, "trainable": int(trainable), "total": int(total)}


def cmd_params():
    builders = [
        ("protoGDA", "config/cellquery_drug_cold.yaml", build_protogda),
        ("CANDELA", "baselines/candela/config_drug_cold.yaml", build_candela),
        ("MGATAF", "baselines/mgataf/config_drug_cold.yaml", build_mgataf),
    ]
    out = []
    for name, cpath, builder in builders:
        cfg = load_config(ROOT / cpath)
        model = builder(cfg)
        out.append(count_params(model, name))
        del model
    print("\nJSON:", out)


# ---------------------------------------------------------------------------
# Latency path (GPU): rebuild fold-0 loaders at a shared batch size, load the
# v3 fold-1 trained checkpoint, and time repeated forwards. The data/model
# construction replicates cv_runner.py (which produced the v3 results) so the
# inputs match real training/evaluation.
# ---------------------------------------------------------------------------

def _load_full_df(model: str):
    """Replicate cv_runner's per-model full-GDSC2 loading + smiles."""
    if model == "candela":
        raw = pd.read_pickle(str(ROOT / "data" / "gdsc2.pkl"))
        raw = raw.rename(columns={"ID1": "Drug_Name", "ID2": "Cell_Name", "Y": "Y"})
        raw["Drug_ID"] = raw["Drug_Name"]
        raw["Cell_Line_ID"] = raw["Cell_Name"]
        df = raw
        drug_smiles = {}
        for did, smi in zip(raw["Drug_ID"].values, raw["X1"].values):
            drug_smiles.setdefault(str(did), str(smi))
        return df, drug_smiles
    from tdc.multi_pred import DrugRes
    dobj = DrugRes(name="GDSC2")
    df = pd.DataFrame({
        "Drug_ID": dobj.entity1_idx.values,
        "Cell_Line_ID": dobj.entity2_idx.values,
        "Y": dobj.y.values,
    })
    drug_smiles = {}
    for did, smi in zip(dobj.entity1_idx.values, dobj.entity1.values):
        drug_smiles.setdefault(str(did), str(smi))
    return df, drug_smiles


def _cellquery_fold0(cfg, df, train_df, valid_df):
    """Fold-0 loaders + model for protoGDA (mirrors cv_runner.run_cellquery)."""
    from src.data.dataset import (_load_full_tdc_data, _build_registries,
                                  DrugCellDataset, collate_drug_cell)
    from src.models.model import CellDrugModel
    from torch.utils.data import DataLoader

    full_df, smiles_series, cell_series = _load_full_tdc_data(cfg.data.dataset_name)
    registries = _build_registries(full_df, smiles_series, cell_series, cfg,
                                   train_cell_ids=set(train_df["Cell_Line_ID"].unique()))
    drug_id_to_idx = registries["drug_id_to_idx"]
    cell_id_to_idx = registries["cell_id_to_idx"]
    cell_dim = registries["cell_table"].shape[1]

    def build_loader(sub_df, shuffle):
        dr = np.array([drug_id_to_idx[str(sub_df.iloc[i]["Drug_ID"])] for i in range(len(sub_df))], dtype=np.int64)
        cl = np.array([cell_id_to_idx[str(sub_df.iloc[i]["Cell_Line_ID"])] for i in range(len(sub_df))], dtype=np.int64)
        lab = sub_df["Y"].values.astype(np.float32)
        ds = DrugCellDataset(dr, cl, lab)
        nw = cfg.data.num_workers
        return DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=shuffle,
                          num_workers=nw, pin_memory=True, collate_fn=collate_drug_cell,
                          persistent_workers=nw > 0)

    tl = build_loader(train_df, True)
    vl = build_loader(valid_df, False)
    model = CellDrugModel.from_config(
        cell_dim=cell_dim, cfg=cfg,
        drug_graphs=registries["drug_graphs"],
        cell_features_table=registries["cell_table"],
        drug_morgan_table=registries.get("drug_morgan_table"),
        drug_chemberta_table=registries.get("drug_chemberta_table"),
    )
    return tl, vl, model


def _baseline_fold0(model_name: str, cfg, df, drug_smiles, train_df, valid_df):
    """Fold-0 loaders + model for CANDELA/MGATAF (mirrors cv_runner)."""
    sys.path.insert(0, str(ROOT / "baselines"))
    if model_name == "candela":
        sys.path.insert(0, str(ROOT / "baselines" / "candela"))
        import candela.data as data_mod
        import candela.model as model_mod
    else:
        sys.path.insert(0, str(ROOT / "baselines" / "mgataf"))
        import mgataf.data as data_mod
        import mgataf.model as model_mod

    tr, va, te, registries, meta = data_mod.get_candela_dataloaders(
        cfg, train_df=train_df, valid_df=valid_df, test_df=valid_df
    ) if model_name == "candela" else data_mod.get_mgataf_dataloaders(
        cfg, train_df=train_df, valid_df=valid_df, test_df=valid_df
    )
    cell_dim = meta["cell_dim"]
    if model_name == "candela":
        model = model_mod.CANDELA.from_config(
            cell_dim=cell_dim, cfg=cfg,
            drug_graphs=registries["drug_graphs"],
            cell_table=registries["cell_table"],
        )
    else:
        model = model_mod.MGATAF.from_config(
            cell_dim=cell_dim, cfg=cfg,
            drug_graphs=registries["drug_graphs"],
            cell_table=registries["cell_table"],
            drug_fp_table=registries["drug_fp_table"],
        )
    return tr, va, model


def _config_and_v3_ckpt(model: str):
    paths = {
        "cellquery": ("config/cellquery_drug_cold.yaml",
                      "checkpoints/cv_drug_cold_v3/fold_1/best.pt"),
        "candela": ("baselines/candela/config_drug_cold.yaml",
                    "checkpoints/cv_candela_drug_cold_v3/fold_1/best.pt"),
        "mgataf": ("baselines/mgataf/config_drug_cold.yaml",
                   "checkpoints/cv_mgataf_drug_cold_v3/fold_1/best.pt"),
    }
    cpath, ck = paths[model]
    ckpt = ROOT / ck
    if not ckpt.exists():
        print(f"  [warn] missing v3 checkpoint {ckpt}; using random init weights")
        ckpt = None
    return ROOT / cpath, ckpt


def cmd_latency():
    """Time one forward pass per model at a shared batch size."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING: CUDA unavailable — latency will be CPU numbers.")
    else:
        try:
            util = torch.cuda.utilization()
        except Exception:
            util = -1.0  # pynvml not installed
        if util > 25:
            print("WARNING: GPU utilization high — numbers may be polluted by a "
                  "concurrent job; prefer running when GPU is idle.")
    batch_size = int(os.environ.get("LATENCY_BATCH", "256"))
    n_repeat = int(os.environ.get("LATENCY_REPEAT", "40"))
    print(f"Latency settings: batch={batch_size}, repeats={n_repeat}")

    sys.path.insert(0, str(ROOT / "scripts"))
    from cv_runner import make_fold_split

    results = []
    for model in ("cellquery", "candela", "mgataf"):
        print(f"\n=== {model} ===", flush=True)
        cpath, ckpt = _config_and_v3_ckpt(model)
        cfg = load_config(cpath)
        cfg.data.batch_size = batch_size          # shared batch for fair timing
        df, drug_smiles = _load_full_df(model)
        # v3 used cv_runner splits: fold_1 = fold_idx 0 under seed 42.
        splits = make_fold_split(df, 6, "drug_cold", 42, 0)
        train_df, valid_df = splits["train"], splits["valid"]
        if model == "cellquery":
            tl, vl, model_obj = _cellquery_fold0(cfg, df, train_df, valid_df)
        else:
            tl, vl, model_obj = _baseline_fold0(
                model, cfg, df, drug_smiles, train_df, valid_df)
        n_params = sum(p.numel() for p in model_obj.parameters() if p.requires_grad)

        if ckpt is not None:
            sd = torch.load(ckpt, map_location=device, weights_only=False)
            model_obj.load_state_dict(sd["model_state_dict"])
        model_obj = model_obj.to(device).eval()

        # Grab one fixed batch to repeat (deterministic timing inputs).
        loader_iter = iter(vl)
        batch = next(loader_iter)
        batch = {k: v.to(device) for k, v in batch.items()
                 if isinstance(v, torch.Tensor)}

        # Warm-up (CUDA kernels / cudnn autotune).
        with torch.inference_mode():
            for _ in range(5):
                model_obj(batch)

            # Timed repeats.
            times_ms = []
            for _ in range(n_repeat):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    model_obj(batch)
                torch.cuda.synchronize()
                times_ms.append((time.perf_counter() - t0) * 1e3)

        mean_ms = float(np.mean(times_ms))
        std_ms = float(np.std(times_ms))
        per_sample = mean_ms / batch_size
        print(f"  params={n_params:,}  {mean_ms:.3f}±{std_ms:.3f} ms/batch  "
              f"({per_sample * 1e3:.3f} µs/sample)")
        results.append({
            "model": model, "params": int(n_params),
            "batch_size": batch_size, "n_repeat": n_repeat,
            "mean_ms_per_batch": mean_ms, "std_ms_per_batch": std_ms,
            "us_per_sample": per_sample * 1e3,
        })
        del model_obj, tl, vl, df, batch
        torch.cuda.empty_cache()

    print("\nJSON:", results)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "params"
    if which == "params":
        cmd_params()
    elif which == "latency":
        cmd_latency()
    else:
        raise SystemExit(f"unknown mode: {which}")
