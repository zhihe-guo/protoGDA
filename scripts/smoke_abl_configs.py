#!/usr/bin/env python
"""CPU smoke: build the model from each ablation config to catch param errors
before launching long GPU runs. Does NOT train."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.dataset import _load_full_tdc_data, _build_registries
from src.models.model import CellDrugModel

CONFIGS = [
    "config/cv_abl_expcov_egnn_gmm_attn.yaml",
    "config/cv_abl_residual_egnn_gmm_attn.yaml",
    "config/cv_abl_gat_residual_edgeattr.yaml",
    "config/cv_abl_tower_drugcold.yaml",
    "config/cv_abl_tower_cellcold.yaml",
    "config/cv_abl_tower_interp.yaml",
]

for cpath in CONFIGS:
    print(f"\n=== {cpath} ===", flush=True)
    cfg = load_config(ROOT / cpath)
    df, smiles_series, cell_series = _load_full_tdc_data(cfg.data.dataset_name)
    registries = _build_registries(df, smiles_series, cell_series, cfg)
    model = CellDrugModel.from_config(
        cell_dim=registries["cell_table"].shape[1],
        cfg=cfg,
        drug_graphs=registries["drug_graphs"],
        cell_features_table=registries["cell_table"],
        drug_morgan_table=registries.get("drug_morgan_table"),
        drug_chemberta_table=registries.get("drug_chemberta_table"),
    )
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  OK  params={n:,}", flush=True)
    del model, registries

print("\nAll configs build successfully.")
