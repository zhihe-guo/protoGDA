#!/usr/bin/env python
"""Run head-to-head comparison of CellQuery, GraphDRP, and MGATAF.

Runs each model under interpolation, drug_cold, and cell_cold split modes,
then prints a side-by-side comparison table matching the format of
MGATAF paper Table 3–4 (PCC and RMSE).

Usage:
    # Run everything (slow — ~9 training jobs):
    python scripts/compare_all.py

    # Run only CellQuery (e.g. after it was previously missing):
    python scripts/compare_all.py --models cellquery

    # Collect existing results without retraining:
    python scripts/compare_all.py --collect-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

ALL_MODELS = ["cellquery", "graphdrp", "mgataf", "candela"]
ALL_SPLITS = ["interpolation", "drug_cold", "cell_cold"]

# Checkpoint directory mapping
CKPT_DIR = {
    ("cellquery", "interpolation"): "checkpoints/cellquery_interpolation",
    ("cellquery", "drug_cold"):     "checkpoints/cellquery_drug_cold",
    ("cellquery", "cell_cold"):     "checkpoints/cellquery_cell_cold",
    ("graphdrp", "interpolation"): "checkpoints/graphdrp_interpolation",
    ("graphdrp", "drug_cold"):     "checkpoints/graphdrp_drug_cold",
    ("graphdrp", "cell_cold"):     "checkpoints/graphdrp_cell_cold",
    ("mgataf",   "interpolation"): "checkpoints/mgataf_interpolation",
    ("mgataf",   "drug_cold"):     "checkpoints/mgataf_drug_cold",
    ("mgataf",   "cell_cold"):     "checkpoints/mgataf_cell_cold",
    ("candela",  "interpolation"): "checkpoints/candela_interpolation",
    ("candela",  "drug_cold"):     "checkpoints/candela_drug_cold",
    ("candela",  "cell_cold"):     "checkpoints/candela_cell_cold",
}

# Display names
SPLIT_LABELS = {
    "interpolation": "Mixed",
    "drug_cold":     "DrugCold",
    "cell_cold":     "CellCold",
}

# MGATAF paper reference values (GDSC v6, different preprocessing — reference only).
# IMPORTANT: the paper normalizes IC50 to [0,1] before training, so its RMSE
# is on the normalized scale. Our MGATAF/GraphDRP configs enable
# `data.normalize_label: true` to match this scale; both normalized RMSE
# and the original log10(IC50)-scale RMSE_raw are reported.
MGATAF_PAPER = {
    "Mixed":    {"pearson": 0.9312, "rmse": 0.0225},
    "CellCold": {"pearson": 0.8536, "rmse": 0.0321},
}


def _metric_value(metrics: dict, name: str) -> float | None:
    """Handle the metric naming used by the independent baseline trainers."""
    aliases = {
        "pearson": ("pearson", "pearson_r"),
        "spearman": ("spearman", "spearman_r"),
    }
    for key in aliases.get(name, (name,)):
        if key in metrics:
            return metrics[key]
    return None


def get_best_metrics(ckpt_dir: str) -> dict | None:
    """Read test metrics when available, else the best validation metrics."""
    ckpt_path = Path(ckpt_dir)

    # Prefer summary.json. Test metrics are the comparable result; older
    # summaries without them retain their validation fallback.
    summary_path = ckpt_path / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            s = json.load(f)
        best = s.get("best_val", {})
        metrics = s.get("test") or best
        if not metrics:
            return None
        return {
            "epoch": s.get("best_epoch"),
            "rmse": metrics["rmse"],
            "pearson": _metric_value(metrics, "pearson"),
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "spearman": _metric_value(metrics, "spearman"),
            **{
                key: metrics[key]
                for key in ("rmse_raw", "pearson_raw")
                if key in metrics
            },
        }

    # Fallback to history.json
    hist_path = ckpt_path / "history.json"
    if not hist_path.exists():
        return None

    with open(hist_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    if not history:
        return None

    # Best by validation RMSE
    best = min(history, key=lambda r: r["valid"]["rmse"])
    out = {
        "epoch": best["epoch"],
        "rmse": best["valid"]["rmse"],
        "pearson": best["valid"]["pearson"],
        "mae": best["valid"]["mae"],
        "r2": best["valid"]["r2"],
        "spearman": best["valid"]["spearman"],
    }
    # If the trainer reported original-scale metrics, surface them too
    v = best["valid"]
    if "rmse_raw" in v:
        out["rmse_raw"] = v["rmse_raw"]
        out["pearson_raw"] = v.get("pearson_raw", 0.0)
    return out


def _make_config(model: str, split: str) -> str:
    """Create a disposable YAML config for model+split (returns path)."""
    ckpt_dir = CKPT_DIR[(model, split)]

    if model == "cellquery":
        cfg = OmegaConf.load(str(ROOT / "config" / "default.yaml"))
        cfg.data.split_mode = split
        cfg.training.checkpoint_dir = ckpt_dir
    elif model == "graphdrp":
        cfg = OmegaConf.load(str(ROOT / "baselines" / "graphdrp" / "config_interpolation.yaml"))
        cfg.data.split_mode = split
        cfg.training.checkpoint_dir = ckpt_dir
    elif model == "mgataf":
        cfg = OmegaConf.load(str(ROOT / "baselines" / "mgataf" / f"config_{split}.yaml"))
        cfg.training.checkpoint_dir = ckpt_dir
    elif model == "candela":
        cfg = OmegaConf.load(str(ROOT / "baselines" / "candela" / f"config_{split}.yaml"))
        cfg.training.checkpoint_dir = ckpt_dir
    else:
        raise ValueError(f"Unknown model: {model}")

    # Ensure checkpoint dir is set correctly (belt-and-suspenders)
    cfg.training.checkpoint_dir = ckpt_dir

    # Write to temp file that persists after subprocess
    if model == "cellquery":
        config_dir = ROOT / "config"
    elif model in ["graphdrp", "mgataf", "candela"]:
        config_dir = ROOT / "baselines" / model
    else:
        config_dir = ROOT / "config"

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        dir=str(config_dir),
    )
    OmegaConf.save(cfg, tmp)
    tmp.close()
    return tmp.name


def run_experiment(model: str, split: str, dry_run: bool = False) -> bool:
    """Run one training experiment. Return True if successful."""
    config_path = _make_config(model, split)
    ckpt_dir = CKPT_DIR[(model, split)]

    if model == "cellquery":
        train_script = str(ROOT / "scripts" / "train.py")
    elif model in ["mgataf", "graphdrp", "candela"]:
        train_script = str(ROOT / "baselines" / model / "train.py")
    else:
        raise ValueError(f"Unknown model: {model}")

    cmd = [
        sys.executable,
        train_script,
        "--config", config_path,
    ]

    if dry_run:
        print(f"[DRY-RUN] {' '.join(cmd)}")
        return True

    print(f"\n{'='*70}")
    print(f"  Running: {model} / {split}")
    print(f"  Config:  {config_path}")
    print(f"  Checkpt: {ckpt_dir}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, cwd=str(ROOT))
    # Clean up temp config
    Path(config_path).unlink(missing_ok=True)
    return result.returncode == 0


def collect_results(models: list[str], splits: list[str]) -> dict:
    """Collect best metrics for all model × split combinations."""
    results: dict[str, dict[str, dict]] = {}

    for model in models:
        results[model] = {}
        for split in splits:
            ckpt_dir = CKPT_DIR[(model, split)]
            metrics = get_best_metrics(ckpt_dir)
            if metrics:
                results[model][split] = metrics
                print(f"  ✓ {model:12s} / {split:16s}  "
                      f"RMSE={metrics['rmse']:.4f}  Pearson={metrics['pearson']:.4f}")
            else:
                print(f"  ✗ {model:12s} / {split:16s}  (no results)")

    return results


def print_table(results: dict):
    """Print comparison table in paper format."""
    print("\n\n" + "=" * 110)
    print("                        COMPARISON RESULTS TABLE")
    print("=" * 110)
    print()

    # Detect whether any model reported raw-scale metrics
    has_raw = any(
        "rmse_raw" in results[m][s]
        for m in results for s in results[m]
    )

    # --- PCC table ---
    print("          Pearson Correlation Coefficient (PCC)  [normalized-scale; raw-scale in brackets if available]")
    print("-" * 90)
    header = f"{'Model':<18}"
    for split in ALL_SPLITS:
        label = SPLIT_LABELS[split]
        header += f"  {label:>16}"
    print(header)
    print("-" * 90)

    for model in ALL_MODELS:
        if model not in results:
            continue
        row = f"{model:<18}"
        for split in ALL_SPLITS:
            m = results[model].get(split)
            if m:
                if "pearson_raw" in m:
                    row += f"  {m['pearson']:>7.4f}({m['pearson_raw']:.3f})"
                else:
                    row += f"  {m['pearson']:>16.4f}"
            else:
                row += f"  {'N/A':>16}"
        print(row)

    row = f"{'MGATAF (paper)*':<18}"
    for split in ALL_SPLITS:
        label = SPLIT_LABELS[split]
        ref = MGATAF_PAPER.get(label, {})
        val = ref.get("pearson")
        if val is not None:
            row += f"  {val:>16.4f}"
        else:
            row += f"  {'N/A':>16}"
    print(row)
    print()

    # --- RMSE table ---
    print("          Root Mean Square Error (RMSE)  [normalized-scale; raw-scale in brackets if available]")
    print("-" * 90)
    header = f"{'Model':<18}"
    for split in ALL_SPLITS:
        label = SPLIT_LABELS[split]
        header += f"  {label:>16}"
    print(header)
    print("-" * 90)

    for model in ALL_MODELS:
        if model not in results:
            continue
        row = f"{model:<18}"
        for split in ALL_SPLITS:
            m = results[model].get(split)
            if m:
                if "rmse_raw" in m:
                    row += f"  {m['rmse']:>7.4f}({m['rmse_raw']:.3f})"
                else:
                    row += f"  {m['rmse']:>16.4f}"
            else:
                row += f"  {'N/A':>16}"
        print(row)

    row = f"{'MGATAF (paper)*':<18}"
    for split in ALL_SPLITS:
        label = SPLIT_LABELS[split]
        ref = MGATAF_PAPER.get(label, {})
        val = ref.get("rmse")
        if val is not None:
            row += f"  {val:>16.4f}"
        else:
            row += f"  {'N/A':>16}"
    print(row)

    print()
    print("* MGATAF 原文使用 GDSC v6 + 不同预处理管线（IC50 归一化到 [0,1]），数值仅供参考对比趋势")
    print(f"  原文 DrugCold 场景未测试，标注 N/A")
    print(f"  本项目已开启 normalize_label，括号外数字 = 归一化尺度 (与原文同尺度)，")
    print(f"  括号内数字 (raw) = 原 log10(IC50) 尺度。Pearson 为尺度无关量，二者相同。")
    print("=" * 110)


def main():
    parser = argparse.ArgumentParser(
        description="Run comparison experiments across models and splits."
    )
    parser.add_argument(
        "--models", nargs="+", default=ALL_MODELS,
        help=f"Models to run (default: {' '.join(ALL_MODELS)})",
    )
    parser.add_argument(
        "--splits", nargs="+", default=ALL_SPLITS,
        help=f"Split modes to run (default: {' '.join(ALL_SPLITS)})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--collect-only", action="store_true",
        help="Only collect existing results without training",
    )
    args = parser.parse_args()

    models = args.models
    splits = args.splits

    for m in models:
        if m not in ALL_MODELS:
            print(f"Unknown model: {m}. Choices: {ALL_MODELS}")
            sys.exit(1)
    for s in splits:
        if s not in ALL_SPLITS:
            print(f"Unknown split: {s}. Choices: {ALL_SPLITS}")
            sys.exit(1)

    print(f"Models: {', '.join(models)}")
    print(f"Splits: {', '.join(splits)}")
    print(f"Jobs:   {len(models) * len(splits)} experiment(s)")
    print()

    if not args.collect_only:
        for model in models:
            for split in splits:
                if not run_experiment(model, split, dry_run=args.dry_run):
                    print(f"  ✗ FAILED: {model} / {split}")

    if not args.dry_run:
        print("\n\nCollecting results...")
        results = collect_results(models, splits)
        if results:
            print_table(results)


if __name__ == "__main__":
    main()
