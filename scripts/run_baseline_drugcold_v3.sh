#!/bin/bash
# Rerun drug_cold baselines (CANDELA / MGATAF) under the v3 deterministic
# protocol used for the main model (cv_drug_cold_v3):
#   - 6-fold, fixed splits (seed=42), per-fold seed = 42 + fold_idx
#   - 60 epochs, early stopping disabled (patience 999)
#   - best.pt selected by validation Pearson R
cd /trashpaper/ProjectCellQuery
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

log=/tmp/baseline_drugcold_v3.log
: > "$log"

run() {
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name" >> "$log"
  python -u scripts/cv_runner.py "$@" >> "$log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $name (exit $?)" >> "$log"
}

run CANDELA \
  --model candela --config baselines/candela/config_drug_cold.yaml \
  --k 6 --split_by drug_cold \
  --epochs 60 --patience 999 --checkpoint-metric pearson_r \
  --out checkpoints/cv_candela_drug_cold_v3

run MGATAF \
  --model mgataf --config baselines/mgataf/config_drug_cold.yaml \
  --k 6 --split_by drug_cold \
  --epochs 60 --patience 999 --checkpoint-metric pearson_r \
  --out checkpoints/cv_mgataf_drug_cold_v3

echo "[$(date +%H:%M:%S)] ALL DONE" >> "$log"
