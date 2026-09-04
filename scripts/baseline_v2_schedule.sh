#!/bin/bash
# Master schedule: wait for our drug_cold v2 to finish, then rerun all baselines
# under the SAME corrected protocol (60 epochs, no early stopping, seed=42, 6-fold).
cd /trashpaper/ProjectCellQuery

echo "[scheduler] waiting for cv_drug_v2.log to show CV pearson..."
while [ ! -f /tmp/cv_drug_v2.log ] || ! grep -q "CV pearson" /tmp/cv_drug_v2.log; do
  sleep 60
done
echo "[scheduler] drug_cold v2 done -> starting baseline reruns"

run() {
  echo "[scheduler] START: $1" >> /tmp/baseline_v2_schedule.log
  python -u "$@" > /tmp/$(basename $2).log 2>&1
  echo "[scheduler] DONE: $1 (exit $?)" >> /tmp/baseline_v2_schedule.log
}

run scripts/cv_runner.py --model candela --split_by scaffold_cold --patience 999 --out checkpoints/cv_candela_scaffold_cold_v2
run scripts/cv_runner.py --model mgataf   --split_by scaffold_cold --patience 999 --out checkpoints/cv_mgataf_scaffold_cold_v2
run scripts/cv_runner.py --model candela --split_by drug_cold     --patience 999 --out checkpoints/cv_candela_drug_cold_v2
run scripts/cv_runner.py --model mgataf   --split_by drug_cold     --patience 999 --out checkpoints/cv_mgataf_drug_cold_v2

echo "[scheduler] ALL BASELINE RERUNS COMPLETE" >> /tmp/baseline_v2_schedule.log
