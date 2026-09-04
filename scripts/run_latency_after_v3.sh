#!/bin/bash
# Wait for the v3 drug_cold baseline rerun (CANDELA + MGATAF, 6 folds each) to
# finish, then run the GPU inference-latency benchmark in a dedicated queue so
# timing is not polluted by concurrent training.
#
# Usage: bash scripts/run_latency_after_v3.sh   (launch in background)
set -u

log=/tmp/baseline_drugcold_v3.log
latlog=/tmp/latency_v3.log

echo "[$(date +%F_%T)] latency-queue: waiting for v3 baselines to finish..."
while :; do
    if [ -f "$log" ] && grep -q "ALL DONE" "$log" 2>/dev/null; then
        break
    fi
    sleep 120
done

echo "[$(date +%F_%T)] v3 baselines done. Sleeping 60s for GPU drain, then benchmarking."
sleep 60

: > "$latlog"
cd /trashpaper/ProjectCellQuery || exit 1
python -u scripts/measure_compute_cost.py latency >> "$latlog" 2>&1
echo "[$(date +%F_%T)] latency done (exit $?)" >> "$latlog"
