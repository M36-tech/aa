#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p save trainlogs
export PYTHONPATH="$(cd ../.. && pwd)${PYTHONPATH:+:${PYTHONPATH}}"

for seed in ${SEEDS:-3407}; do
  save_path="save/EvoGrad-${DATASET:-office-31}-sd${seed}"
  python -u trainer.py --method EvoGrad --seed "$seed" --gpu "${GPU:-0}" \
    --dataset "${DATASET:-office-31}" --save-path "$save_path" "$@" \
    > "trainlogs/EvoGrad-${DATASET:-office-31}-sd${seed}.log" 2>&1
done
