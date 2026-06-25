#!/usr/bin/env bash
set -u
PY=/home/eth-admin/miniconda3/envs/retfound/bin/python
REPO=/home/eth-admin/Desktop/isaack/RETFound-main
NEED_MIB=6000
LOG="$REPO/run_p6_pilot.log"
cd "$REPO"
echo "[$(date '+%F %T')] P6 pilot waiter started; need ${NEED_MIB} MiB free" | tee "$LOG"
while true; do
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -v need="$NEED_MIB" '$2+0 >= need {print $1; exit}' | tr -d ' ')
  if [ -n "$GPU" ]; then
    echo "[$(date '+%F %T')] GPU $GPU free — executing pilot notebook" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m jupyter nbconvert --to notebook --execute \
      --inplace --ExecutePreprocessor.kernel_name=retfound --ExecutePreprocessor.timeout=-1 \
      phase6_3class_pilot.ipynb 2>&1 | tee -a "$LOG"
    echo "[$(date '+%F %T')] pilot exited ${PIPESTATUS[0]}" | tee -a "$LOG"
    break
  fi
  sleep 30
done
