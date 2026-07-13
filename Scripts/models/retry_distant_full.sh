#!/bin/bash
# Self-healing retry wrapper for the full distant PU-loss pretrain.
#
# Root cause of repeated kills (confirmed via `pmset -g log`): macOS Power
# Nap periodically puts the machine into "Maintenance Sleep" (~every 10-15
# min) even on AC power, which kills background processes. `caffeinate -i`
# (idle-sleep only) does NOT block this; `caffeinate -d -i -m -s` (full
# system-sleep block, requires AC power) does.
#
# If the process still gets killed for some other reason, this automatically
# resumes from the latest saved checkpoint instead of needing manual
# intervention. Each retry re-runs a full epoch over train_distant, but
# seeded from the best checkpoint so far, and checkpoints every 5000 steps
# to bound how much progress a future interruption can lose.
cd /Users/baiohelseu/Desktop/Project/Information_Extraction
CKPT=Scripts/models/dk_checkpoints/distant_full_pu.pt
LOG=Scripts/models/logs/distant_full_pu_retry.log

for i in $(seq 1 20); do
  echo "=== attempt $i, $(date) ===" >> "$LOG"
  if [ -f "$CKPT" ]; then
    INIT_ARG="--init_checkpoint $CKPT"
  else
    INIT_ARG=""
  fi
  caffeinate -d -i -m -s python3 Scripts/models/dk_train.py \
    --device cpu --train_split train_distant \
    --epochs 1 --eval_every 5000 --patience 0 --seed 42 \
    --use_pu_loss --na_weight 0.5 \
    $INIT_ARG \
    --save_name distant_full_pu.pt >> "$LOG" 2>&1
  EXIT_CODE=$?
  if [ $EXIT_CODE -eq 0 ]; then
    echo "=== SUCCESS on attempt $i, $(date) ===" >> "$LOG"
    break
  fi
  echo "=== attempt $i failed (exit $EXIT_CODE), $(date), retrying ===" >> "$LOG"
done

# stage 2: annotated finetune, only runs if stage 1 succeeded
caffeinate -d -i -m -s python3 Scripts/models/dk_train.py \
  --device cpu --train_split train_annotated \
  --init_checkpoint "$CKPT" \
  --epochs 3 --eval_every 3053 --patience 1 --seed 42 \
  --save_name final_full_pu.pt >> Scripts/models/logs/final_full_pu.log 2>&1
