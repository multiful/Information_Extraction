#!/bin/bash
# Detached orchestrator: waits for chunked_distant_full.sh to finish all 34 chunks
# and log "starting stage 2", then kills the old-params stage-2 invocation it just
# launched (epochs 3/patience 1) and replaces it with the updated one (epochs
# 8/patience 3). Then waits for that to finish and writes a final status line.
# Run fully detached (nohup + disown) so it isn't tied to any single tool call's
# timeout -- this whole thing is expected to take a few hours.
cd /Users/baiohelseu/Desktop/Project/Information_Extraction
CHUNKS_LOG=Scripts/models/logs/distant_full_pu_chunks.log
FINAL_LOG=Scripts/models/logs/final_full_pu.log
STATUS=Scripts/models/logs/orchestration_status.txt
CKPT=Scripts/models/dk_checkpoints/distant_full_pu.pt

echo "=== orchestrator started, waiting for all 34 chunks, $(date) ===" > "$STATUS"

until grep -q "starting stage 2" "$CHUNKS_LOG"; do sleep 15; done
echo "=== all chunks done, detected stage-2 trigger, $(date) ===" >> "$STATUS"

sleep 3
pkill -f "chunked_distant_full.sh"
pkill -f "train_split train_annotated"
sleep 2
echo "=== old-params stage 2 killed, launching updated (epochs 8, patience 3), $(date) ===" >> "$STATUS"

caffeinate -d -i -m -s python3 Scripts/models/dk_train.py \
  --device mps --train_split train_annotated \
  --init_checkpoint "$CKPT" \
  --epochs 8 --eval_every 3053 --patience 3 --seed 42 \
  --save_name final_full_pu.pt >> "$FINAL_LOG" 2>&1
EXIT_CODE=$?

echo "=== stage 2 finished, exit code $EXIT_CODE, $(date) ===" >> "$STATUS"
tail -8 "$FINAL_LOG" >> "$STATUS"
echo "=== ALL DONE, $(date) ===" >> "$STATUS"
