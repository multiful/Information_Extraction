#!/bin/bash
# Chunked version of the full distant PU-loss pretrain: instead of one long
# ~101,873-doc run (which kept getting killed unpredictably, sometimes within
# minutes, for reasons beyond confirmed sleep/memory causes), this runs many
# SMALL chunks (a few thousand random docs each, different sample each time),
# each resuming from the previous chunk's checkpoint. If a chunk dies, only
# that chunk's progress (a few minutes) is lost, not hours.
#
# Chunks 1-14 ran on --device cpu before the MPS OOM was root-caused and fixed
# (see dk_train.py module docstring: allocator fragmentation + eager-attention
# fallback + a driver-level graph-cache leak from variable doc lengths, all
# fixed). Resuming from chunk 15 on --device mps, ~1.7x faster end-to-end in a
# same-machine timing comparison (chunk 12/13 CPU: 3000 docs in ~13m44s =
# ~3.6 docs/s; MPS smoke test: 700 docs + 3 evals in ~2m11s = ~6.0 docs/s).
# Stage 2 (annotated finetune) switched to mps too for the same reason.
cd /Users/baiohelseu/Desktop/Project/Information_Extraction
CKPT=Scripts/models/dk_checkpoints/distant_full_pu.pt
LOG=Scripts/models/logs/distant_full_pu_chunks.log
CHUNK_SIZE=3000
NUM_CHUNKS=34   # 34 * 3000 = 102,000 doc-exposures, ~1 epoch-equivalent over 101,873 distant docs

for i in $(seq 15 $NUM_CHUNKS); do
  echo "=== chunk $i/$NUM_CHUNKS, $(date) ===" >> "$LOG"
  if [ -f "$CKPT" ]; then
    INIT_ARG="--init_checkpoint $CKPT"
  else
    INIT_ARG=""
  fi
  caffeinate -d -i -m -s python3 Scripts/models/dk_train.py \
    --device mps --train_split train_distant \
    --max_train_docs $CHUNK_SIZE --sample_seed $i \
    --epochs 1 --eval_every 100000 --patience 0 --seed 42 \
    --use_pu_loss --na_weight 0.5 \
    $INIT_ARG \
    --save_name distant_full_pu.pt >> "$LOG" 2>&1
  EXIT_CODE=$?
  if [ $EXIT_CODE -eq 0 ]; then
    echo "=== chunk $i done, $(date) ===" >> "$LOG"
  else
    echo "=== chunk $i FAILED (exit $EXIT_CODE), $(date), moving to next chunk anyway ===" >> "$LOG"
  fi
done

echo "=== all chunks attempted, starting stage 2 (annotated finetune), $(date) ===" >> "$LOG"

# stage 2: annotated finetune. epochs bumped 3->8 (patience 1->3) since real ATLOP
# uses 30 epochs on this same annotated set and 3 was likely leaving F1 on the table --
# distant pretrain isn't scaled up too (noisy labels, more passes = more overfitting to
# noise risk), but annotated is gold-labeled so more epochs is the safe/standard lever.
caffeinate -d -i -m -s python3 Scripts/models/dk_train.py \
  --device mps --train_split train_annotated \
  --init_checkpoint "$CKPT" \
  --epochs 8 --eval_every 3053 --patience 3 --seed 42 \
  --save_name final_full_pu.pt >> Scripts/models/logs/final_full_pu.log 2>&1
