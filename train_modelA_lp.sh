#!/usr/bin/env bash
# Model A — Retinopathy (R0/R1/R2/R3A), linear probe
#
# WHY linear probe first?
#   Freezes the backbone, trains only the final classification head.
#   Runs in ~10 minutes and validates that RETFound features already separate
#   the classes before you commit hours to a full fine-tune.
#
# WHY these class weights?
#   Training counts: R0=1944, R1=1089, R2=204, R3A=124.
#   Balanced formula: weight_i = (N / n_classes) / count_i, then scaled so
#   min weight = 1.0.  R3A gets 15.7x the gradient signal of R0.
#
# WHY batch_size=64 for lp (vs 24 for full fine-tune)?
#   Backbone is frozen → no backbone gradients in memory → ~4x less VRAM.
#   Larger batch = fewer steps per epoch = faster wall-clock time.

set -e
cd "$(dirname "$0")"

ADAPTATION="lp"
MODEL="RETFound_dinov2"
MODEL_ARCH="retfound_dinov2"
FINETUNE="RETFound_dinov2_meh"
DATASET="modelA"
NUM_CLASS=4
DATA_PATH="image_trees/${DATASET}"
TASK="${MODEL_ARCH}_${DATASET}_${ADAPTATION}"

# Balanced class weights: R0  R1      R2      R3A
CLASS_WEIGHTS="1.0 1.7851 9.5294 15.6774"

torchrun --nproc_per_node=1 --master_port=48766 main_finetune.py \
  --model          "${MODEL}"        \
  --model_arch     "${MODEL_ARCH}"   \
  --finetune       "${FINETUNE}"     \
  --adaptation     "${ADAPTATION}"  \
  --savemodel                        \
  --global_pool                      \
  --batch_size     64                \
  --world_size     1                 \
  --epochs         50                \
  --nb_classes     "${NUM_CLASS}"    \
  --data_path      "${DATA_PATH}"    \
  --input_size     224               \
  --task           "${TASK}"         \
  --output_dir     "output_dir/${TASK}" \
  --log_dir        "output_logs/${TASK}" \
  --class_weights  ${CLASS_WEIGHTS}
