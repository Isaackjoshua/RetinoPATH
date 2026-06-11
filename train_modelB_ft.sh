#!/usr/bin/env bash
# Model B — Maculopathy (M0/M1), full fine-tune

set -e
cd "$(dirname "$0")"

ADAPTATION="finetune"
MODEL="RETFound_dinov2"
MODEL_ARCH="retfound_dinov2"
FINETUNE="RETFound_dinov2_meh"
DATASET="modelB"
NUM_CLASS=2
DATA_PATH="image_trees/${DATASET}"
TASK="${MODEL_ARCH}_${DATASET}_${ADAPTATION}"

# Balanced class weights: M0  M1
CLASS_WEIGHTS="1.0 2.7807"

torchrun --nproc_per_node=1 --master_port=48766 main_finetune.py \
  --model          "${MODEL}"        \
  --model_arch     "${MODEL_ARCH}"   \
  --finetune       "${FINETUNE}"     \
  --adaptation     "${ADAPTATION}"  \
  --savemodel                        \
  --global_pool                      \
  --batch_size     24                \
  --world_size     1                 \
  --epochs         50                \
  --nb_classes     "${NUM_CLASS}"    \
  --data_path      "${DATA_PATH}"    \
  --input_size     224               \
  --task           "${TASK}"         \
  --output_dir     "output_dir/${TASK}" \
  --log_dir        "output_logs/${TASK}" \
  --class_weights  ${CLASS_WEIGHTS}
