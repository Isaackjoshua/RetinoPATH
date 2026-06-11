#!/usr/bin/env bash
# Model A — Retinopathy (R0/R1/R2/R3A), full fine-tune
#
# WHY full fine-tune after linear probe?
#   Updates all backbone weights with a low, layer-wise-decayed learning rate.
#   The backbone learns retinopathy-specific features rather than just fitting
#   a linear boundary on top of general retinal features.
#
# WHY layer_decay=0.65?
#   Early ViT layers learn low-level features (edges, textures) that transfer
#   well and shouldn't change much.  layer_decay multiplies the LR by 0.65
#   for each layer down from the head, so layer 24 gets full LR but layer 1
#   gets 0.65^23 ≈ 0.0002× — almost frozen.  0.65 is the RETFound default
#   and matches what the authors used for their benchmark tasks.
#
# WHY batch_size=24?
#   Full fine-tune stores gradients for all ~300M backbone parameters.
#   12 GB VRAM with batch_size=24 is comfortable; larger batches risk OOM.

set -e
cd "$(dirname "$0")"

ADAPTATION="finetune"
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
