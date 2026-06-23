# Phase 5 — RETFound-MAE backbone (single-variable vs P2B)

**Date:** 2026-06-23
**Status:** Design approved, pending spec review
**Owner:** Isaack Joshua

## Goal

Measure whether the RETFound **MAE**-MEH backbone beats the RETFound **DINOv2**-MEH
backbone (current P2B / recommended config) on the Model A DR-grading task
(R0/R1/R2/R3A), holding the entire P2B pipeline constant so any performance delta
is attributable to the backbone alone.

The two backbones share the same pretraining domain (Moorfields/MEH NHS fundus
data) but differ in:
- **SSL objective:** MAE (masked-autoencoder reconstruction) vs DINOv2 (self-distillation)
- **Patch size:** 16 (MAE) vs 14 (DINOv2)
- **Pooling head:** global average pool over patch tokens + `fc_norm` (MAE) vs
  DINOv2's pre-logits pooled embedding

## Non-goals / constraints

- **Do not affect current progress.** No existing file is modified: not P2B/P4
  notebooks, not `output_dir/phase2b_cv/` or `phase4_mt_cv/`, not the recommended
  config, not `CLAUDE.md` (CLAUDE.md is updated only *after* a result exists).
- No git worktree: notebooks must stay at repo root for relative data paths to
  resolve (per CLAUDE.md). `output_dir/` is gitignored, so a parallel output
  folder is fully isolated without a worktree.
- Single-variable discipline: **only** the backbone loader changes vs P2B. Same
  folds (StratifiedKFold seed=42), loss, weights, LR schedule, transforms, TTA,
  and patient pooling.

## Architecture

The MAE fine-tuning code already exists in the repo (`models_vit.py`,
`main_finetune.py`, `util/pos_embed.py`). Phase 5 reuses it, wrapped in the
P2B-style CV pipeline.

### New module: `p5_mae.py`

Mirrors the established `p4_multitask.py` pattern — importable so the wiring is
unit-tested off-GPU before any training.

- `load_backbone_mae(device, num_classes=4, seed=None)`:
  1. `model = models_vit.RETFound_mae(img_size=224, num_classes=num_classes,
     global_pool=True, drop_path_rate=0.2)` → a `vit_large_patch16` (patch 16,
     embed 1024, depth 24, heads 16, mlp_ratio 4, standard MLP).
  2. Download `hf_hub_download(repo_id="YukunZhou/RETFound_mae_meh",
     filename="RETFound_mae_meh.pth")`.
  3. Key hygiene + load, **exactly as `main_finetune.py` (lines 205–235)**:
     `checkpoint["model"]` → strip `backbone.` / remap `mlp.w12.→mlp.fc1.` /
     `mlp.w3.→mlp.fc2.` → drop `head.{weight,bias}` if shape-mismatched →
     `interpolate_pos_embed(model, checkpoint_model)` → `load_state_dict(..., strict=False)`
     → `trunc_normal_(model.head.weight, std=2e-5)`, zero head bias.
  4. All params `requires_grad=True` (full fine-tune). Return `model.to(device)`.

- **Forward contract:** `model(x)` returns `(B, num_classes)` logits. The custom
  `forward_features` global-pool path returns `(B, 1, 1024)` (keepdim), so the
  forward must squeeze the singleton token dim before `head`. This is the single
  integration risk and is pinned by a unit test (below).

### New notebook: `phase5_mae_pilot.ipynb`

A near-clone of `phase2b_full_finetune.ipynb`. Only the backbone-loader cell is
replaced (`load_backbone_fft` → `load_backbone_mae`). All other cells byte-identical:

- Focal loss γ=2, `CLASS_WEIGHTS=[1.0, 1.796, 10.8469, 17.502]`.
- BASE_LR 5e-5, LLRD 0.75, WEIGHT_DECAY 0.05, warmup 5, max 50, patience 10.
- BATCH_SIZE 16, ACCUM_STEPS 2 (effective 32).
- Train transform: RandomResizedCrop(224, scale 0.6–1.0) + flips + ColorJitter;
  eval transform: Resize(256, bicubic) + CenterCrop(224); ImageNet normalisation.
- StratifiedKFold(n_splits=5, shuffle=True, random_state=42) on patient max grade
  — **identical folds to P1/P2A/P2B/P4**, so OOF/test comparison is direct.

### MAE-specific adaptations (only two)

1. **LLRD `get_depth`:** MAE deletes `model.norm` and adds `fc_norm`. Map
   `fc_norm` → depth 1 (final-norm group); keep `head`→0, `blocks.X`→`24-X+1`,
   `patch_embed`/`cls_token`/`pos_embed`→26. `no_decay` matches
   `bias`/`norm`/`fc_norm`/`cls_token`/`pos_embed`.
2. **Gradient checkpointing OFF:** the custom `forward_features` loops blocks
   directly and does not honor timm's checkpointing, so it is a no-op here. The
   A6000 (48 GB) has headroom for ViT-L full fine-tune at batch 16 without it.
   **Fallback on OOM:** batch 8 × accum 4 (effective batch unchanged at 32).

## Data flow

Identical to P2B: `labels/splits.csv` → grade_int map → patient-stratified folds →
per-fold train/val datasets + shared test set → full fine-tune → early-stop on val
macro-AUROC → restore best → save OOF (val) + test probs per fold.

## Outputs (all under `output_dir/phase5_mae_cv/`)

- `best_fold_{0,1}.pth` (pilot), `fold_{0,1}_{oof,test}_{probs,labels}.npy`
- `fold_results_pilot.json` (per-fold best/oof AUROC + macro-sens)
- On full run: `best_fold_{2,3,4}.pth`, their probs, and TTA probs
  (`fold_{0-4}_test_tta_probs.npy`, `test_tta_probs.npy`)

## Testing

- **Off-GPU unit test in `p5_mae.py`** (run before training, like p4):
  - backbone builds and loads MEH weights without error;
  - `model(torch.randn(2,3,224,224)).shape == (2, 4)` (pins the forward contract);
  - LLRD optimizer builds, every trainable param assigned to exactly one group,
    `fc_norm` lands in the no-decay / depth-1 group.
- **Sanity:** focal-loss γ=0 == weighted CE (already in the P2B notebook, carried over).
- **Smoke:** one mini-batch forward+backward on GPU before the full epoch loop.

## Scope & decision gate (pilot first — P3 lesson)

1. Run **folds 0–1** only. Save OOF/test probs. Print OOF AUROC vs P2B's **0.911**.
2. **Gate:** if MAE pilot OOF AUROC ≥ ~0.906 (within ~0.005) or better → run folds
   2–4, then 4-way TTA + PtMean for the full test head-to-head vs the recommended
   config (Kappa 0.850 / macro-sens 0.751 / AUROC 0.948). If clearly worse → stop,
   record as a negative result.
3. All GPU work runs behind the `nvidia-smi` ≥6 GB-free waiter pattern (the other
   project intermittently holds both GPUs).

## Risks

- **Forward-shape mismatch** (global-pool keepdim) — mitigated by the off-GPU
  shape unit test before any GPU time.
- **GPU contention** — mitigated by the waiter; pilot is 2 folds (~3–4 h) not 5.
- **patch-16 vs patch-14 token count** — `interpolate_pos_embed` handles it; at
  224px MAE is 14×14 patches so no interpolation is even needed (no-op).
- **Marginal/negative result** is an acceptable outcome — the goal is to *measure*,
  and a clean negative closes the MAE question like P3/P4 did for resolution/multi-task.
