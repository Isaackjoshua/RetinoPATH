# RetinoPATH — CLAUDE.md

Project context for Claude Code. Read this before starting any session.

---

## Project Goal

Two single-task models for automated diabetic retinopathy screening from NHS UK fundus photographs:
- **Model A** — DR grading: R0 (normal) / R1 (mild-mod NPDR) / R2 (mod-sev NPDR) / R3A (PDR)
- **Model B** — Maculopathy: M0 / M1 (binary) — not yet started

Backbone: **RETFound-DINOv2-MEH** (`YukunZhou/RETFound_dinov2_meh` on HuggingFace), ViT-Large, 307M params, 24 blocks, hidden dim 1024, patch size 14, img_size 224. Pre-trained on MEH/NHS UK fundus data — chosen for domain match.

---

## Environment

- Conda env: `retfound` — always use `/home/eth/miniconda3/envs/retfound/bin/python`
- GPU: RTX 3060 12 GB
- Working directory: `/home/eth/Desktop/Isaack/RETFound-main`
- Notebooks live at repo root (not in a subdirectory) to keep relative paths working

---

## Data

- Images: `Data/Diabetic Retinopathy IMAGES/` (note the space) — gitignored, not version-controlled
- Labels: `labels/splits.csv` — columns: `code, folder, eye, retinopathy, maculopathy, image_quality, image_path, split`
- Split: 4075 CV images (990 patients, split=train/val), 702 test images (175 patients, split=test)
- Test class distribution: R0=91, R1=63, R2=12, R3A=9 patients — R3A is very sparse
- Grade mapping: `{'R0': 0, 'R1': 1, 'R2': 2, 'R3A': 3}`
- `image_path` column is the filepath field (NOT `filepath`)
- `ranking=1` is the definitive grade per patient

---

## Recommended Configuration (Model A)

**P2B · Patient Max Pooling · 4-Way TTA · Argmax**

| Metric | Value |
|---|---|
| Accuracy | 85.71% (150/175) |
| Cohen's Kappa (quadratic) | 0.8220 |
| Macro AUROC | 0.9370 |
| Macro Sensitivity | 0.6999 |
| R0 Sensitivity | 0.9780 |
| R1 Sensitivity | 0.7937 |
| R2 Sensitivity | 0.5833 |
| R3A Sensitivity | 0.4444 |

Runner-up: P2B + PtMean + Argmax (AUROC 0.9456, Kappa 0.8212, R1=0.810) — better if R1 ≥ 0.80 is a hard requirement.

---

## Completed Phases (Model A)

| Phase | Notebook | Description | Key Result |
|---|---|---|---|
| P1 | `phase1_cv.ipynb` (not in repo) | 5-fold CV linear probe, CE loss | AUROC 0.826, Kappa 0.667 |
| P2C | (in aggregation nb) | Youden threshold tuning on P1 OOF | Didn't generalise to test |
| P2A | `phase2a_focal_loss.ipynb` | Linear probe + focal loss γ=2 | AUROC 0.818 |
| P2B | `phase2b_full_finetune.ipynb` | Full fine-tune, LLRD + grad ckpt + accum | AUROC 0.927, Kappa 0.767 |
| Aggregation | `phase2_patient_aggregation.ipynb` | Patient mean/max pooling, 18 configs | AUROC 0.946, Kappa 0.821 (PtMean) |
| P2D TTA | `phase2d_tta.ipynb` | 4-way TTA over 5 P2B models | MacroSens 0.700, R3A=0.444 (PtMax+TTA) |
| Evaluation | `model_evaluation.ipynb` | Full metrics for recommended config | See above |
| P2E | `phase2e_balanced_sampling.ipynb` | Side experiment: WeightedRandomSampler + plain CE | R1 collapsed to 0.000 — not viable |
| P2E Results | `phase2e_results.ipynb` | Results analysis for P2E | Documented R1 collapse |

---

## Planned Experiments

| Phase | Description | Status |
|---|---|---|
| P2F | Oversampling (WeightedRandomSampler) + P2B focal loss + class weights | Not started |
| P2G | Minority-class aggressive augmentation (stronger transforms for R2/R3A only) | Not started |

Both use P2B as the base — single-variable changes. Output dirs: `output_dir/phase2f_cv/` and `output_dir/phase2g_cv/`.

---

## Saved Artifacts

```
output_dir/
  phase1_cv/          — P1 OOF + test probs
  phase2a_cv/         — P2A OOF + test probs, phase2a_summary.json
  phase2b_cv/         — P2B OOF + test probs, best_fold_{0-4}.pth,
                        phase2b_summary.json, oof_tta_probs.npy, test_tta_probs.npy
  phase2c_thresholds/ — thresholds.json (P1 Youden thresholds)
  phase2e_cv/         — P2E OOF + test probs, best_fold_{0-4}.pth, phase2e_summary.json

figures/              — All evaluation plots (confusion matrix, ROC, etc.)
reports/              — PDF reports + generator scripts
labels/splits.csv     — Ground truth + train/val/test split
```

Threshold key structures:
- P1: `thresholds.json → youden_thresholds[c]`
- P2A: `phase2a_summary.json → youden_thresholds_p2a[c]`
- P2B: `phase2b_summary.json → youden_thresholds_p2b[c]`

---

## P2B Training Details

```python
# Key hyperparameters
BASE_LR       = 5e-5       # head learning rate
LLRD_DECAY    = 0.75       # per-block-group LR multiplier toward input
WEIGHT_DECAY  = 0.05
WARMUP_EPOCHS = 5
MAX_EPOCHS    = 50
PATIENCE      = 10         # early stop on val AUROC
BATCH_SIZE    = 16
ACCUM_STEPS   = 2          # effective batch = 32
FOCAL_GAMMA   = 2.0
CLASS_WEIGHTS = [1.0, 1.7851, 9.5294, 15.6774]  # inverse frequency

# Loss
criterion = FocalLoss(gamma=2.0, weight=torch.tensor(CLASS_WEIGHTS))

# Folds: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Stratified on patient-level max grade
```

---

## Inference Pipeline

1. Load `test_tta_probs.npy` (shape: 702×4, TTA-averaged over 5 folds)
2. Normalise rows to sum to 1
3. Patient max pooling: for each patient, element-wise max across image probs, re-normalise
4. Argmax → predicted grade

---

## 4-Way TTA

```python
tta_transforms = [
    lambda img: eval_tf(img),
    lambda img: eval_tf(TF.hflip(img)),
    lambda img: eval_tf(TF.vflip(img)),
    lambda img: eval_tf(TF.vflip(TF.hflip(img))),
]
# Average probability vectors across 4 augmentations per image
```

---

## Class Imbalance Handling (Current P2B)

Three techniques combined:
1. **Focal loss (γ=2)** — down-weights easy examples, focuses on hard/uncertain ones
2. **Inverse-frequency class weights** [1.0, 1.79, 9.53, 15.68] — passed to focal loss
3. **Patient-level stratified K-fold** — ensures R2/R3A appear in every fold's validation set

At inference: patient max pooling + TTA give minority classes more chances to be detected.

**P2E lesson:** WeightedRandomSampler with plain CE caused R1 to collapse to 0.000 sensitivity. The model over-fired on R2, misclassifying R1 as R2. Focal loss + class weights is more stable than the sampler alone.

---

## Key User Constraint

> "I must understand every decision. If you're about to write code that implements a technique I haven't seen before (focal loss, stratified k-fold, bootstrap confidence intervals, etc.), STOP and explain what it does, why it helps, and what would happen without it — BEFORE writing the code."

Always brief before building. Explain options and trade-offs. Ask or pick + justify at decision points.

---

## Eval Transform (fixed, no augmentation)

```python
T.Resize(256, interpolation=T.InterpolationMode.BICUBIC)
T.CenterCrop(224)
T.ToTensor()
T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
```

## Train Transform (P2B standard)

```python
T.RandomResizedCrop(224, scale=(0.6, 1.0))
T.RandomHorizontalFlip()
T.RandomVerticalFlip()
T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
T.ToTensor()
T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
```
