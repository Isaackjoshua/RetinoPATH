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

- Conda env: `retfound` — always use `/home/eth-admin/miniconda3/envs/retfound/bin/python`
- GPU: RTX A6000 48 GB (note: original P1–P2G runs were on an RTX 3060 12 GB; the P2B
  config — batch 16 × accum 2 + grad checkpointing — is still 12 GB-tuned and left as-is
  to keep results comparable, even though the A6000 has far more headroom)
- Working directory: `/home/eth-admin/Desktop/isaack/RETFound-main`
- Notebooks live at repo root (not in a subdirectory) to keep relative paths working

---

## Data

**Rebuilt 2026-06-20 on the more-complete image export `Data/Diabetic Retinopathy IMAGES 2/`.**
The original `Data/Diabetic Retinopathy IMAGES/` had ~2400 eye-folders empty of usable
images (only 1302 imaged patients); the new folder is a strict superset (2392 imaged
patients), nearly doubling the cohort and the rare classes. `splits.csv` + `per_eye_labels.csv`
were regenerated from it (`data_pipeline/build_label_table.py` + `build_splits.py`).

- Images: `Data/Diabetic Retinopathy IMAGES 2/` (note the space) — gitignored, not version-controlled
- Labels: `labels/splits.csv` — columns: `code, folder, eye, retinopathy, maculopathy, image_quality, image_path, split`
  — **gitignored (PHI: `image_path` embeds patient name + DOB); never commit**
- Split (current, new cohort): **2,147 patients / 8,844 images** — CV pool (train+val)
  1,824 patients / 7,495 images; test 323 patients / 1,349 images
- Test class distribution: **R0=173, R1=116, R2=20, R3A=14 patients** (R2/R3A far more
  robust than the old 12/9 — much more reliable minority eval)
- Grade mapping: `{'R0': 0, 'R1': 1, 'R2': 2, 'R3A': 3}`
- `image_path` column is the filepath field (NOT `filepath`)
- `ranking=1` selects the definitive adjudicated grade per patient (de-dups multiple reads,
  not a patient filter). Cohort defined by the grades spreadsheet, not the disk folders.
- Exclusions to reach `splits.csv`: must have a `ranking==1` grade, ≥1 usable (non-zero-byte)
  image, `image_quality=='Adequate'`, retinopathy ∈ {R0,R1,R2,R3A} (U / R3S dropped)

---

## Recommended Configuration (Model A) — new-data cohort (2026-06-20)

**P2B · Patient MEAN Pooling · 4-Way TTA · Argmax**
(On the new data PtMean overtook PtMax — the richer minority data favours mean-pooling.)

| Metric | Value | vs old-data best (PtMax+TTA) |
|---|---|---|
| Accuracy | 84.83% (274/323) | 85.71% |
| Cohen's Kappa (quadratic) | **0.8501** | 0.8220 |
| Macro AUROC | **0.9475** | 0.9370 |
| Macro Sensitivity | **0.7513** | 0.6999 |
| R0 Sensitivity | 0.9769 | 0.9780 |
| R1 Sensitivity | 0.7069 | 0.7937 |
| R2 Sensitivity | **0.7500** | 0.5833 |
| R3A Sensitivity | **0.5714** | 0.4444 |

Per-class AUROC: R0 0.931, R1 0.900, R2 0.980, R3A 0.979.

Runner-up: P2B + PtMax + Argmax (Kappa **0.8544**, MacroSens 0.7438, R1=0.750) — best Kappa
and best R1; prefer it if R1 ≥ 0.75 matters more than peak macro-sensitivity.

**Takeaway:** rebuilding on the new image export was the biggest lever for the rare classes —
R2 0.583→0.750 and R3A 0.444→0.571, with Kappa/AUROC/macro-sens all up, on a ~2× larger test set.

---

## Completed Phases (Model A)

**Key Result values below are the original OLD-data run.** P2B + P2D TTA + Evaluation were
**re-run on the new cohort (2026-06-20)** — see Recommended Configuration for the current numbers.

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
| Dataset Rebuild | `data_pipeline/build_*.py` | Rebuild on `…IMAGES 2` export | 1,165→2,147 patients; rare classes ~2× |
| P2B (new data) | `phase2b_full_finetune.ipynb` | Re-run on new cohort, recomputed class weights | OOF AUROC 0.911, Kappa 0.766 |
| P2D TTA (new data) | `phase2d_tta.ipynb` | 4-way TTA on new cohort | PtMean+TTA MacroSens 0.751, R3A 0.571 |
| Evaluation (new data) | `model_evaluation.ipynb` | Full metrics, PtMean+TTA | Acc 0.848, Kappa 0.850, AUROC 0.948 |
| P3 (518px) | `phase3_res518_*.ipynb`, `phase3_tta_eval.py` | Native 518px full fine-tune (single-var: INPUT_SIZE 224→518) | **NO GAIN — slightly worse.** Don't repeat. |
| P4 (lesion multi-task) | `phase4_mt_pilot.ipynb`, `phase4_mt_folds2to4.ipynb`, `p4_multitask.py`, `run_fold4_eval.py`, `run_fold_tta_test.py` | Shared-backbone aux head predicting 4 binary lesion features (haem/exud/cws/nvd), λ=0.5 BCE added to focal-grade loss (single-var vs P2B) | **NO GAIN — slightly worse.** Don't repeat. |

**Phase 3 / resolution — negative result (don't re-attempt).** Hypothesis: 518px (RETFound's
native res; pos_embed loads cleanly) would resolve the tiny lesions defining R1/R2/R3A and lift
macro-sensitivity past 0.80. Result: **518 ≈ 224 on OOF (0.684 vs 0.682) and slightly worse on
test** (PtMean+TTA macro-sens 0.726 vs 224's 0.751; R3A 0.50 vs 0.57; Kappa 0.830 vs 0.850).
The fold-0 pilot was a false positive (high single-fold variance on ~43 R3A / ~80 R2 per fold).
Lesson: pilot ≥2 folds. Also established: **threshold/decision-rule tuning is capped ~0.73-0.75**
(macro-sens is a zero-sum reallocation in single-label classification — boosting R2/R3A collapses
R1). To exceed 0.80 the **model** must better separate adjacent grades.

**Phase 4 / lesion multi-task — negative result (don't re-attempt).** Hypothesis: a shared-backbone
auxiliary head predicting 4 binary lesion features (haem/exud/cws/nvd from the grades spreadsheet),
trained jointly via `loss = focal(grade) + 0.5·BCE(features)`, would force the backbone to separate
adjacent grades and lift macro-sens past 0.80. Single-variable change vs P2B (same folds/seed/LLRD/
focal/weights; only the aux head + loss term added). Result: **at-or-below P2B everywhere.** 5-fold
aggregate OOF AUROC 0.915 vs P2B 0.911 (flat); test PtMean+TTA **macro-sens 0.719 vs 0.751, Kappa
0.833 vs 0.850, AUROC 0.942 vs 0.948, acc 0.817 vs 0.848**. Per-class test sens R0 0.977 / R1 0.629 /
R2 0.700 / R3A 0.571 — R3A *matched* P2B but R1 (−0.078) and R2 (−0.050) regressed. The aux lesion
signal did not sharpen adjacent-grade separation. **Two structural bets now negative (P3 resolution,
P4 multi-task)** — with this backbone + data the ~0.75 macro-sens ceiling is the binding constraint;
auxiliary supervision and resolution don't move it. (Pilot lesson held: folds 0–1 OOF AUROC 0.935/
0.922 looked promising but the full 5-fold + test recipe was flat — always confirm on test+TTA.)

---

## Planned Experiments

| Phase | Description | Status |
|---|---|---|
| P2F | Oversampling (WeightedRandomSampler) + P2B focal loss + class weights | Not started |
| P2G | Minority-class aggressive augmentation (stronger transforms for R2/R3A only) | Built (module + notebook), run superseded by dataset rebuild |

P2F uses P2B as the base — single-variable change. Output dir: `output_dir/phase2f_cv/`.
P2G was fully built (`p2g_augmentation.py`, `phase2g_minority_augmentation.ipynb`,
`docs/superpowers/specs|plans/2026-06-20-p2g-*`); its old-data run was stopped to free the GPU
for the dataset rebuild. Re-run on the new data only if minority sensitivity still needs lifting.

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
  phase4_mt_cv/       — P4 multi-task OOF + test probs (per-fold + TTA), best_fold_{0-4}.pth,
                        fold_{0-4}_test_tta_probs.npy, test_tta_probs.npy (negative result)

figures/              — All evaluation plots (confusion matrix, ROC, etc.)
reports/              — PDF reports + generator scripts
labels/splits.csv     — Ground truth + train/val/test split (GITIGNORED — PHI, local only)
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
CLASS_WEIGHTS = [1.0, 1.796, 10.8469, 17.502]   # inverse frequency, recomputed on new cohort
                                                 # (old data was [1.0, 1.7851, 9.5294, 15.6774])

# Loss
criterion = FocalLoss(gamma=2.0, weight=torch.tensor(CLASS_WEIGHTS))

# Folds: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Stratified on patient-level max grade
```

---

## Inference Pipeline

1. Load `test_tta_probs.npy` (shape: 1349×4 on new cohort, TTA-averaged over 5 folds)
2. Normalise rows to sum to 1
3. Patient MEAN pooling: for each patient, element-wise mean across image probs, re-normalise
   (recommended config; patient label = worst grade across the patient's eyes)
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
2. **Inverse-frequency class weights** [1.0, 1.80, 10.85, 17.50] (new cohort) — passed to focal loss
3. **Patient-level stratified K-fold** — ensures R2/R3A appear in every fold's validation set

At inference: patient pooling (mean, recommended) + TTA give minority classes more chances to be detected.

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
