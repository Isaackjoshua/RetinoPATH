# Phase 6 — 3-class (R0/R1/R2) variant of the recommended model

**Date:** 2026-06-23
**Status:** Design approved, pending spec review
**Owner:** Isaack Joshua

## Goal

Train the recommended architecture (P2B: RETFound-**DINOv2**-MEH full fine-tune) as a
**3-class** problem — R0/R1/R2, with R3A removed — and measure performance. Same
backbone, LLRD, focal loss, transforms, TTA, and patient-mean pooling; only the
label space (4→3) and the cohort change.

## Non-goals / constraints

- **Do not affect current progress.** No existing file is modified: not P2B/P4/P5
  notebooks/modules, not any `output_dir/*` artifact, not the recommended config,
  not `CLAUDE.md` (updated only after a result exists, in the final task).
- No git worktree (notebooks stay at repo root for relative data paths). All outputs
  to a new `output_dir/phase6_3class_cv/` (gitignored).
- Same-architecture discipline: the backbone, optimizer, schedule, transforms, TTA,
  and pooling match P2B exactly. Only label count, cohort filter, class weights, and
  folds change (the minimum the 3-class reframe requires).

## Cohort construction (drop R3A patients)

- Compute each patient's max grade over all eyes (the project convention: patient
  label = worst grade). **Keep only patients whose max grade ∈ {R0, R1, R2}.** Any
  patient with an R3A eye is dropped entirely, so no R3A images remain and labels
  stay truthful (no PDR-as-R2 relabeling).
- Grade map `{'R0':0, 'R1':1, 'R2':2}`, `NUM_CLASSES = 3`.
- Test set after filtering: **R0=173, R1=116, R2=20** patients (drops the 14 R3A).

## Architecture

Reuses the P2B pipeline (`phase2b_full_finetune.ipynb`) via a generator clone.

### Changes vs P2B (the minimum the 3-class reframe requires)

1. **`NUM_CLASSES = 4 → 3`** — the DINOv2 head is built with 3 outputs; `load_backbone_fft`
   and `build_llrd_optimizer` are otherwise unchanged.
2. **Cohort filter** — in the splits-loading cell, drop patients whose max grade is
   R3A before building `df_cv` / `df_test` / `pat_grade`.
3. **Class weights** — recomputed in-notebook from the filtered training distribution
   (inverse-frequency, normalised so the majority class = 1.0, matching P2B's scheme).
   The hardcoded 4-class `[1.0, 1.796, 10.8469, 17.502]` no longer applies.
4. **Folds** — `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` re-run on
   the 3-class patient set. Membership differs from P2B (R3A patients removed); per-fold
   numbers will not line up with P2B, which is expected for a different task.
5. **Output dir** — `output_dir/phase6_3class_cv/`.
6. **Kernel** — notebook `metadata.kernelspec.name = 'retfound'`; runs pass
   `nbconvert --ExecutePreprocessor.kernel_name=retfound` (the default `python3` kernel
   resolves to a broken `~/.local` py3.10 — learned in P5).

### Unchanged from P2B

`load_backbone_fft` (DINOv2-MEH loader), `build_llrd_optimizer`, focal γ=2.0,
BASE_LR 5e-5, LLRD_DECAY 0.75, WEIGHT_DECAY 0.05, WARMUP 5, MAX_EPOCHS 50, PATIENCE 10,
BATCH_SIZE 16, ACCUM_STEPS 2, train/eval transforms, 4-way TTA, PtMean pooling,
early-stop on val macro-AUROC.

## Build method

`build_p6_notebook.py` clones `phase2b_full_finetune.ipynb` and applies the changes
above by cell-source substitution + injection (same pattern as `build_p5_notebook.py`),
clears stale outputs, and pins the kernel. Produces `phase6_3class_pilot.ipynb`
(fold range `range(2)`).

## Data flow

`labels/splits.csv` → grade map → **drop R3A-max patients** → `df_cv` / `df_test` →
patient-stratified 3-class folds → recompute class weights → full fine-tune →
early-stop on val AUROC → restore best → save OOF (val) + test probs per fold.

## Outputs (all under `output_dir/phase6_3class_cv/`)

- `best_fold_{0,1}.pth` (pilot), `fold_{0,1}_{oof,test}_{probs,labels}.npy`
- `fold_results_pilot.json` (per-fold best/oof AUROC, kappa, macro-sens)
- On full run: `best_fold_{2,3,4}.pth`, their probs, `fold_{0-4}_test_tta_probs.npy`,
  `test_tta_probs.npy`

## Testing

- **Off-GPU test** (no GPU/training): a small `p6_cohort.py` helper holds the filter +
  weight recompute as pure functions, unit-tested on a synthetic dataframe:
  - `filter_r0r2_patients(df)` drops exactly the patients whose max grade is R3A and
    keeps all images of the rest;
  - `inverse_freq_weights(counts)` returns majority-class-normalised weights (R0→1.0).
- **Notebook generator test**: generated notebook contains `NUM_CLASSES = 3`,
  `output_dir/phase6_3class_cv`, `range(2)`, kernel `retfound`; no leftover
  `NUM_CLASSES = 4` / `phase2b_cv`.
- **GPU smoke** before the long run: build the DINOv2 backbone with `num_classes=3`,
  one forward+backward on a batch → logits shape `(B, 3)`.

## Scope & reporting (pilot first — P3 lesson)

1. Run **folds 0–1** only; save OOF/test probs; print 3-class metrics.
2. Report: accuracy, quadratic kappa, macro-AUROC, macro-sens, per-class R0/R1/R2
   sensitivity, confusion matrix.
3. Informative (not perfectly controlled) read: whether **R2 sensitivity** improves vs
   the 4-class R2=0.750 — removing R3A eliminates the R2↔R3A confusion that hurt R2.
4. Decision: if the pilot is coherent (no collapse), offer folds 2–4 + TTA + PtMean for
   the full picture. No fixed numeric gate (no baseline to beat — this measures a new task).
5. All GPU work runs behind the `nvidia-smi` ≥6000 MiB-free waiter; GPU 1 currently free.

## Risks

- **Cohort-filter correctness** (dropping the right patients) — pinned by the off-GPU
  unit test on a synthetic frame.
- **R2 still tiny** (20 test patients) — single-class sensitivity remains high-variance;
  report it but don't over-read one fold (pilot ≥2 folds enforced).
- **GPU contention** — waiter; pilot is 2 folds (~3–4 h).
