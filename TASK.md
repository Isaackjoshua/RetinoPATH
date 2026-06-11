# Task: fine-tune RETFound for diabetic retinopathy + maculopathy

## My learning goal — read this first
I'm learning ML through this project. Do NOT just execute. Before each
significant step, explain the options, tradeoffs, and why you chose what you
chose, in plain language. Pause at genuinely ambiguous decisions and either ask
me or pick a sensible default and justify it. Comment code with the *why*. Work
in small, reviewable steps. Treat me as a capable beginner.

## Goal
Fine-tune RETFound (already cloned here) to detect diabetic retinopathy and
maculopathy from colour fundus photos. FIRST PASS: two SEPARATE single-task
models using the stock repo as-is (no multi-task code changes yet):
- Model A — retinopathy, 4 classes: R0, R1, R2, R3A
- Model B — maculopathy, binary: M0, M1
Use the RETFound-DINOv2 / MEH checkpoint (UK fundus cohort matching this UK
screening data; it's already the repo default).

## Data layout (already in place)
- `data/Diabetic Retinopathy IMAGES/` — one folder per patient, named by the
  `code` column WITHOUT the trailing `_T` (code `0001_T` → folder `0001`).
- Each patient folder has `LE` and `RE` subfolders with that eye's image(s).
  Assume LE = Left Eye, RE = Right Eye (matches the grade column names) — flag
  this assumption for me to confirm.
- `data/Homterton_Reading_Centre_Grades.xlsx` — labels (sheet `Sheet1`).
- Note the space in the images folder name; handle it.

## What I already know about the metadata (verify, don't blindly trust)
- ~5,920 rows, 2,401 unique patients. Per-eye columns: `Retinopathy (Left Eye)`
  /`(Right Eye)` (R0/R1/R2/R3A/U); `Maculopathy (Left Eye)`/`(Right Eye)`
  (M0/M1/U); `Image quality (Left Eye)`/`(Right Eye)` (Adequate/Inadequate).
- `Age` is 74 for every row (placeholder, unusable) and `Ethnicity` is empty —
  so NO demographic/fairness analysis is possible; don't attempt it.
- Multiple grading rows per patient: `Procedure Kind` ∈ {Primary, Secondary,
  Arbitration, Referral Outcome} Retinal Grading, with a `ranking` column. Must
  collapse to ONE final grade per eye (see decisions to surface).
- Rough per-eye balance: R0 ~49%, R1 ~28%, U ~14%, R2 ~5%, R3A ~4%; M0 ~62%,
  M1 ~22%, U ~16%. R2/R3A are small — expect imbalance.

## Decisions already locked — apply these
1. Each EYE is a sample; use the per-eye grade columns + matching LE/RE folder.
2. Exclude U per task: drop U-retinopathy eyes from Model A, U-maculopathy eyes
   from Model B. An eye can be valid for one task and not the other.
3. Split by PATIENT (all of a patient's eyes/images in the same split) to
   prevent leakage. Recommend a ratio and explain.
4. Stratify the split by class where feasible so rare classes appear in each split.
5. Build two ImageFolder trees (`train/val/test/<class>/...`), one per model,
   using symlinks (not copies) to save disk.
6. Handle class imbalance (class weighting); report per-class metrics + AUROC,
   and sensitivity/specificity at a sensible threshold — not just overall AUROC.
7. train.sh: MODEL=RETFound_dinov2, MODEL_ARCH=retfound_dinov2,
   FINETUNE=RETFound_dinov2_meh, input_size 224. NUM_CLASS=4 (A) / 2 (B). Run
   ADAPTATION=lp (linear probe) first as a cheap baseline, then =finetune.

## Decisions you MUST surface and explain (don't silently choose)
- **Which grading row is the label** per eye, given Primary/Secondary/
  Arbitration/Referral Outcome rows. Explain the options and the `ranking`
  column, recommend one, tell me why.
- **Multiple images per eye:** check if eye folders hold more than one image. If
  so, decide and explain — label every image with the eye's grade vs one per
  eye — keeping all of an eye's images in the same split either way.
- **Exact split ratio and stratification** — state them and why.

## Suggested step order (adjust and explain as you go)
1. Set up env (conda python 3.11, torch 2.5.1, requirements.txt); confirm GPU;
   HuggingFace login for the gated RETFound weights.
2. Build a clean per-eye label table (one final grade per eye per task) after
   the grade-collapsing decision.
3. Link each eye to its images (strip `_T`; map LE/RE → Left/Right Eye grade).
   Report patients with no image folder and folders with no metadata.
4. Patient-level stratified split; print per-class counts per split for both
   tasks; assert the train/val/test patient sets are disjoint.
5. Build the two symlinked ImageFolder trees.
6. Sanity check: render a handful of images with their assigned class/eye/
   patient to confirm they're CFP (round colour fundus, not OCT) and that
   labels line up. Print final per-class counts.
7. Configure and run train.sh for Model A (linear probe, then fine-tune), then B.
8. Report AUROC + per-class metrics + sensitivity/specificity + confusion matrices.

Start by reading the README and train.sh, confirming the environment, then walk
me through the step-2 grade-collapsing decision before writing any label table.