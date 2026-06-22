# Phase 4 — Lesion-Feature Multi-Task (P4-MT) Design

**Date:** 2026-06-22
**Model:** A (DR grading, R0/R1/R2/R3A)
**Base:** P2B (full fine-tune, focal loss + class weights, 224px) on the rebuilt 2,147-patient cohort
**Status:** Approved — ready for implementation plan

---

## 1. Goal

Lift macro-sensitivity past the ceiling that decision-rule tuning (~0.73–0.75) and higher
resolution (no gain) could not reach. Target **macro-sens > 0.80**; minimum bar **beat the
current best 0.7513** (P2B · PtMean · TTA · Argmax on the new cohort).

**Why this should work where the others didn't.** The bottleneck is the model's inability to
separate *adjacent* ordinal grades (R1↔R2↔R3A), because the distinguishing features are small
lesions. Threshold tuning can only reallocate a fixed confusion matrix; resolution didn't sharpen
separation. Auxiliary supervision on the actual lesion features that *define* the grade boundaries
teaches the backbone what to look for, improving the underlying separation rather than the decision
rule. The features were validated to carry clean, grade-discriminating signal (see §3).

---

## 2. Approach (chosen)

**Auxiliary multi-task.** Shared backbone, the existing 4-class grade head (main output), plus a
small auxiliary head predicting lesion features. Joint loss. Main output and inference path stay the
grade head — features only shape the learned representation.

Rejected alternatives:
- *Feature-based grading* (predict lesions → clinical rule → grade): feature labels too sparse for
  rare classes, rule mapping brittle, replaces the proven inference path. Higher risk.
- *Hybrid* (feature logits feed the grade head): higher ceiling but more moving parts to tune;
  defer unless plain auxiliary multi-task underperforms.

---

## 3. Auxiliary features (validated signal)

Four binary, per-eye lesion features from `Homterton_Reading_Centre_Grades.xlsx` (rank-1 rows).
Blank = absent (0), `1.0` = present (1). Validated prevalence by grade (% of eyes):

| Feature (key) | R0 | R1 | R2 | R3A | Role |
|---|---|---|---|---|---|
| `haem` — Retinal haemorrhage(s) | 0% | 48% | 97% | 76% | R1↔R2 severity gradient |
| `exud` — Any exudate (w/ DR features) | 0% | 43% | 80% | 49% | R1↔R2 gradient |
| `cws` — Cotton wool spots | 0% | 18% | 57% | 27% | R1↔R2 gradient |
| `nvd` — New vessels on disc | 0% | 0% | 0% | 30% | **R3A discriminator** |

`R0 = 0%` on every feature confirms blank = absent (clean labels, not missing data). `nvd` is
R3A-specific — directly targets the worst class. Source columns are the `… (Left Eye)` /
`… (Right Eye)` variants; each eye gets its own 4-vector.

Exact source column names:
- `Retinal haemorrhage(s) (<side>)`
- `Any exudate in the presence of other features of DR (<side>)`
- `Any number of cotton wool spots (CWS) in the presence of other features of DR (<side>)`
- `New vessels on disc (NVD) (<side>)`

where `<side> ∈ {Left Eye, Right Eye}` maps to `eye ∈ {LE, RE}`.

---

## 4. Architecture

```
RETFound ViT-L (vit_large_patch14_dinov2.lvd142m, 224px, shared, full fine-tune)
 ├── grade head    Linear(1024 → 4)  → R0/R1/R2/R3A      [main]
 └── feature head  Linear(1024 → 4)  → haem/exud/cws/nvd  [auxiliary, multi-label sigmoid]

loss = FocalLoss(grade, γ=2, weight=CLASS_WEIGHTS)
       + λ · BCEWithLogitsLoss(features, pos_weight=per-feature inverse prevalence)
λ = 0.5   (tune {0.3, 0.5, 1.0} in pilot if grading degrades or aux underfits)
```

- The feature head consumes the same pooled backbone embedding (`forward_features` → pooled
  1024-d) that the grade head uses. timm ViT exposes the pre-head feature via
  `model.forward_features(x)` then `model.forward_head(x, pre_logits=True)`; the implementation
  plan will pin the exact call. A thin wrapper module holds the backbone + both heads.
- `pos_weight` per feature = (neg count / pos count) on the training split, so rare features
  (esp. `nvd`) are not drowned out.

---

## 5. Data flow

1. **Extend `data_pipeline/build_label_table.py`** to also pull the 4 feature columns per eye
   (blank→0, `1`→1), emitting columns `haem, exud, cws, nvd` in `per_eye_labels.csv`.
2. `build_splits.py` carries those columns through to `splits.csv` unchanged (already propagates
   all per-eye columns).
3. **Dataset** returns `(image_tensor, grade_int, feature_vec)` where `feature_vec` is a
   float32 tensor `[haem, exud, cws, nvd]`. All images of an eye inherit that eye's labels.
4. Train/eval transforms unchanged from P2B (224px).

`splits.csv` stays **gitignored** (PHI in `image_path`); the new feature columns add no new PHI.

---

## 6. Training

Identical to P2B except the added head + loss term:
- FocalLoss γ=2.0, CLASS_WEIGHTS `[1.0, 1.796, 10.8469, 17.502]`, AdamW + LLRD 0.75,
  BASE_LR 5e-5, warmup 5 / max 50, patience 10 (early stop on val **grade** AUROC — unchanged),
  BATCH_SIZE 16 × ACCUM_STEPS 2, grad clip 1.0, grad checkpointing on.
- StratifiedKFold(5, SEED=42) on patient-level max grade.
- Output dir `output_dir/phase4_mt_cv/` (per-fold OOF/test grade probs, `best_fold_{k}.pth`).
- `torch.load(..., weights_only=True)` for the HF checkpoint (security-hardened pattern).

**Pilot:** train folds 0 and 1 only first. Compare OOF grade macro-sens + R3A sensitivity against
P2B's folds 0 and 1 (0.633 / 0.668 macro-sens; R3A 0.44 / 0.43). Commit the remaining folds only
if the pilot shows a clear, consistent lift across *both* folds (the 518 lesson: n=1 is not enough).

---

## 7. Inference (unchanged)

Grade head only → 4-way TTA → patient **mean** pooling → argmax. The auxiliary head is discarded
at inference (optionally inspected for interpretability — e.g. NVD activation on R3A cases). The
existing `phase2d`-style TTA + aggregation applies directly to `phase4_mt_cv` checkpoints.

---

## 8. Success criteria

- **Pilot gate:** folds 0 & 1 OOF grade macro-sens clearly above P2B's same folds, R3A not worse.
- **Full result:** OOF macro-sens > 0.682 (P2B 5-fold OOF) AND aggregated test (PtMean+TTA)
  macro-sens > 0.7513. **Target > 0.80.**
- **Watch:** R3A (driven by NVD head) and R1↔R2 (haem/exud/cws). R1 must not collapse (P2E lesson).

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Negative transfer (aux distracts grading, grade metrics drop) | Tune λ; features are grade-aligned (positive transfer likely); pilot catches it across 2 folds |
| `nvd` very sparse (~56 positive eyes cohort-wide) | `pos_weight` in BCE; even a weak NVD signal adds R3A-relevant gradient |
| Eye-level labels applied to all eye images / blank=absent assumption | Validated clean (R0 = 0% on all features); accept |
| Feature columns absent/renamed in a future Excel | `build_label_table.py` asserts the 4 columns exist before emitting |

---

## 10. Out of scope (YAGNI)

- No change to the inference pipeline, loss for grading, class weights, folds, or resolution.
- No hybrid feature→grade conditioning (deferred unless plain multi-task underperforms).
- No new aggregation/TTA logic — reuse the existing 224 PtMean+TTA path.
- No threshold tuning (capped ~0.75, established).
