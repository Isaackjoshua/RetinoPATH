# P2G — Minority-Class Aggressive Augmentation (Design)

**Date:** 2026-06-20
**Model:** A (DR grading, R0/R1/R2/R3A)
**Base:** P2B (full fine-tune, focal loss + class weights + stratified 5-fold)
**Status:** Approved — ready for implementation plan

---

## 1. Goal

Lift sensitivity on the sparse minority classes **R2** and **R3A** *without* collapsing
R1 sensitivity. This is a **single-variable** change off P2B: only the train-time
augmentation for minority-class images changes. Everything else — loss, sampling,
class weights, optimizer, folds, schedule — stays byte-identical to P2B so the
comparison against P2B on the same folds is valid.

### Why augmentation (the technique, briefly)

R2 (12 test patients) and R3A (9) are extremely sparse. With so few examples the
model tends to **memorise** them rather than learn generalisable lesion features,
which caps minority sensitivity. Applying *stronger, class-conditional* augmentation
to only R2/R3A manufactures more *effective* variety per minority image, acting as a
regulariser against minority-class overfitting — **without** touching the loss or the
class balance. Because balance/loss are untouched, this avoids the R1-collapse failure
mode seen in P2E (WeightedRandomSampler + plain CE drove R1 to 0.000).

---

## 2. Mechanism

P2B applies one `train_transform` (timm `build_transform`) to every training image.
P2G keeps that exact transform for R0/R1 and routes R2/R3A through a **second,
stronger transform**, selected inside the Dataset by label.

- R0 / R1 → unchanged P2B timm transform (RandAugment m9 + RandomErasing p=0.25).
- R2 / R3A → new geometry-heavy, lesion-safe transform (defined below).
- Validation / test → unchanged eval transform (no augmentation).

The `RetinopathyDataset` already receives the integer label per record, so transform
selection is a per-item branch: `minority_tf if label in {2, 3} else standard_tf`.

---

## 3. Augmentation choice — geometry-heavy, lesion-safe

### Medical-validity rationale

P2B's existing augmentation already includes two knobs that are *risky for this task*
if turned up on exactly the minority classes:

- **Random Erasing** blanks a patch of the image. The lesions that *define* R2/R3A
  (microaneurysms, hard exudates, neovascularisation) are often tiny. Increasing
  erasing on minority images risks erasing the very feature that makes them R2/R3A —
  teaching the model the wrong thing.
- **Heavy colour jitter** can mask or fabricate the appearance of those lesions.

Geometry (rotation, flips, scale/zoom) is medically safe: a rotated/zoomed fundus is
still the same grade. So P2G leans on geometry for extra variety and **drops Random
Erasing to zero** for minority images, keeping colour mild.

### Implementation note (why hand-built, not timm)

timm's RandAugment string (e.g. `rand-m12-mstd0.5-inc1`) cannot be made geometry-only;
it still randomly fires photometric ops (solarize, posterize, colour) — exactly what we
agreed to avoid on minority images. To genuinely honour the geometry-heavy intent the
minority transform is **hand-built in torchvision**, giving exact control over which
ops run. R0/R1 continue to use the untouched timm P2B transform.

### Transforms

```python
import argparse
import torchvision.transforms as T
from util.datasets import build_transform

_aug_args = argparse.Namespace(
    input_size=INPUT_SIZE, color_jitter=None,
    aa='rand-m9-mstd0.5-inc1', reprob=0.25, remode='pixel', recount=1,
)

# R0 / R1 — identical to P2B
standard_tf = build_transform('train', _aug_args)
eval_tf     = build_transform('val',   _aug_args)

# R2 / R3A — geometry-heavy, lesion-safe (hand-built)
minority_tf = T.Compose([
    T.RandomResizedCrop(224, scale=(0.5, 1.0),
                        interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(20, interpolation=T.InterpolationMode.BICUBIC),
    T.ColorJitter(brightness=0.1, contrast=0.1),   # mild only
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # NO RandomErasing — protects tiny grade-defining lesions
])

MINORITY_LABELS = {2, 3}   # R2, R3A
```

### Dataset

```python
class RetinopathyDatasetP2G(Dataset):
    def __init__(self, records, standard_tf, minority_tf, eval_tf=None, train=True):
        self.records     = records          # list of (path, label)
        self.standard_tf = standard_tf
        self.minority_tf = minority_tf
        self.eval_tf     = eval_tf
        self.train       = train
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        path, label = self.records[idx]
        img = Image.open(path).convert('RGB')
        if not self.train:
            tf = self.eval_tf
        else:
            tf = self.minority_tf if label in MINORITY_LABELS else self.standard_tf
        return tf(img), label
```

`make_records` is unchanged (already yields `(image_path, grade_int)`).

---

## 4. What stays fixed (identical to P2B)

| Component | Value |
|---|---|
| Loss | FocalLoss(γ=2.0, weight=[1.0, 1.7851, 9.5294, 15.6774]) |
| Optimizer | AdamW + LLRD decay 0.75 |
| BASE_LR / MIN_LR | 5e-5 / 1e-7 |
| Warmup / Max epochs | 5 / 50 |
| Patience (early stop on val AUROC) | 10 |
| Batch / accum | 16 × 2 (effective 32) |
| Grad clip | 1.0 |
| Folds | StratifiedKFold(5, shuffle=True, random_state=42), patient-level max grade |
| Backbone | RETFound_dinov2_meh, grad checkpointing on |
| Val/test transform | eval transform (no aug) |

Only the **train transform for R2/R3A images** differs.

---

## 5. Artifacts

New notebook: `phase2g_minority_augmentation.ipynb` (repo root, to keep relative paths).
Output dir: `output_dir/phase2g_cv/` — same layout as P2B:

```
output_dir/phase2g_cv/
  fold_{0-4}_oof_probs.npy   fold_{0-4}_oof_labels.npy
  fold_{0-4}_test_probs.npy  fold_{0-4}_test_labels.npy
  best_fold_{0-4}.pth
  oof_labels_all.npy         oof_probs_all.npy
  fold_results.json
```

---

## 6. Success criterion

On OOF and on test, compared against P2B on the same folds:

- **Win:** R2 and/or R3A sensitivity improves *and* R1 sensitivity holds at roughly
  P2B's level (≈0.79, must not drop below ~0.79).
- **Fail:** R1 sensitivity collapses (the P2E failure mode) — geometry-only augmentation
  is not supposed to do this, so a collapse would be a signal to stop, not to tune.

Aggregation/TTA (P2D-style patient max pooling + 4-way TTA) is **out of scope** for
this spec; P2G first produces clean per-image fold probs, and aggregation is evaluated
separately if the base result is promising.

---

## 7. Cost

Full 5-fold fine-tune of a 307M-param ViT-L on an RTX 3060 (12 GB) — same multi-hour
runtime as P2B. Augmentation does not change training length.

---

## 8. Out of scope (YAGNI)

- No changes to loss, class weights, sampler, or folds.
- No new aggregation/TTA logic in this notebook.
- No tuning sweep over augmentation strength — one fixed geometry-heavy recipe; if it
  fails the success criterion we analyse why rather than grid-searching here.
