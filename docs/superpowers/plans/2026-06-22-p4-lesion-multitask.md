# Phase 4 — Lesion-Feature Multi-Task Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an auxiliary lesion-feature head (haem/exud/cws/nvd) to the P2B grade model so the backbone learns the features that define adjacent grades, aiming to lift macro-sensitivity past 0.7513 (target >0.80).

**Architecture:** A `MultiTaskRETFound` wrapper shares the RETFound ViT-L embedding between the existing grade head and a new 4-way multi-label feature head; trained with `FocalLoss(grade) + λ·BCEWithLogits(features)`. Single-variable change off P2B (224px, new cohort); inference uses the grade head only (unchanged TTA + PtMean).

**Tech Stack:** PyTorch, timm (`vit_large_patch14_dinov2.lvd142m`), Jupyter notebook (plain-JSON `.ipynb`), conda env `retfound`.

## Global Constraints

- Python interpreter: `/home/eth-admin/miniconda3/envs/retfound/bin/python` (verbatim, every command). `pytest` NOT installed → tests are plain-assert scripts with a `__main__` runner.
- Four auxiliary features (binary, per eye), decoded blank→0 / `1.0`→1, from these exact Excel columns (`<side>` ∈ {Left Eye, Right Eye}):
  - `haem` ← `Retinal haemorrhage(s) (<side>)`
  - `exud` ← `Any exudate in the presence of other features of DR (<side>)`
  - `cws`  ← `Any number of cotton wool spots (CWS) in the presence of other features of DR (<side>)`
  - `nvd`  ← `New vessels on disc (NVD) (<side>)`
- `FEATURE_NAMES = ['haem', 'exud', 'cws', 'nvd']` (this exact order everywhere).
- Loss: `FocalLoss(γ=2.0, weight=[1.0, 1.796, 10.8469, 17.502])` + `λ * BCEWithLogitsLoss(pos_weight=per-feature neg/pos)`, **λ = 0.5**.
- All other training identical to P2B: AdamW + LLRD 0.75, BASE_LR 5e-5, warmup 5 / max 50, patience 10 on val **grade** AUROC, BATCH_SIZE 16 × ACCUM_STEPS 2, grad clip 1.0, grad checkpointing on, StratifiedKFold(5, SEED=42) on patient-level max grade, 224px.
- **MODEL-PROTECTION GUARD:** P4 writes ONLY to `output_dir/phase4_mt_cv/`. It must NEVER write to `output_dir/phase2b_cv/` (the recommended model). A read-only backup exists at `model_backups/RECOMMENDED_P2B_224_PtMeanTTA/`.
- `torch.load(..., weights_only=True)` for the HF checkpoint.
- `labels/splits.csv` is gitignored (PHI in `image_path`); new feature columns add no new PHI.
- Pilot = folds 0 and 1 only; full 5-fold only after the pilot gate passes.

---

## File Structure

- `data_pipeline/build_label_table.py` (modify) — emit `haem,exud,cws,nvd` per eye.
- `p4_multitask.py` (create, repo root) — `FEATURE_NAMES`, `MultiTaskRETFound`, `build_multitask_model`, `MultiTaskLoss`, `P4Dataset`, `make_records_mt`, `compute_feature_pos_weight`.
- `tests/test_p4_multitask.py` (create) — CPU unit tests for the module.
- `tests/verify_p4_features.py` (create) — asserts the feature columns in `splits.csv` carry the validated signal.
- `build_p4_notebook.py` (create) — derives `phase4_mt_pilot.ipynb` from `phase2b_full_finetune.ipynb`.
- `phase4_mt_pilot.ipynb` (generated) — pilot training (folds 0–1).

---

### Task 1: Add lesion-feature columns to the label pipeline

**Files:**
- Modify: `data_pipeline/build_label_table.py` (the per-eye record loop)
- Test: `tests/verify_p4_features.py`

**Interfaces:**
- Produces: `labels/splits.csv` with added integer columns `haem, exud, cws, nvd` (0/1), grade/split assignment unchanged.

- [ ] **Step 1: Write the failing verification test**

Create `tests/verify_p4_features.py`:

```python
"""Verify P4 lesion-feature columns in splits.csv carry the validated signal.
Run: /home/eth-admin/miniconda3/envs/retfound/bin/python tests/verify_p4_features.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

FEATS = ['haem', 'exud', 'cws', 'nvd']

def main():
    df = pd.read_csv('labels/splits.csv')
    for c in FEATS:
        assert c in df.columns, f'missing feature column: {c}'
        assert set(df[c].dropna().unique()) <= {0, 1}, f'{c} not binary: {df[c].unique()}'
    # R0 must be 0% on every feature (validated: blank == absent)
    r0 = df[df['retinopathy'] == 'R0']
    for c in FEATS:
        assert r0[c].sum() == 0, f'R0 has nonzero {c} ({r0[c].sum()})'
    # haemorrhage prevalence must rise R1 -> R2 (severity gradient)
    pv = df.groupby('retinopathy')['haem'].mean()
    assert pv.get('R2', 0) > pv.get('R1', 0) > 0, f'haem gradient wrong: {pv.to_dict()}'
    # nvd must be R3A-specific (present in R3A, ~0 in R0/R1/R2)
    nvd = df.groupby('retinopathy')['nvd'].mean()
    assert nvd.get('R3A', 0) > 0.1, f'nvd not present in R3A: {nvd.to_dict()}'
    assert nvd.get('R1', 0) == 0 and nvd.get('R2', 0) == 0, f'nvd leaked to R1/R2: {nvd.to_dict()}'
    print('PASS: feature columns present, binary, R0=0, haem gradient, nvd R3A-specific.')

if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/home/eth-admin/miniconda3/envs/retfound/bin/python tests/verify_p4_features.py`
Expected: FAIL — `AssertionError: missing feature column: haem`

- [ ] **Step 3: Extend `build_label_table.py`**

In `data_pipeline/build_label_table.py`, after the line `df["folder"] = df["code"].str.replace(r"_T$", "", regex=True)` (the rank-1 filter / folder derivation, ~line 33), add the source-column guard:

```python
# P4 lesion features (per eye). Blank => absent (0); 1 => present. Validated:
# R0 is 0% on all four; haem/exud/cws rise R0->R2; nvd is R3A-specific.
_FEATURE_SRC = {
    "haem": "Retinal haemorrhage(s)",
    "exud": "Any exudate in the presence of other features of DR",
    "cws":  "Any number of cotton wool spots (CWS) in the presence of other features of DR",
    "nvd":  "New vessels on disc (NVD)",
}
for _side in ("Left Eye", "Right Eye"):
    for _src in _FEATURE_SRC.values():
        assert f"{_src} ({_side})" in df.columns, f"missing feature column: {_src} ({_side})"
```

Then, inside the unpivot loop, in the `records.append({...})` dict (the `for eye, side in [("LE", "Left Eye"), ("RE", "Right Eye")]` loop), add the four feature keys alongside `retinopathy`/`maculopathy`:

```python
            "haem":             int(row[f"{_FEATURE_SRC['haem']} ({side})"] == 1),
            "exud":             int(row[f"{_FEATURE_SRC['exud']} ({side})"] == 1),
            "cws":              int(row[f"{_FEATURE_SRC['cws']} ({side})"] == 1),
            "nvd":              int(row[f"{_FEATURE_SRC['nvd']} ({side})"] == 1),
```

(`build_splits.py` already propagates every per-eye column to `splits.csv`, so no change there.)

- [ ] **Step 4: Regenerate the label table + splits, then run the verification**

Run:
```bash
/home/eth-admin/miniconda3/envs/retfound/bin/python data_pipeline/build_label_table.py
/home/eth-admin/miniconda3/envs/retfound/bin/python data_pipeline/build_splits.py
/home/eth-admin/miniconda3/envs/retfound/bin/python tests/verify_p4_features.py
```
Expected: build scripts complete; verification prints `PASS: ...`.

- [ ] **Step 5: Confirm grade/split assignment is unchanged (sanity vs P2B)**

Run:
```bash
/home/eth-admin/miniconda3/envs/retfound/bin/python -c "
import pandas as pd
d=pd.read_csv('labels/splits.csv')
print('rows', len(d), 'patients', d['code'].nunique())
print(d.groupby('split')['code'].nunique().to_dict())
"
```
Expected: `rows 8844 patients 2147` and `{'test': 323, 'train': 1502, 'val': 322}` — identical to the P2B cohort (only feature columns added).

- [ ] **Step 6: Commit** (splits.csv is gitignored — only the script + test commit)

```bash
git add data_pipeline/build_label_table.py tests/verify_p4_features.py
git commit -m "feat(p4): emit lesion-feature columns (haem/exud/cws/nvd) per eye

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Multi-task module (model + loss + dataset)

**Files:**
- Create: `p4_multitask.py`
- Test: `tests/test_p4_multitask.py`

**Interfaces:**
- Consumes: a timm model created with a grade head (`num_classes=4`, `num_features=1024`).
- Produces:
  - `FEATURE_NAMES = ['haem','exud','cws','nvd']`
  - `MultiTaskRETFound(backbone, n_features=4)` → `forward(x) -> (grade_logits[B,4], feature_logits[B,4])`; `.set_grad_checkpointing(enable)`; `.backbone`
  - `build_multitask_model(backbone) -> MultiTaskRETFound`
  - `MultiTaskLoss(focal, feature_pos_weight, lam=0.5)` → `forward(grade_logits, grade_targets, feat_logits, feat_targets) -> scalar`
  - `P4Dataset(records, transform)` where records are `(path, grade_int, feat_list[4])` → `__getitem__ -> (img_tensor, grade_int, feat_tensor[4])`
  - `make_records_mt(df) -> list[(image_path, grade_int, [haem,exud,cws,nvd])]`
  - `compute_feature_pos_weight(df) -> tensor[4]` (neg/pos per feature, clamped ≥1)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_p4_multitask.py`:

```python
"""CPU unit tests for the P4 multi-task module (no GPU, no HF download).
Run: /home/eth-admin/miniconda3/envs/retfound/bin/python tests/test_p4_multitask.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, timm
from PIL import Image
import torchvision.transforms as T

from p4_multitask import (FEATURE_NAMES, MultiTaskRETFound, build_multitask_model,
                          MultiTaskLoss, P4Dataset, compute_feature_pos_weight)

def _backbone():
    return timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=False,
                             img_size=224, num_classes=4)

def test_feature_names_order():
    assert FEATURE_NAMES == ['haem', 'exud', 'cws', 'nvd']

def test_forward_shapes():
    m = build_multitask_model(_backbone()).eval()
    g, f = m(torch.randn(2, 3, 224, 224))
    assert g.shape == (2, 4) and f.shape == (2, 4)

def test_grade_path_unchanged():
    # multi-task wrapper must not alter the grade output vs the raw backbone
    bb = _backbone().eval()
    m = MultiTaskRETFound(bb).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(bb(x), m(x)[0], atol=1e-5)

def test_loss_is_finite_and_lambda_zero_equals_focal():
    focal = nn.CrossEntropyLoss()  # stand-in with same signature for the test
    pw = torch.ones(4)
    gl = torch.randn(3, 4); gt = torch.tensor([0, 1, 2])
    fl = torch.randn(3, 4); ft = torch.randint(0, 2, (3, 4)).float()
    full = MultiTaskLoss(focal, pw, lam=0.5)(gl, gt, fl, ft)
    assert torch.isfinite(full)
    only = MultiTaskLoss(focal, pw, lam=0.0)(gl, gt, fl, ft)
    assert torch.allclose(only, focal(gl, gt))

def test_dataset_returns_triplet(tmp=None):
    arr = (np.random.rand(40, 40, 3) * 255).astype('uint8')
    p = '/tmp/_p4_test.png'; Image.fromarray(arr).save(p)
    tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()])
    ds = P4Dataset([(p, 2, [1, 0, 1, 0])], tf)
    img, g, fv = ds[0]
    assert img.shape == torch.Size([3, 224, 224])
    assert g == 2 and fv.tolist() == [1.0, 0.0, 1.0, 0.0] and fv.dtype == torch.float32

def test_pos_weight_rewards_rare_features():
    import pandas as pd
    df = pd.DataFrame({'haem': [1, 1, 0, 0], 'exud': [1, 0, 0, 0],
                       'cws': [0, 0, 0, 0], 'nvd': [1, 0, 0, 0]})
    pw = compute_feature_pos_weight(df)
    assert pw.shape == (4,)
    assert pw[1] > pw[0]   # exud rarer (1 pos) than haem (2 pos) -> higher weight
    assert (pw >= 1).all() # clamped

if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn(); print(f'PASS {fn.__name__}')
    print(f'\nAll {len(fns)} tests passed.')
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/eth-admin/miniconda3/envs/retfound/bin/python tests/test_p4_multitask.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'p4_multitask'`

- [ ] **Step 3: Write `p4_multitask.py`**

```python
"""P4 — lesion-feature multi-task: shared-backbone model, loss, dataset.

The grade path is untouched (backbone.head); a parallel Linear head predicts
4 binary lesion features (haem/exud/cws/nvd) from the same pooled embedding.
Kept importable so the wiring is unit-tested off-GPU.
"""
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

FEATURE_NAMES = ['haem', 'exud', 'cws', 'nvd']


class MultiTaskRETFound(nn.Module):
    """Wrap a timm ViT (with a grade head) and add a lesion-feature head that
    shares the pooled pre-logits embedding."""

    def __init__(self, backbone, n_features=4):
        super().__init__()
        self.backbone = backbone
        self.feature_head = nn.Linear(backbone.num_features, n_features)

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        pooled = self.backbone.forward_head(feats, pre_logits=True)
        return self.backbone.head(pooled), self.feature_head(pooled)

    def set_grad_checkpointing(self, enable=True):
        self.backbone.set_grad_checkpointing(enable)


def build_multitask_model(backbone, n_features=4):
    return MultiTaskRETFound(backbone, n_features)


class MultiTaskLoss(nn.Module):
    """L = focal(grade) + lam * BCEWithLogits(features, pos_weight)."""

    def __init__(self, focal, feature_pos_weight, lam=0.5):
        super().__init__()
        self.focal = focal
        self.bce = nn.BCEWithLogitsLoss(pos_weight=feature_pos_weight)
        self.lam = lam

    def forward(self, grade_logits, grade_targets, feat_logits, feat_targets):
        return self.focal(grade_logits, grade_targets) + self.lam * self.bce(feat_logits, feat_targets)


class P4Dataset(Dataset):
    """records: list of (image_path, grade_int, feat_list[4 floats])."""

    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, grade, feats = self.records[idx]
        img = self.transform(Image.open(path).convert('RGB'))
        return img, grade, torch.tensor(feats, dtype=torch.float32)


def make_records_mt(df_subset):
    return [(r.image_path, r.grade_int, [getattr(r, f) for f in FEATURE_NAMES])
            for r in df_subset.itertuples()]


def compute_feature_pos_weight(df):
    """neg/pos per feature on the given split, clamped to >=1 (rare feature => higher weight)."""
    w = []
    for f in FEATURE_NAMES:
        pos = float(df[f].sum()); neg = float(len(df) - pos)
        w.append(max(1.0, neg / pos) if pos > 0 else 1.0)
    return torch.tensor(w, dtype=torch.float)
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/eth-admin/miniconda3/envs/retfound/bin/python tests/test_p4_multitask.py`
Expected: `All 6 tests passed.`

- [ ] **Step 5: Commit**

```bash
git add p4_multitask.py tests/test_p4_multitask.py
git commit -m "feat(p4): multi-task model, loss, and dataset module (tested)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Generate the P4 pilot notebook (folds 0–1)

**Files:**
- Create: `build_p4_notebook.py`
- Create (generated): `phase4_mt_pilot.ipynb`

**Interfaces:**
- Consumes: `phase2b_full_finetune.ipynb` (source), `p4_multitask` (imported by the notebook).
- Produces: a 2-fold multi-task training notebook writing to `output_dir/phase4_mt_cv/`.

- [ ] **Step 1: Write the build script**

Create `build_p4_notebook.py`:

```python
"""Derive phase4_mt_pilot.ipynb from phase2b_full_finetune.ipynb.

Patches the config (output dir, lambda), dataset cell, the train/eval helpers
(to handle the (img, grade, feat) batch + dual loss), and the CV loop (folds 0-1,
multi-task model + loss). All other cells are inherited byte-for-byte.
Run: /home/eth-admin/miniconda3/envs/retfound/bin/python build_p4_notebook.py
"""
import json

SRC, DST = "phase2b_full_finetune.ipynb", "phase4_mt_pilot.ipynb"

def lines(s):
    out = s.strip("\n").split("\n")
    return [l + "\n" for l in out[:-1]] + [out[-1]]

CELL0 = lines("""# Phase 4 (PILOT) — Lesion-Feature Multi-Task

Single-variable change from P2B: add an auxiliary head predicting 4 lesion
features (haem/exud/cws/nvd) that define adjacent grades. Loss =
focal(grade) + 0.5*BCE(features). Folds 0-1 pilot. Inference uses the GRADE
head only. Output: output_dir/phase4_mt_cv/ (NEVER touches phase2b_cv).""")

CELL4 = lines('''# ── P4 multi-task dataset + feature pos_weight ────────────────────────────────
import argparse
from util.datasets import build_transform
from p4_multitask import (FEATURE_NAMES, P4Dataset, make_records_mt,
                          compute_feature_pos_weight)

_aug_args = argparse.Namespace(
    input_size=INPUT_SIZE, color_jitter=None,
    aa='rand-m9-mstd0.5-inc1', reprob=0.25, remode='pixel', recount=1,
)
train_transform = build_transform('train', _aug_args)
eval_transform  = build_transform('val',   _aug_args)

FEATURE_POS_WEIGHT = compute_feature_pos_weight(
    df_cv if 'df_cv' in dir() else pd.read_csv('labels/splits.csv')
).to(DEVICE)
print('Feature names:', FEATURE_NAMES, '| pos_weight:', FEATURE_POS_WEIGHT.tolist())''')

# Patch cell 9: add multi-task train epoch + grade-only eval (append to existing helpers)
CELL9_APPEND = '''

# ── P4 multi-task train/eval (grade tuple-aware) ──────────────────────────────
def train_epoch_fft_mt(model, loader, optimizer, criterion, device, scaler, epoch):
    model.train()
    head_lr = get_lr(epoch, WARMUP_EPOCHS, MAX_EPOCHS, BASE_LR, MIN_LR)
    lr_scale = head_lr / BASE_LR
    for pg in optimizer.param_groups:
        pg['lr'] = pg['initial_lr'] * lr_scale
    optimizer.zero_grad()
    total_loss = 0.0; n = 0; step = 0
    for i, (imgs, grades, feats) in enumerate(loader):
        imgs, grades, feats = imgs.to(device), grades.to(device), feats.to(device)
        is_last = (i + 1 == len(loader))
        should_step = ((step + 1) % ACCUM_STEPS == 0) or is_last
        with torch.cuda.amp.autocast():
            g_logits, f_logits = model(imgs)
            loss = criterion(g_logits, grades, f_logits, feats) / ACCUM_STEPS
        scaler.scale(loss).backward()
        total_loss += loss.item() * ACCUM_STEPS * len(grades); n += len(grades); step += 1
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(); step = 0
    return total_loss / n, head_lr

@torch.no_grad()
def eval_fold_mt(model, loader, device):
    model.eval(); all_labels, all_probs = [], []
    for imgs, grades, feats in loader:
        with torch.cuda.amp.autocast():
            g_logits, _ = model(imgs.to(device))
        all_probs.append(torch.softmax(g_logits, dim=1).cpu().float()); all_labels.append(grades)
    return torch.cat(all_labels).numpy(), torch.cat(all_probs).numpy()

print('P4 multi-task train/eval helpers defined.')'''

CELL11 = lines('''# ── P4 multi-task CV loop (folds 0-1 pilot) ───────────────────────────────────
from p4_multitask import build_multitask_model, MultiTaskLoss

weight_tensor = torch.tensor(CLASS_WEIGHTS, dtype=torch.float).to(DEVICE)
focal_cv = FocalLoss(gamma=FOCAL_GAMMA, weight=weight_tensor)
criterion_cv = MultiTaskLoss(focal_cv, FEATURE_POS_WEIGHT, lam=P4_LAMBDA)

oof_labels_all = np.zeros(len(df_cv), dtype=np.int64)
oof_probs_all  = np.zeros((len(df_cv), NUM_CLASSES), dtype=np.float32)
fold_results   = []

for fold in range(2):  # PILOT — folds 0,1 only
    print(f'\\n{"="*60}\\n  FOLD {fold+1}/2  [P4 multi-task, lambda={P4_LAMBDA}]\\n{"="*60}')
    val_pats   = pat_grade[pat_grade['fold'] == fold]['code'].values
    train_pats = pat_grade[pat_grade['fold'] != fold]['code'].values
    df_fold_train = df_cv[df_cv['code'].isin(train_pats)]
    df_fold_val   = df_cv[df_cv['code'].isin(val_pats)]
    val_cv_indices = df_fold_val['cv_idx'].values

    ds_train = P4Dataset(make_records_mt(df_fold_train), train_transform)
    ds_val   = P4Dataset(make_records_mt(df_fold_val),   eval_transform)
    ds_test  = P4Dataset(make_records_mt(df_test),       eval_transform)
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    loader_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    loader_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = build_multitask_model(load_backbone_fft(device=DEVICE, seed=SEED + fold)).to(DEVICE)
    model.set_grad_checkpointing(True)
    optimizer = build_llrd_optimizer(model, BASE_LR, WEIGHT_DECAY, decay=LLRD_DECAY)
    scaler  = torch.cuda.amp.GradScaler()
    ckpt    = CV_OUTPUT / f'best_fold_{fold}.pth'
    stopper = EarlyStoppingFFT(patience=PATIENCE, model=model, checkpoint_path=ckpt)

    for epoch in range(MAX_EPOCHS):
        tr_loss, cur_lr = train_epoch_fft_mt(model, loader_train, optimizer, criterion_cv, DEVICE, scaler, epoch)
        val_labels, val_probs = eval_fold_mt(model, loader_val, DEVICE)
        m = compute_metrics(val_labels, val_probs)
        print(f'  ep {epoch:02d} | loss={tr_loss:.4f} | AUROC={m["auroc"]:.4f} | sens={m["macro_sensitivity"]:.4f}')
        if stopper.step(m['auroc'], model):
            print(f'  Early stop epoch {epoch} (best AUROC={stopper.best_auroc:.4f})'); break

    stopper.restore(model, DEVICE)
    oof_labels, oof_probs = eval_fold_mt(model, loader_val, DEVICE)
    oof_labels_all[val_cv_indices] = oof_labels
    oof_probs_all[val_cv_indices]  = oof_probs
    test_labels_fold, test_probs_fold = eval_fold_mt(model, loader_test, DEVICE)
    np.save(CV_OUTPUT / f'fold_{fold}_oof_probs.npy',   oof_probs)
    np.save(CV_OUTPUT / f'fold_{fold}_oof_labels.npy',  oof_labels)
    np.save(CV_OUTPUT / f'fold_{fold}_test_probs.npy',  test_probs_fold)
    np.save(CV_OUTPUT / f'fold_{fold}_test_labels.npy', test_labels_fold)
    m_fold = compute_metrics(oof_labels, oof_probs)
    fold_results.append({'fold': fold, 'best_auroc': stopper.best_auroc,
                         'oof_auroc': m_fold['auroc'], 'oof_macro_sens': m_fold['macro_sensitivity']})
    print(f'  OOF AUROC {m_fold["auroc"]:.4f}  macroSens {m_fold["macro_sensitivity"]:.4f}')
    del model; torch.cuda.empty_cache()

with open(CV_OUTPUT / 'fold_results_pilot.json', 'w') as f:
    json.dump(fold_results, f, indent=2)
print('Pilot folds 0-1 complete.')''')

def patch_cell1(src):
    text = "".join(src)
    assert "output_dir/phase2b_cv" in text
    text = text.replace("output_dir/phase2b_cv", "output_dir/phase4_mt_cv")
    text = text.replace("phase2b_full_finetune.ipynb", "phase4_mt_pilot.ipynb")
    text = text.replace("FOCAL_GAMMA = 2.0", "FOCAL_GAMMA = 2.0\nP4_LAMBDA   = 0.5   # auxiliary feature-loss weight")
    return lines(text)

def main():
    nb = json.load(open(SRC)); cells = nb["cells"]
    cells[0]["source"] = CELL0
    cells[1]["source"] = patch_cell1(cells[1]["source"])
    cells[4]["source"] = CELL4
    cells[9]["source"] = lines("".join(cells[9]["source"]) + CELL9_APPEND)
    cells[11]["source"] = CELL11
    nb["cells"] = cells[:12]   # drop ensemble/results cells (pilot)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []; c["execution_count"] = None
    json.dump(nb, open(DST, "w"), indent=1)
    print(f"Wrote {DST} ({len(nb['cells'])} cells).")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the build script**

Run: `/home/eth-admin/miniconda3/envs/retfound/bin/python build_p4_notebook.py`
Expected: `Wrote phase4_mt_pilot.ipynb (12 cells).`

- [ ] **Step 3: Static-verify (incl. the model-protection guard)**

Run:
```bash
/home/eth-admin/miniconda3/envs/retfound/bin/python -c "
import json
nb=json.load(open('phase4_mt_pilot.ipynb')); src=lambda i: ''.join(nb['cells'][i]['source'])
full=''.join(src(i) for i in range(len(nb['cells'])))
assert 'phase4_mt_cv' in src(1) and 'P4_LAMBDA' in src(1)
assert 'phase2b_cv' not in full, 'GUARD FAILED: references phase2b_cv'   # never touch recommended model
assert 'P4Dataset' in src(4) and 'compute_feature_pos_weight' in src(4)
assert 'train_epoch_fft_mt' in src(9) and 'eval_fold_mt' in src(9)
assert 'build_multitask_model' in src(11) and 'for fold in range(2):' in src(11)
b=json.load(open('phase2b_full_finetune.ipynb'))
for i in [2,3,5,6,7,8,10]:
    assert ''.join(nb['cells'][i]['source'])==''.join(b['cells'][i]['source']), f'cell {i} drifted'
print('Static checks passed: P4 pilot writes phase4_mt_cv only; never touches phase2b_cv.')
"
```
Expected: `Static checks passed: ...`. (If the `phase2b_cv` guard fails, STOP — do not run.)

- [ ] **Step 4: Commit**

```bash
git add build_p4_notebook.py phase4_mt_pilot.ipynb
git commit -m "feat(p4): generate multi-task pilot notebook (folds 0-1, phase4_mt_cv)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Run the pilot (folds 0–1) and evaluate the gate

This is a GPU step (~2 folds at 224px, ~2–3 h). Requires the local `Data/` images and a free GPU.

**Files:**
- Run/populate: `phase4_mt_pilot.ipynb`
- Produces: `output_dir/phase4_mt_cv/` (fold 0–1 artifacts)

**Interfaces:**
- Consumes: `phase4_mt_pilot.ipynb` (Task 3), `p4_multitask` (Task 2), regenerated `splits.csv` (Task 1).

- [ ] **Step 1: Pre-flight — GPU free + phase2b_cv intact**

Run:
```bash
mkdir -p output_dir/phase4_mt_cv
/home/eth-admin/miniconda3/envs/retfound/bin/python -c "import torch; print('CUDA', torch.cuda.is_available())"
ls model_backups/RECOMMENDED_P2B_224_PtMeanTTA/best_fold_0.pth && echo 'recommended backup intact'
```
Expected: CUDA True; backup present. (Do not start if another training is using the GPU.)

- [ ] **Step 2: Execute the pilot notebook**

Run (long):
```bash
/home/eth-admin/miniconda3/envs/retfound/bin/python -m jupyter nbconvert --to notebook \
  --execute --inplace --ExecutePreprocessor.timeout=-1 phase4_mt_pilot.ipynb
```
Expected: both folds train; per-epoch `AUROC=` / `sens=` lines; `Pilot folds 0-1 complete.`

- [ ] **Step 3: Confirm phase2b_cv was untouched**

Run:
```bash
ls -l output_dir/phase2b_cv/best_fold_0.pth   # mtime must be unchanged (Jun 20)
ls -l output_dir/phase4_mt_cv/fold_0_oof_probs.npy output_dir/phase4_mt_cv/fold_1_oof_probs.npy
```
Expected: `phase2b_cv/best_fold_0.pth` mtime still Jun 20; two new P4 fold files present.

- [ ] **Step 4: Evaluate the pilot gate (vs P2B folds 0–1)**

Run:
```bash
/home/eth-admin/miniconda3/envs/retfound/bin/python -c "
import numpy as np
from sklearn.metrics import roc_auc_score, cohen_kappa_score, confusion_matrix
def m(d,k):
    p=np.load(f'{d}/fold_{k}_oof_probs.npy').astype(np.float64); y=np.load(f'{d}/fold_{k}_oof_labels.npy')
    p=p/p.sum(1,keepdims=True); pr=p.argmax(1)
    cm=confusion_matrix(y,pr,labels=[0,1,2,3]); s=[cm[i,i]/cm[i].sum() if cm[i].sum() else np.nan for i in range(4)]
    return roc_auc_score(y,p,multi_class='ovr',average='macro',labels=[0,1,2,3]), np.nanmean(s), np.array(s)
for k in [0,1]:
    a=m('output_dir/phase2b_cv',k); b=m('output_dir/phase4_mt_cv',k)
    print(f'fold {k}: P2B mSens={a[1]:.3f} (R3A {a[2][3]:.2f}) -> P4 mSens={b[1]:.3f} (R3A {b[2][3]:.2f})  d={b[1]-a[1]:+.3f}')
"
```
Apply the gate: **PASS** if P4 macro-sens is clearly above P2B on *both* folds and R3A is not worse → proceed to full 5-fold (change `range(2)` → `range(5)` in a `phase4_mt_full.ipynb`, retrain folds 2–4, then run TTA + PtMean via `phase3_tta_eval.py` pointed at `phase4_mt_cv`). **FAIL** if it's a wash/worse → stop, record the negative result, and consider λ tuning ({0.3, 1.0}) or the LDAM alternative. Do NOT proceed to 5-fold on a single good fold (the 518 lesson).

- [ ] **Step 5: Commit pilot result**

```bash
git add phase4_mt_pilot.ipynb
git commit -m "feat(p4): run multi-task pilot (folds 0-1) + gate evaluation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 6: Record outcome in CLAUDE.md**

Add a P4 row to the Completed Phases table with the pilot gate result (PASS→full run planned, or FAIL→negative result + reason). Commit:

```bash
git add CLAUDE.md
git commit -m "docs: record P4 multi-task pilot outcome

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:**
- §2 auxiliary multi-task → Task 2 model/loss. ✓
- §3 four features + exact columns + blank→0 → Task 1 (`_FEATURE_SRC`) + verify test. ✓
- §4 shared pooled embedding + λ=0.5 + pos_weight → Task 2 (`MultiTaskRETFound`, `MultiTaskLoss`, `compute_feature_pos_weight`); λ in Task 3 cell 1. ✓
- §5 data flow (build_label_table → splits → dataset triplet) → Task 1 + Task 2 `P4Dataset`/`make_records_mt`. ✓
- §6 training identical to P2B + phase4_mt_cv + weights_only → Task 3 cells (inherited P2B helpers) + guard. ✓
- §7 inference unchanged → Task 4 Step 4 reuses `phase3_tta_eval.py`. ✓
- §8 success/gate (pilot 2 folds, >0.682 OOF) → Task 4 Step 4. ✓
- §9 risks (negative transfer→λ; nvd sparse→pos_weight; label noise→verified) → Task 1 verify + Task 2 pos_weight. ✓
- MODEL-PROTECTION guard → Task 3 static check (`phase2b_cv not in full`) + Task 4 Step 3. ✓

**2. Placeholder scan:** No TBD/TODO; every code step has full code; commands have expected output. ✓

**3. Type consistency:** `FEATURE_NAMES` order, `MultiTaskRETFound(backbone)`, `build_multitask_model`, `MultiTaskLoss(focal, pos_weight, lam)`, `P4Dataset(records, transform)`, `make_records_mt`, `compute_feature_pos_weight`, `train_epoch_fft_mt`, `eval_fold_mt` used consistently across Tasks 2–4. ✓
