# P2G — Minority-Class Aggressive Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Phase 2G — route R2/R3A training images through a geometry-heavy, lesion-safe augmentation while R0/R1 keep the exact P2B transform — as a single-variable change off P2B, and compare the result against P2B.

**Architecture:** The one novel piece (label→transform routing) lives in an importable, unit-tested module at repo root (`p2g_augmentation.py`). A deterministic build script clones `phase2b_full_finetune.ipynb` into `phase2g_minority_augmentation.ipynb`, patching only the cells that must change (title, config output dir, dataset, dataset instantiation, results). All loss/backbone/optimizer/training/helper cells stay byte-identical to P2B, which is what makes "single variable changed" provably true.

**Tech Stack:** PyTorch, torchvision transforms, timm `create_transform` (via `util.datasets.build_transform`), Jupyter notebook (plain-JSON `.ipynb`), conda env `retfound`.

## Global Constraints

- Python interpreter: `/home/eth-admin/miniconda3/envs/retfound/bin/python` (conda env `retfound`) — verbatim, for every command.
- `pytest` is NOT installed → tests are plain-assert scripts with a `__main__` runner, invoked directly with the interpreter above.
- Label column in `labels/splits.csv` is `image_path` (NOT `filepath`); grade map `{'R0':0,'R1':1,'R2':2,'R3A':3}`; `MINORITY_LABELS = {2, 3}` (R2, R3A).
- Everything except the train transform for R2/R3A images stays identical to P2B: FocalLoss(γ=2.0, weight=[1.0, 1.7851, 9.5294, 15.6774]); AdamW + LLRD 0.75; BASE_LR 5e-5 / MIN_LR 1e-7; warmup 5 / max 50; patience 10; batch 16 × accum 2; grad clip 1.0; StratifiedKFold(5, shuffle=True, random_state=42) on patient-level max grade; eval transform for val/test.
- Output dir: `output_dir/phase2g_cv/` — same artifact layout as `output_dir/phase2b_cv/`.
- New notebook lives at **repo root** (not a subdir) to keep relative paths working.
- Normalization constants: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.
- Success criterion: vs P2B on the same folds, R2 and/or R3A sensitivity improves AND R1 sensitivity does not drop below ~0.79. R1 collapse = P2G fails (do not tune to rescue — analyse).

---

### Task 1: Importable augmentation module + unit test

**Files:**
- Create: `p2g_augmentation.py` (repo root)
- Test: `tests/test_p2g_augmentation.py`

**Interfaces:**
- Consumes: `util.datasets.build_transform(is_train, args)` (existing).
- Produces:
  - `MINORITY_LABELS: frozenset = {2, 3}`
  - `build_standard_train_transform(input_size=224) -> Compose` (timm m9 + erase 0.25; identical to P2B train)
  - `build_eval_transform(input_size=224) -> Compose` (timm val transform)
  - `build_minority_train_transform(input_size=224) -> Compose` (geometry-heavy, no erasing)
  - `class P2GDataset(Dataset)` with `.transform_for(label) -> Compose` and `__getitem__ -> (tensor, int)`

- [ ] **Step 1: Write the failing test**

Create `tests/test_p2g_augmentation.py`:

```python
"""Plain-assert tests for P2G augmentation routing (no pytest available).
Run: /home/eth-admin/miniconda3/envs/retfound/bin/python tests/test_p2g_augmentation.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from p2g_augmentation import (
    MINORITY_LABELS,
    build_standard_train_transform,
    build_eval_transform,
    build_minority_train_transform,
    P2GDataset,
)

def _synthetic_image():
    arr = (np.random.rand(300, 300, 3) * 255).astype("uint8")
    return Image.fromarray(arr)

def _contains_random_erasing(compose):
    return any(type(t).__name__ == "RandomErasing" for t in compose.transforms)

def test_minority_labels_are_r2_r3a():
    assert MINORITY_LABELS == frozenset({2, 3})

def test_routing_selects_correct_transform():
    std = build_standard_train_transform()
    min_tf = build_minority_train_transform()
    eval_tf = build_eval_transform()
    ds = P2GDataset(records=[], standard_tf=std, minority_tf=min_tf,
                    eval_tf=eval_tf, train=True)
    assert ds.transform_for(0) is std
    assert ds.transform_for(1) is std
    assert ds.transform_for(2) is min_tf
    assert ds.transform_for(3) is min_tf

def test_eval_mode_always_uses_eval_transform():
    std = build_standard_train_transform()
    min_tf = build_minority_train_transform()
    eval_tf = build_eval_transform()
    ds = P2GDataset(records=[], standard_tf=std, minority_tf=min_tf,
                    eval_tf=eval_tf, train=False)
    for lbl in (0, 1, 2, 3):
        assert ds.transform_for(lbl) is eval_tf

def test_minority_transform_has_no_random_erasing():
    # The whole point of the geometry-heavy choice: protect tiny lesions.
    assert not _contains_random_erasing(build_minority_train_transform())

def test_standard_transform_does_have_random_erasing():
    # Sanity: baseline (P2B) really does carry erasing, so the difference is real.
    assert _contains_random_erasing(build_standard_train_transform())

def test_minority_transform_output_shape_and_normalized():
    t = build_minority_train_transform()(_synthetic_image())
    assert t.shape == torch.Size([3, 224, 224])
    assert t.dtype == torch.float32
    # Normalized images leave [0,1]: at least some negative values expected.
    assert t.min() < 0.0

def test_eval_transform_is_deterministic():
    img = _synthetic_image()
    t = build_eval_transform()
    assert torch.allclose(t(img), t(img))

def test_minority_transform_is_stochastic():
    img = _synthetic_image()
    t = build_minority_train_transform()
    assert not torch.allclose(t(img), t(img))

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/eth-admin/miniconda3/envs/retfound/bin/python tests/test_p2g_augmentation.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'p2g_augmentation'`

- [ ] **Step 3: Write the module**

Create `p2g_augmentation.py`:

```python
"""P2G — class-conditional augmentation for minority DR grades (R2, R3A).

The only novel piece of Phase 2G. R0/R1 images use the exact P2B timm
transform; R2/R3A images use a geometry-heavy, lesion-safe transform with
Random Erasing turned OFF (erasing can blank out the tiny microaneurysms /
exudates / neovascularisation that define the minority grades).

Kept importable so the routing is unit-tested independently of the
multi-hour GPU training in phase2g_minority_augmentation.ipynb.
"""
import argparse

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from util.datasets import build_transform

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MINORITY_LABELS = frozenset({2, 3})  # R2, R3A


def _p2b_aug_args(input_size):
    # Identical to phase2b_full_finetune.ipynb cell 4.
    return argparse.Namespace(
        input_size=input_size, color_jitter=None,
        aa="rand-m9-mstd0.5-inc1", reprob=0.25, remode="pixel", recount=1,
    )


def build_standard_train_transform(input_size=224):
    """R0/R1 train transform — timm RandAugment m9 + RandomErasing p=0.25."""
    return build_transform("train", _p2b_aug_args(input_size))


def build_eval_transform(input_size=224):
    """Val/test transform — deterministic resize + center crop (no aug)."""
    return build_transform("val", _p2b_aug_args(input_size))


def build_minority_train_transform(input_size=224):
    """R2/R3A train transform — geometry-heavy, lesion-safe, NO erasing."""
    return T.Compose([
        T.RandomResizedCrop(input_size, scale=(0.5, 1.0),
                            interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(20, interpolation=T.InterpolationMode.BICUBIC),
        T.ColorJitter(brightness=0.1, contrast=0.1),  # mild only
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        # NO RandomErasing — protects tiny grade-defining lesions.
    ])


class P2GDataset(Dataset):
    """Routes each image to a transform by label.

    records: list of (image_path, int_label).
    train=True  -> minority_tf for label in minority_labels, else standard_tf.
    train=False -> eval_tf for every label.
    """

    def __init__(self, records, standard_tf, minority_tf, eval_tf=None,
                 train=True, minority_labels=MINORITY_LABELS):
        self.records = records
        self.standard_tf = standard_tf
        self.minority_tf = minority_tf
        self.eval_tf = eval_tf
        self.train = train
        self.minority_labels = minority_labels

    def __len__(self):
        return len(self.records)

    def transform_for(self, label):
        if not self.train:
            return self.eval_tf
        return self.minority_tf if label in self.minority_labels else self.standard_tf

    def __getitem__(self, idx):
        path, label = self.records[idx]
        img = Image.open(path).convert("RGB")
        return self.transform_for(label)(img), label
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/eth-admin/miniconda3/envs/retfound/bin/python tests/test_p2g_augmentation.py`
Expected: `All 8 tests passed.`

- [ ] **Step 5: Commit**

```bash
git add p2g_augmentation.py tests/test_p2g_augmentation.py
git commit -m "feat(p2g): add tested class-conditional augmentation module

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Smoke test on real images (data path end-to-end)

Proves the routing works on **real fundus images through a DataLoader**: minority images get live (stochastic) augmentation, eval mode is deterministic, and batches have the right shape. CPU, runs in seconds. De-risks the data path before the multi-hour training run.

**Files:**
- Test: `tests/smoke_p2g_datapath.py`

**Interfaces:**
- Consumes: `p2g_augmentation` (Task 1), `labels/splits.csv`.

- [ ] **Step 1: Write the smoke test**

Create `tests/smoke_p2g_datapath.py`:

```python
"""Smoke test: P2G data path on real images.
Run: /home/eth-admin/miniconda3/envs/retfound/bin/python tests/smoke_p2g_datapath.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from p2g_augmentation import (
    build_standard_train_transform, build_eval_transform,
    build_minority_train_transform, P2GDataset, MINORITY_LABELS,
)

GRADE = {"R0": 0, "R1": 1, "R2": 2, "R3A": 3}

def main():
    df = pd.read_csv("labels/splits.csv")
    df["grade_int"] = df["retinopathy"].map(GRADE)
    df = df[df["split"].isin(["train", "val"])]

    # Tiny balanced-ish subset incl. minority classes.
    recs = []
    for g in (0, 1, 2, 3):
        sub = df[df["grade_int"] == g].head(8)
        recs += [(r.image_path, r.grade_int) for r in sub.itertuples()]
    assert any(lbl in MINORITY_LABELS for _, lbl in recs), "subset has no R2/R3A"
    print(f"Subset: {len(recs)} images, labels = {sorted({l for _,l in recs})}")

    std = build_standard_train_transform()
    minp = build_minority_train_transform()
    ev = build_eval_transform()

    # Train loader yields correct shape.
    ds_tr = P2GDataset(recs, std, minp, eval_tf=ev, train=True)
    xb, yb = next(iter(DataLoader(ds_tr, batch_size=8, shuffle=True)))
    assert xb.shape == torch.Size([8, 3, 224, 224]), xb.shape
    assert xb.dtype == torch.float32
    print("Train batch OK:", tuple(xb.shape))

    # Find a minority record; train mode is stochastic, eval mode is deterministic.
    mi = next(i for i, (_, l) in enumerate(recs) if l in MINORITY_LABELS)
    a, _ = ds_tr[mi]; b, _ = ds_tr[mi]
    assert not torch.allclose(a, b), "minority aug should be stochastic"
    ds_ev = P2GDataset(recs, std, minp, eval_tf=ev, train=False)
    c, _ = ds_ev[mi]; d, _ = ds_ev[mi]
    assert torch.allclose(c, d), "eval mode must be deterministic"
    print("Routing OK: minority train stochastic, eval deterministic.")
    print("\nSmoke test passed.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the smoke test**

Run: `/home/eth-admin/miniconda3/envs/retfound/bin/python tests/smoke_p2g_datapath.py`
Expected: ends with `Smoke test passed.` (If it errors that image files are missing, the `Data/` dir is gitignored — confirm images are present locally before the training run.)

- [ ] **Step 3: Commit**

```bash
git add tests/smoke_p2g_datapath.py
git commit -m "test(p2g): smoke-test data path on real images

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Generate the P2G notebook from P2B (deterministic patch)

**Files:**
- Create: `build_p2g_notebook.py` (repo root — provenance: documents exactly how P2G derives from P2B)
- Create (generated): `phase2g_minority_augmentation.ipynb` (repo root)

**Interfaces:**
- Consumes: `phase2b_full_finetune.ipynb` (source), `p2g_augmentation` (imported by the generated notebook).
- Produces: a notebook whose cells 0/1/4/11/14/15 are P2G-specific and all other cells byte-identical to P2B.

- [ ] **Step 1: Write the build script**

Create `build_p2g_notebook.py`:

```python
"""Derive phase2g_minority_augmentation.ipynb from phase2b_full_finetune.ipynb.

Patches only the cells that must change for P2G; every other cell (loss,
backbone, LLRD, training helpers, CV loop body, test ensemble) is copied
byte-for-byte, which is what makes "single variable changed" provable.
Run: /home/eth-admin/miniconda3/envs/retfound/bin/python build_p2g_notebook.py
"""
import json

SRC = "phase2b_full_finetune.ipynb"
DST = "phase2g_minority_augmentation.ipynb"


def lines(s):
    """Store source as a list of lines with trailing newlines (nbformat style)."""
    out = s.strip("\n").split("\n")
    return [l + "\n" for l in out[:-1]] + [out[-1]]


CELL0 = lines("""# Phase 2G — Minority-Class Aggressive Augmentation

**Single-variable change from Phase 2B.** Only the *training* transform for
**R2 / R3A** images changes; R0 / R1 keep the exact P2B transform, and loss,
class weights, sampler, optimizer, folds, and schedule are all identical to P2B.

**What R2/R3A get instead:** a geometry-heavy, lesion-safe transform
(RandomResizedCrop scale 0.5-1.0, flips, rotation +/-20 deg, mild colour),
with **Random Erasing turned off**. Erasing can blank out the tiny
microaneurysms / exudates / neovascularisation that define the minority
grades, so it is removed for exactly those classes.

**Why:** R2 (12 test patients) and R3A (9) are sparse; the model memorises
them rather than learning generalisable lesion features. Stronger
class-conditional augmentation manufactures effective variety for the
minority classes without touching the loss or class balance — avoiding the
R1-collapse failure mode seen in P2E (sampler + plain CE drove R1 to 0.000).

**Success criterion:** vs P2B on the same folds, R2 and/or R3A sensitivity
improves AND R1 sensitivity stays at roughly P2B's level (must not drop below
~0.79).""")

CELL4 = lines('''# ── P2G dataset: class-conditional augmentation ───────────────────────────────
from p2g_augmentation import (
    build_standard_train_transform, build_eval_transform,
    build_minority_train_transform, P2GDataset, MINORITY_LABELS,
)

standard_tf = build_standard_train_transform(INPUT_SIZE)  # R0/R1 — identical to P2B
minority_tf = build_minority_train_transform(INPUT_SIZE)  # R2/R3A — geometry-heavy
eval_tf     = build_eval_transform(INPUT_SIZE)            # val/test — no aug

def make_records(df_subset):
    return [(row.image_path, row.grade_int) for row in df_subset.itertuples()]

print(f"P2G transforms ready. Minority labels (geometry-heavy aug): {sorted(MINORITY_LABELS)}")''')

CELL14 = lines("## Results: Phase 2B (baseline) vs Phase 2G")

CELL15 = lines('''# ── P2G vs P2B comparison (argmax) ────────────────────────────────────────────
P2B_DIR = Path('output_dir/phase2b_cv')

def load_oof(d):
    probs  = np.load(d/'oof_probs_all.npy').astype(np.float64)
    labels = np.load(d/'oof_labels_all.npy')
    return labels, probs / probs.sum(axis=1, keepdims=True)

def load_test(d):
    probs  = np.load(d/'test_ensemble_probs.npy').astype(np.float64)
    labels = np.load(d/'test_ensemble_labels.npy')
    return labels, probs / probs.sum(axis=1, keepdims=True)

rows = [
    ('P2B FFT     Argmax', *load_oof(P2B_DIR),  *load_test(P2B_DIR)),
    ('P2G MinAug  Argmax', *load_oof(CV_OUTPUT), *load_test(CV_OUTPUT)),
]

for split_name in ('OOF', 'TEST'):
    print('\\n' + '=' * 100)
    print(f'  {split_name}')
    print('=' * 100)
    hdr = (f'{"Configuration":<20} | {"AUROC":>6} | {"Kappa":>6} | {"MacSens":>7} | '
           f'{" | ".join(f"{c:>6}" for c in CLASSES)}')
    print(hdr); print('-' * len(hdr))
    for name, ol, op, tl, tp in rows:
        lbl, prb = (ol, op) if split_name == 'OOF' else (tl, tp)
        m = compute_metrics(lbl, prb)
        s = m['sensitivity']
        print(f'{name:<20} | {m["auroc"]:>6.4f} | {m["kappa"]:>6.4f} | '
              f'{m["macro_sensitivity"]:>7.4f} | '
              f'{" | ".join(f"{s[i]:>6.4f}" for i in range(NUM_CLASSES))}')

# ── Success-criterion check (TEST, argmax) ────────────────────────────────────
_, p2b_tp = load_test(P2B_DIR)
p2g_tl, p2g_tp = load_test(CV_OUTPUT)
s_p2b = compute_metrics(*load_test(P2B_DIR))['sensitivity']
s_p2g = compute_metrics(p2g_tl, p2g_tp)['sensitivity']
print('\\n' + '=' * 60)
print('  SUCCESS CRITERION (test, argmax)')
print('=' * 60)
for i, c in enumerate(CLASSES):
    print(f'  {c:<4} sens: P2B {s_p2b[i]:.4f} -> P2G {s_p2g[i]:.4f}  '
          f'({"+" if s_p2g[i] >= s_p2b[i] else ""}{s_p2g[i]-s_p2b[i]:+.4f})')
r1_ok = s_p2g[1] >= 0.79
minority_up = (s_p2g[2] > s_p2b[2]) or (s_p2g[3] > s_p2b[3])
print(f'\\n  R1 floor (>=0.79): {"PASS" if r1_ok else "FAIL"}  (R1={s_p2g[1]:.4f})')
print(f'  Minority improved: {"YES" if minority_up else "NO"}')
print(f'  => P2G {"WIN" if (r1_ok and minority_up) else "does NOT beat P2B"}')

# ── Save P2G summary ──────────────────────────────────────────────────────────
m_p2g_oof = compute_metrics(*load_oof(CV_OUTPUT))
m_p2g_tst = compute_metrics(p2g_tl, p2g_tp)
summary = {
    'base': 'P2B', 'change': 'class-conditional geometry-heavy aug for R2/R3A (no erasing)',
    'minority_labels': sorted(MINORITY_LABELS),
    'oof':  {'macro_sensitivity': m_p2g_oof['macro_sensitivity'],
             'sensitivity': m_p2g_oof['sensitivity'].tolist(),
             'auroc': m_p2g_oof['auroc'], 'kappa': m_p2g_oof['kappa']},
    'test': {'macro_sensitivity': m_p2g_tst['macro_sensitivity'],
             'sensitivity': m_p2g_tst['sensitivity'].tolist(),
             'auroc': m_p2g_tst['auroc'], 'kappa': m_p2g_tst['kappa']},
    'success': {'r1_floor_pass': bool(r1_ok), 'minority_improved': bool(minority_up)},
}
with open(CV_OUTPUT / 'phase2g_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=float)
print(f'\\nSummary saved to {CV_OUTPUT}/phase2g_summary.json')''')


def patch_cell11(src_lines):
    """Swap the three RetinopathyDataset instantiations for P2GDataset."""
    text = "".join(src_lines)
    old = ("    ds_train = RetinopathyDataset(make_records(df_fold_train), train_transform)\n"
           "    ds_val   = RetinopathyDataset(make_records(df_fold_val),   eval_transform)\n"
           "    ds_test  = RetinopathyDataset(make_records(df_test),       eval_transform)")
    new = ("    ds_train = P2GDataset(make_records(df_fold_train), standard_tf, minority_tf,\n"
           "                          eval_tf=eval_tf, train=True)\n"
           "    ds_val   = P2GDataset(make_records(df_fold_val),   standard_tf, minority_tf,\n"
           "                          eval_tf=eval_tf, train=False)\n"
           "    ds_test  = P2GDataset(make_records(df_test),       standard_tf, minority_tf,\n"
           "                          eval_tf=eval_tf, train=False)")
    assert old in text, "cell 11 dataset block not found — P2B notebook changed?"
    return lines(text.replace(old, new))


def patch_cell1(src_lines):
    """Repoint output dir and chdir target to P2G."""
    text = "".join(src_lines)
    assert "output_dir/phase2b_cv" in text and "phase2b_full_finetune.ipynb" in text
    text = text.replace("output_dir/phase2b_cv", "output_dir/phase2g_cv")
    text = text.replace("phase2b_full_finetune.ipynb", "phase2g_minority_augmentation.ipynb")
    return lines(text)


def main():
    nb = json.load(open(SRC))
    cells = nb["cells"]
    cells[0]["source"] = CELL0
    cells[1]["source"] = patch_cell1(cells[1]["source"])
    cells[4]["source"] = CELL4
    cells[11]["source"] = patch_cell11(cells[11]["source"])
    cells[14]["source"] = CELL14
    cells[15]["source"] = CELL15
    # Clear stale outputs/exec counts so the new notebook starts clean.
    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None
    json.dump(nb, open(DST, "w"), indent=1)
    print(f"Wrote {DST} ({len(cells)} cells).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the build script**

Run: `/home/eth-admin/miniconda3/envs/retfound/bin/python build_p2g_notebook.py`
Expected: `Wrote phase2g_minority_augmentation.ipynb (16 cells).`
(If it raises `AssertionError`, the source P2B notebook differs from what this plan assumed — stop and reconcile before continuing.)

- [ ] **Step 3: Static-verify the generated notebook**

Run:
```bash
/home/eth-admin/miniconda3/envs/retfound/bin/python -c "
import json
nb = json.load(open('phase2g_minority_augmentation.ipynb'))
src = lambda i: ''.join(nb['cells'][i]['source'])
assert 'output_dir/phase2g_cv' in src(1)
assert 'phase2b_cv' not in src(1)
assert 'P2GDataset' in src(4) and 'from p2g_augmentation import' in src(4)
assert 'P2GDataset(make_records(df_fold_train)' in src(11)
assert 'RetinopathyDataset(' not in src(11)
assert 'phase2g_summary.json' in src(15)
# Unchanged-from-P2B cells must be byte-identical (single-variable proof).
b = json.load(open('phase2b_full_finetune.ipynb'))
for i in [2,3,5,6,7,8,9,10,12,13]:
    assert ''.join(nb['cells'][i]['source']) == ''.join(b['cells'][i]['source']), f'cell {i} drifted'
print('Static checks passed: P2G notebook is a clean single-variable derivative of P2B.')
"
```
Expected: `Static checks passed: P2G notebook is a clean single-variable derivative of P2B.`

- [ ] **Step 4: Commit**

```bash
git add build_p2g_notebook.py phase2g_minority_augmentation.ipynb
git commit -m "feat(p2g): generate notebook as single-variable derivative of P2B

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Run the full 5-fold experiment and evaluate

This is the heavy step — a full 5-fold fine-tune of a 307M-param ViT-L on the RTX 3060 (multi-hour, same runtime profile as P2B). Requires the GPU and the local `Data/` images (gitignored).

**Files:**
- Modify (run/populate): `phase2g_minority_augmentation.ipynb`
- Produces: `output_dir/phase2g_cv/` artifacts + `phase2g_summary.json`

**Interfaces:**
- Consumes: `phase2g_minority_augmentation.ipynb` (Task 3), `p2g_augmentation` (Task 1), local `Data/` images.

- [ ] **Step 1: Pre-flight — confirm GPU + images present**

Run:
```bash
/home/eth-admin/miniconda3/envs/retfound/bin/python -c "
import torch, pandas as pd, os
print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
df = pd.read_csv('labels/splits.csv')
p = df.iloc[0]['image_path']
print('sample image exists:', os.path.exists(p), '->', p)
"
```
Expected: CUDA True (RTX 3060) and `sample image exists: True`. If image is missing, stop — the `Data/` dir must be present locally.

- [ ] **Step 2: Execute the notebook end-to-end**

Run (long — hours; runs all folds and writes artifacts + comparison):
```bash
/home/eth-admin/miniconda3/envs/retfound/bin/python -m jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  phase2g_minority_augmentation.ipynb
```
Expected: completes without a cell error; per-fold logs show `AUROC=...` rising and early-stop messages.

- [ ] **Step 3: Verify artifacts were written**

Run:
```bash
ls -1 output_dir/phase2g_cv/ && echo '---' && \
/home/eth-admin/miniconda3/envs/retfound/bin/python -c "
import numpy as np, json
oof = np.load('output_dir/phase2g_cv/oof_probs_all.npy')
tst = np.load('output_dir/phase2g_cv/test_ensemble_probs.npy')
assert oof.shape == (4075, 4), oof.shape
assert tst.shape == (702, 4), tst.shape
print('OOF', oof.shape, 'TEST', tst.shape)
print(json.load(open('output_dir/phase2g_cv/phase2g_summary.json'))['success'])
"
```
Expected: 11 fold files + `best_fold_{0-4}.pth` + summary; shapes `(4075, 4)` / `(702, 4)`; a `success` dict printed.

- [ ] **Step 4: Read the comparison and decide WIN / FAIL**

Open the executed notebook's final cell output (the `SUCCESS CRITERION` block). Apply the criterion verbatim:
- **WIN** if R1 sensitivity ≥ 0.79 AND (R2 or R3A sensitivity improved vs P2B).
- **FAIL — R1 collapse** if R1 < 0.79: do NOT tune to rescue it; record the result and stop (this is the P2E lesson — geometry-only aug was not supposed to do this, so a collapse is a finding, not a knob to turn).
- **No improvement** if R1 holds but minorities don't move: record and stop.

- [ ] **Step 5: Commit results**

```bash
git add phase2g_minority_augmentation.ipynb output_dir/phase2g_cv/
git commit -m "feat(p2g): run 5-fold minority-augmentation experiment + P2B comparison

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 6: Update CLAUDE.md status**

In `CLAUDE.md`, move P2G from "Planned Experiments" to "Completed Phases" with its one-line key result (e.g. `MacroSens X, R1 Y, R2 Z, R3A W — WIN/FAIL vs P2B`), and add `phase2g_cv/` to "Saved Artifacts". Commit:

```bash
git add CLAUDE.md
git commit -m "docs: record P2G result in CLAUDE.md

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:**
- Goal / single-variable claim → Task 3 static check proves byte-identical unchanged cells. ✓
- Mechanism (label→transform routing) → Task 1 module + tests. ✓
- Geometry-heavy, no-erasing transform → Task 1 `build_minority_train_transform` + erasing-absence test. ✓
- Fixed P2B hyperparameters → inherited unchanged via notebook derivation (Task 3), asserted in static check. ✓
- Artifacts in `output_dir/phase2g_cv/` (same layout) → Task 4 Step 3 verifies shapes/files. ✓
- Success criterion (R2/R3A up, R1 ≥ ~0.79) → encoded in generated cell 15 + Task 4 Step 4. ✓
- Cost heads-up → Task 4 framing. ✓
- Out of scope (no loss/sampler/fold changes, no TTA, no aug sweep) → nothing in plan adds these. ✓

**2. Placeholder scan:** No TBD/TODO; every code step contains full code; every command has expected output. ✓

**3. Type consistency:** `build_standard_train_transform` / `build_minority_train_transform` / `build_eval_transform` / `P2GDataset` / `transform_for` / `MINORITY_LABELS` used identically across module, tests, smoke, and generated notebook. ✓
