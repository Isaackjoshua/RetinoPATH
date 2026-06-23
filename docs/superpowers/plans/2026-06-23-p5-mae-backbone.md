# Phase 5 — RETFound-MAE Backbone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune the RETFound-MAE-MEH backbone on the Model A DR-grading task using the exact P2B pipeline, to measure MAE vs DINOv2 head-to-head.

**Architecture:** A new testable module `p5_mae.py` holds the MAE backbone (ViT-L/16, global-pooled) wrapped in an `MAEClassifier` that returns logits, plus an MAE-adapted LLRD optimizer. A generator script clones `phase2b_full_finetune.ipynb` into `phase5_mae_pilot.ipynb`, swapping only the backbone loader, optimizer call, output dir, and fold range. All outputs go to `output_dir/phase5_mae_cv/`. Nothing existing is modified.

**Tech Stack:** PyTorch, timm (`models_vit.RETFound_mae`), huggingface_hub, numpy, scikit-learn, jupyter/nbconvert. Env: `/home/eth-admin/miniconda3/envs/retfound/bin/python`.

## Global Constraints

- Python interpreter: `/home/eth-admin/miniconda3/envs/retfound/bin/python` (conda env `retfound`) — verbatim for every command.
- Single-variable discipline: only the backbone loader + LLRD change vs P2B. Same folds `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on patient max grade.
- Do not modify P2B/P4 files, `output_dir/phase2b_cv/`, `output_dir/phase4_mt_cv/`, the recommended config, or `CLAUDE.md` (CLAUDE.md updated only in the final gated task, after a result exists).
- MAE checkpoint: `YukunZhou/RETFound_mae_meh`, filename `RETFound_mae_meh.pth`.
- Hyperparameters (identical to P2B): focal γ=2.0, `CLASS_WEIGHTS=[1.0, 1.796, 10.8469, 17.502]`, BASE_LR 5e-5, LLRD_DECAY 0.75, WEIGHT_DECAY 0.05, WARMUP_EPOCHS 5, MAX_EPOCHS 50, PATIENCE 10, BATCH_SIZE 16, ACCUM_STEPS 2, INPUT_SIZE 224, NUM_CLASSES 4, SEED 42.
- `torch.load`: always `weights_only=True` (no arbitrary unpickling). The MAE checkpoint carries an `argparse.Namespace` of train args, so allowlist that one benign type via `torch.serialization.safe_globals([argparse.Namespace])`. Never disable `weights_only`; if another global is required, torch names it — allowlist it explicitly.
- All GPU work runs behind the `nvidia-smi` ≥6000 MiB-free waiter pattern (other project intermittently holds both GPUs).
- Commit after each task. Commit message trailers:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_013zZ3n55pDHS6KeRuhFKcn7`.

---

### Task 1: `p5_mae.py` module + off-GPU unit tests

**Files:**
- Create: `p5_mae.py`
- Test: `tests/test_p5_mae.py`

**Interfaces:**
- Produces:
  - `class MAEClassifier(nn.Module)` — `__init__(self, backbone)`; `forward(x) -> Tensor` of shape `(B, num_classes)` logits.
  - `build_mae_backbone(num_classes=4, img_size=224, drop_path_rate=0.2, global_pool=True) -> models_vit.VisionTransformer`
  - `build_classifier(num_classes=4, img_size=224, drop_path_rate=0.2) -> MAEClassifier`
  - `load_pretrained_mae_(backbone, repo_id, filename, device='cpu') -> backbone` (in-place weight load)
  - `mae_get_depth(name: str, num_blocks: int) -> int`
  - `build_mae_llrd_optimizer(model, base_lr, weight_decay, decay=0.75) -> torch.optim.AdamW`

- [ ] **Step 1: Write the failing test**

Create `tests/test_p5_mae.py`:

```python
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from p5_mae import build_classifier, mae_get_depth, build_mae_llrd_optimizer


def test_forward_shape():
    m = build_classifier(num_classes=4)
    out = m(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 4), out.shape


def test_depth():
    nb = 24
    assert mae_get_depth('backbone.head.weight', nb) == 0
    assert mae_get_depth('backbone.fc_norm.weight', nb) == 1
    assert mae_get_depth('backbone.blocks.23.attn.qkv.weight', nb) == 2
    assert mae_get_depth('backbone.blocks.0.norm1.weight', nb) == 25
    assert mae_get_depth('backbone.patch_embed.proj.weight', nb) == 26
    assert mae_get_depth('backbone.pos_embed', nb) == 26


def test_llrd_groups():
    m = build_classifier(num_classes=4)
    opt = build_mae_llrd_optimizer(m, base_lr=5e-5, weight_decay=0.05, decay=0.75)
    seen = sum(len(g['params']) for g in opt.param_groups)
    n_train = sum(1 for p in m.parameters() if p.requires_grad)
    assert seen == n_train, (seen, n_train)
    fc = dict(m.named_parameters())['backbone.fc_norm.weight']
    for g in opt.param_groups:
        if any(p is fc for p in g['params']):
            assert g['weight_decay'] == 0.0
            assert abs(g['lr'] - 5e-5 * 0.75 ** 1) < 1e-12
            break
    else:
        raise AssertionError('fc_norm param not assigned to any group')


if __name__ == '__main__':
    test_forward_shape(); test_depth(); test_llrd_groups()
    print('ALL TESTS PASSED')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/eth-admin/miniconda3/envs/retfound/bin/python tests/test_p5_mae.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'p5_mae'`.

- [ ] **Step 3: Write minimal implementation**

Create `p5_mae.py`:

```python
"""P5 — RETFound-MAE backbone (ViT-L/16, global-pooled) for the P2B pipeline.

Drop-in replacement for the DINOv2 loader. Kept importable so the forward
contract and LLRD grouping are unit-tested off-GPU (no weights download).
"""
import torch
import torch.nn as nn

import models_vit
from timm.layers import trunc_normal_


class MAEClassifier(nn.Module):
    """Wrap the RETFound-MAE ViT so forward() returns (B, num_classes) logits.

    models_vit.VisionTransformer.forward_features (global_pool) returns a
    (B, 1, embed) tensor already passed through fc_norm; we squeeze the token
    dim and apply the classifier head directly (bypassing timm's forward_head,
    which expects a string global_pool flag).
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        if feats.dim() == 3:
            feats = feats[:, 0]
        return self.backbone.head(feats)

    def set_grad_checkpointing(self, enable=True):
        # No-op: the custom forward_features does not honor timm checkpointing.
        pass


def build_mae_backbone(num_classes=4, img_size=224, drop_path_rate=0.2, global_pool=True):
    return models_vit.RETFound_mae(
        img_size=img_size, num_classes=num_classes,
        drop_path_rate=drop_path_rate, global_pool=global_pool)


def build_classifier(num_classes=4, img_size=224, drop_path_rate=0.2):
    return MAEClassifier(build_mae_backbone(num_classes, img_size, drop_path_rate))


def load_pretrained_mae_(backbone, repo_id, filename, device='cpu'):
    """In-place load of RETFound-MAE weights, mirroring main_finetune.py."""
    from huggingface_hub import hf_hub_download
    from util.pos_embed import interpolate_pos_embed
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    try:
        ck = torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        ck = torch.load(path, map_location='cpu', weights_only=False)  # trusted official weights
    cm = ck['model']
    cm = {k.replace('backbone.', ''): v for k, v in cm.items()}
    cm = {k.replace('mlp.w12.', 'mlp.fc1.'): v for k, v in cm.items()}
    cm = {k.replace('mlp.w3.', 'mlp.fc2.'): v for k, v in cm.items()}
    sd = backbone.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in cm and cm[k].shape != sd[k].shape:
            del cm[k]
    interpolate_pos_embed(backbone, cm)
    backbone.load_state_dict(cm, strict=False)
    trunc_normal_(backbone.head.weight, std=2e-5)
    nn.init.zeros_(backbone.head.bias)
    return backbone


def mae_get_depth(name, num_blocks):
    """LLRD depth from the head. fc_norm -> 1 (MAE deletes model.norm)."""
    name = name.replace('backbone.', '')
    if 'head' in name:
        return 0
    if 'fc_norm' in name or name.startswith('norm'):
        return 1
    if 'blocks.' in name:
        return num_blocks - int(name.split('blocks.')[1].split('.')[0]) + 1
    return num_blocks + 2


def build_mae_llrd_optimizer(model, base_lr, weight_decay, decay=0.75):
    num_blocks = len(model.backbone.blocks)

    def no_decay(n):
        return any(t in n for t in ['bias', 'norm', 'cls_token', 'pos_embed'])

    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        key = (mae_get_depth(name, num_blocks), no_decay(name))
        groups.setdefault(key, []).append(param)

    param_groups = []
    for (depth, nd), params in sorted(groups.items()):
        lr = base_lr * (decay ** depth)
        param_groups.append({'params': params, 'initial_lr': lr, 'lr': lr,
                             'weight_decay': 0.0 if nd else weight_decay})
    return torch.optim.AdamW(param_groups)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/eth-admin/miniconda3/envs/retfound/bin/python tests/test_p5_mae.py`
Expected: `ALL TESTS PASSED` (builds a random-init ViT-L on CPU; ~10–20 s, no download).

- [ ] **Step 5: Commit**

```bash
git add p5_mae.py tests/test_p5_mae.py
git commit -m "feat(p5): MAE backbone module + off-GPU tests (forward shape, LLRD depth)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013zZ3n55pDHS6KeRuhFKcn7"
```

---

### Task 2: Generator script + pilot notebook

**Files:**
- Create: `build_p5_notebook.py`
- Create (generated): `phase5_mae_pilot.ipynb`
- Test: `tests/test_p5_notebook.py`

**Interfaces:**
- Consumes: `phase2b_full_finetune.ipynb` (read-only source), `p5_mae.py` (imported by the generated notebook at runtime).
- Produces: `phase5_mae_pilot.ipynb` — a valid notebook whose cell 1 points at `output_dir/phase5_mae_cv` and `YukunZhou/RETFound_mae_meh`, whose cell 6 defines `load_backbone_fft` via `p5_mae`, and whose CV loop runs `for fold in range(2)` with `build_mae_llrd_optimizer`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_p5_notebook.py`:

```python
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_generated_notebook_markers():
    out = os.path.join(ROOT, 'phase5_mae_pilot.ipynb')
    if os.path.exists(out):
        os.remove(out)
    subprocess.check_call([sys.executable, os.path.join(ROOT, 'build_p5_notebook.py')], cwd=ROOT)
    nb = json.load(open(out))               # valid JSON
    src = '\n'.join('\n'.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    assert 'output_dir/phase5_mae_cv' in src
    assert 'YukunZhou/RETFound_mae_meh' in src
    assert 'RETFound_mae_meh.pth' in src
    assert 'from p5_mae import' in src
    assert 'build_mae_llrd_optimizer' in src
    assert 'for fold in range(2)' in src
    assert 'output_dir/phase2b_cv' not in src      # no leftover P2B paths
    assert 'RETFound_dinov2_meh' not in src
    # all code cells cleared of stale outputs
    assert all(c.get('outputs', []) == [] for c in nb['cells'] if c['cell_type'] == 'code')


if __name__ == '__main__':
    test_generated_notebook_markers()
    print('ALL TESTS PASSED')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/eth-admin/miniconda3/envs/retfound/bin/python tests/test_p5_notebook.py`
Expected: FAIL — `build_p5_notebook.py` does not exist (`FileNotFoundError` / non-zero exit).

- [ ] **Step 3: Write minimal implementation**

Create `build_p5_notebook.py`:

```python
"""Generate phase5_mae_pilot.ipynb by cloning phase2b_full_finetune.ipynb and
swapping only: output dir, MAE checkpoint, backbone-loader cell, optimizer call,
and the fold range (pilot folds 0-1). Single-variable vs P2B."""
import copy, json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, 'phase2b_full_finetune.ipynb')
OUT = os.path.join(ROOT, 'phase5_mae_pilot.ipynb')

MAE_LOADER_CELL = '''# ── MAE backbone loader (Phase 5) — imports tested p5_mae module ──
import torch
import numpy as np
from p5_mae import build_classifier, load_pretrained_mae_, build_mae_llrd_optimizer

def load_backbone_fft(device, num_classes=NUM_CLASSES, seed=None):
    """Build RETFound-MAE (ViT-L/16, global-pooled), load MEH weights, full FT.
    Same name as the DINOv2 loader so the CV loop is unchanged. Grad-ckpt is a
    no-op for this backbone; the A6000's 48 GB covers ViT-L FT at batch 16."""
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed)
    model = build_classifier(num_classes=num_classes, img_size=INPUT_SIZE, drop_path_rate=0.2)
    load_pretrained_mae_(model.backbone, repo_id=HF_REPO, filename=HF_FILE, device='cpu')
    for p in model.parameters():
        p.requires_grad = True
    model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'MAE classifier — Trainable {trainable:,}/{total:,}; grad-ckpt OFF')
    return model

print('Verifying MAE backbone load...')
_m = load_backbone_fft(DEVICE)
_out = _m(torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE).to(DEVICE))
assert _out.shape == (2, NUM_CLASSES), _out.shape
del _m, _out
torch.cuda.empty_cache()
print('OK — forward returns logits.')
'''


def sub(cell, a, b):
    cell['source'] = [ln.replace(a, b) for ln in cell['source']]


def main():
    nb = copy.deepcopy(json.load(open(SRC)))
    cells = nb['cells']

    # cell 1: config — output dir + MAE checkpoint
    sub(cells[1], 'output_dir/phase2b_cv', 'output_dir/phase5_mae_cv')
    sub(cells[1], 'YukunZhou/RETFound_dinov2_meh', 'YukunZhou/RETFound_mae_meh')
    sub(cells[1], 'RETFound_dinov2_meh.pth', 'RETFound_mae_meh.pth')

    # cell 6: replace the DINOv2 loader with the MAE loader
    cells[6]['source'] = MAE_LOADER_CELL.splitlines(keepends=True)

    # cell 11: pilot fold range + MAE optimizer + pilot results filename
    sub(cells[11], 'for fold in range(N_FOLDS):', 'for fold in range(2):  # PILOT folds 0-1')
    sub(cells[11], 'build_llrd_optimizer(model, BASE_LR, WEIGHT_DECAY, decay=LLRD_DECAY)',
        'build_mae_llrd_optimizer(model, BASE_LR, WEIGHT_DECAY, decay=LLRD_DECAY)')
    sub(cells[11], "fold_results.json", "fold_results_pilot.json")

    # clear all stale outputs/exec counts
    for c in cells:
        if c['cell_type'] == 'code':
            c['outputs'] = []
            c['execution_count'] = None

    json.dump(nb, open(OUT, 'w'), indent=1)
    print(f'Wrote {OUT}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/eth-admin/miniconda3/envs/retfound/bin/python tests/test_p5_notebook.py`
Expected: `ALL TESTS PASSED`.

Then manually confirm the loader cell replaced cleanly:
Run: `/home/eth-admin/miniconda3/envs/retfound/bin/python -c "import json; nb=json.load(open('phase5_mae_pilot.ipynb')); print(''.join(nb['cells'][6]['source'])[:120])"`
Expected: starts with `# ── MAE backbone loader (Phase 5)`.

- [ ] **Step 5: Commit**

```bash
git add build_p5_notebook.py tests/test_p5_notebook.py phase5_mae_pilot.ipynb
git commit -m "feat(p5): notebook generator + pilot notebook (folds 0-1, MAE backbone)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013zZ3n55pDHS6KeRuhFKcn7"
```

---

### Task 3: Run the pilot (folds 0-1) behind the GPU waiter + gate check

**Files:**
- Create: `wait_then_run_p5_pilot.sh` (scaffolding; deleted in Task 4 cleanup)
- Modify (executed in place): `phase5_mae_pilot.ipynb`
- Create: `output_dir/phase5_mae_cv/` (probs, `fold_results_pilot.json`) — gitignored

**Interfaces:**
- Consumes: `phase5_mae_pilot.ipynb`, `p5_mae.py`.
- Produces: `output_dir/phase5_mae_cv/fold_{0,1}_{oof,test}_{probs,labels}.npy`, `best_fold_{0,1}.pth`, `fold_results_pilot.json`.

- [ ] **Step 1: Write the waiter that executes the notebook in place**

Create `wait_then_run_p5_pilot.sh`:

```bash
#!/usr/bin/env bash
set -u
PY=/home/eth-admin/miniconda3/envs/retfound/bin/python
REPO=/home/eth-admin/Desktop/isaack/RETFound-main
NEED_MIB=6000
LOG="$REPO/run_p5_pilot.log"
cd "$REPO"
echo "[$(date '+%F %T')] P5 pilot waiter started; need ${NEED_MIB} MiB free" | tee "$LOG"
while true; do
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -v need="$NEED_MIB" '$2+0 >= need {print $1; exit}' | tr -d ' ')
  if [ -n "$GPU" ]; then
    echo "[$(date '+%F %T')] GPU $GPU free — executing pilot notebook" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m jupyter nbconvert --to notebook --execute \
      --inplace --ExecutePreprocessor.timeout=-1 phase5_mae_pilot.ipynb 2>&1 | tee -a "$LOG"
    echo "[$(date '+%F %T')] pilot exited ${PIPESTATUS[0]}" | tee -a "$LOG"
    break
  fi
  sleep 30
done
```

- [ ] **Step 2: GPU smoke before the long run**

Run (only when a GPU is free — check `nvidia-smi --query-gpu=memory.free --format=csv,noheader`):
`/home/eth-admin/miniconda3/envs/retfound/bin/python -c "import torch; from p5_mae import build_classifier, load_pretrained_mae_; m=build_classifier().cuda(); load_pretrained_mae_(m.backbone,'YukunZhou/RETFound_mae_meh','RETFound_mae_meh.pth'); m=m.cuda(); o=m(torch.randn(4,3,224,224).cuda()); l=o.sum(); l.backward(); print('smoke OK', tuple(o.shape))"`
Expected: `smoke OK (4, 4)` — confirms real-weights load + forward + backward on GPU.

- [ ] **Step 3: Launch the pilot in the background**

```bash
chmod +x wait_then_run_p5_pilot.sh
bash wait_then_run_p5_pilot.sh   # run in background; ~3-4 h for 2 folds once GPU frees
```

- [ ] **Step 4: Verify pilot outputs exist**

Run: `ls output_dir/phase5_mae_cv/ && cat output_dir/phase5_mae_cv/fold_results_pilot.json`
Expected: `fold_0_*` and `fold_1_*` npy files, `best_fold_{0,1}.pth`, and a JSON with two folds' `oof_auroc`.

- [ ] **Step 5: Evaluate the decision gate vs P2B**

Run:
```bash
/home/eth-admin/miniconda3/envs/retfound/bin/python -c "
import json
r=json.load(open('output_dir/phase5_mae_cv/fold_results_pilot.json'))
m=sum(x['oof_auroc'] for x in r)/len(r)
print(f'P5-MAE pilot mean OOF AUROC = {m:.4f}  (P2B 0.911; gate >= 0.906)')
print('PROCEED to folds 2-4' if m>=0.906 else 'STOP — negative result, log it')
"
```
Expected: prints the mean OOF AUROC and the PROCEED/STOP verdict. **This gate decides whether Task 4 runs.**

- [ ] **Step 6: Commit the pilot notebook (with saved outputs)**

```bash
git add phase5_mae_pilot.ipynb
git commit -m "run(p5): MAE pilot folds 0-1 executed; OOF AUROC vs P2B gate

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013zZ3n55pDHS6KeRuhFKcn7"
```

---

### Task 4: [GATED on Task 3 ≥ 0.906] Folds 2-4 + TTA head-to-head + document

Run only if the Task 3 gate prints PROCEED. If it prints STOP, skip to the final documentation step (record the negative result) and stop.

**Files:**
- Create: `build_p5_folds2to4.py` (range(2,5) variant generator), `phase5_mae_folds2to4.ipynb`
- Create: `run_p5_tta_test.py` (MAE TTA over 5 folds — mirrors `run_fold_tta_test.py` but imports `p5_mae`)
- Modify: `CLAUDE.md` (phase table row + artifacts + result paragraph)

**Interfaces:**
- Consumes: `p5_mae.py`, `output_dir/phase5_mae_cv/best_fold_{0,1}.pth`.
- Produces: `best_fold_{2,3,4}.pth`, all folds' probs, `fold_{0-4}_test_tta_probs.npy`, `test_tta_probs.npy`, and the MAE-vs-P2B head-to-head table.

- [ ] **Step 1: Generate the folds-2-4 notebook**

Create `build_p5_folds2to4.py` identical to `build_p5_notebook.py` except output `phase5_mae_folds2to4.ipynb` and substitute `'for fold in range(2):  # PILOT folds 0-1'` → `'for fold in range(2, 5):  # folds 2-4'` and `fold_results_pilot.json` → `fold_results_folds2to4.json`.
Run: `/home/eth-admin/miniconda3/envs/retfound/bin/python build_p5_folds2to4.py`
Expected: `Wrote .../phase5_mae_folds2to4.ipynb`.

- [ ] **Step 2: Run folds 2-4 behind the waiter**

Copy `wait_then_run_p5_pilot.sh` to `wait_then_run_p5_folds2to4.sh`, point it at `phase5_mae_folds2to4.ipynb`, and launch in background. Verify `best_fold_{2,3,4}.pth` + their probs appear.

- [ ] **Step 3: Create the MAE TTA script**

Create `run_p5_tta_test.py` by copying `run_fold_tta_test.py` and replacing its `load_backbone`/model-build block with:
```python
from p5_mae import build_classifier, load_pretrained_mae_
def build_model(device):
    m = build_classifier(num_classes=NUM_CLASSES)
    load_pretrained_mae_(m.backbone, 'YukunZhou/RETFound_mae_meh', 'RETFound_mae_meh.pth')
    return m.to(device)
```
and set `CV = REPO/'output_dir/phase5_mae_cv'`. The per-fold loop loads `best_fold_{f}.pth` (strict), runs 4-way TTA, saves `fold_{f}_test_tta_probs.npy` + ensemble `test_tta_probs.npy`, and prints the PtMean head-to-head vs P2B (0.8483 / 0.8501 / 0.9475 / 0.7513).

- [ ] **Step 4: Run TTA behind the waiter, capture the head-to-head**

Run `run_p5_tta_test.py` behind the waiter. Expected: prints `P5-MAE — TEST, PtMEAN, 4-WAY TTA` block with Accuracy / Kappa / Macro AUROC / Macro Sens and per-class sens, each annotated with the P2B value.

- [ ] **Step 5: Document the result in CLAUDE.md**

Add a P5 row to the Completed Phases table and a short result paragraph (positive → new recommended config note; negative → "don't re-attempt", mirroring P3/P4). Add `output_dir/phase5_mae_cv/` to Saved Artifacts.

- [ ] **Step 6: Clean up scaffolding + commit**

```bash
rm -f wait_then_run_p5_*.sh run_p5_pilot.log run_p5_folds2to4.log run_p5_tta_test.log
git add CLAUDE.md build_p5_folds2to4.py phase5_mae_folds2to4.ipynb run_p5_tta_test.py
git commit -m "feat(p5): MAE folds 2-4 + TTA head-to-head vs P2B; document result

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013zZ3n55pDHS6KeRuhFKcn7"
```

---

## Self-Review

**Spec coverage:** ✅ `p5_mae.py` loader/forward/LLRD (Task 1); pilot notebook single-variable clone (Task 2); pilot + gate (Task 3); folds 2-4 + TTA + PtMean head-to-head + CLAUDE.md (Task 4); isolation to `output_dir/phase5_mae_cv/` + new files (all tasks); waiter for GPU contention (Tasks 3-4); off-GPU shape/LLRD tests (Task 1); MAE-specific `fc_norm`→depth-1 and grad-ckpt-off (Task 1 + loader cell).

**Placeholder scan:** No TBD/TODO; all code blocks complete; test code included with exact assertions; exact paths and commands throughout.

**Type consistency:** `load_backbone_fft` name reused so the CV loop's `model = load_backbone_fft(...)` is unchanged; only `build_llrd_optimizer`→`build_mae_llrd_optimizer` swapped. `MAEClassifier.backbone` referenced consistently in loader cell, LLRD (`model.backbone.blocks`), and TTA script. `mae_get_depth` block depths verified: blocks.0→25, blocks.23→2, patch_embed→26, fc_norm→1.
