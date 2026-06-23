"""Standalone fold-4 evaluation for P4 multi-task CV.

The folds-2-4 run was interrupted right after fold-4 training: best_fold_4.pth
exists but its OOF/test probs were never saved. This script reloads that
checkpoint and reproduces exactly the per-fold save step from
phase4_mt_folds2to4.ipynb (cell 11) — non-TTA eval_fold_mt on the fold-4 val
split (OOF) and the test set — writing the four missing fold_4_*.npy files.

No retraining. Pure inference. Architecture/config copied verbatim from the nb.
"""
import os, sys, json, math
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import PIL.ImageFile
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, cohen_kappa_score, confusion_matrix

REPO = Path('/home/eth-admin/Desktop/isaack/RETFound-main')
os.chdir(REPO)
sys.path.insert(0, str(REPO))

# ── Config (verbatim from phase4_mt_folds2to4.ipynb) ──────────────────────────
N_FOLDS, NUM_CLASSES, INPUT_SIZE, SEED = 5, 4, 224, 42
FOLD = 4
CLASSES = ['R0', 'R1', 'R2', 'R3A']
HF_REPO, HF_FILE = 'YukunZhou/RETFound_dinov2_meh', 'RETFound_dinov2_meh.pth'
CV_OUTPUT = REPO / 'output_dir/phase4_mt_cv'
BATCH_SIZE = 16

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if DEVICE.type == 'cuda':
    p = torch.cuda.get_device_properties(DEVICE)
    print(f'GPU: {p.name}  VRAM {p.total_memory/1e9:.1f} GB  '
          f'(CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES","unset")})')

# ── Splits + fold assignment (identical SEED=42 → identical folds) ────────────
GRADE = {'R0': 0, 'R1': 1, 'R2': 2, 'R3A': 3}
df_all = pd.read_csv('labels/splits.csv')
df_all['grade_int'] = df_all['retinopathy'].map(GRADE)
df_cv   = df_all[df_all['split'].isin(['train', 'val'])].copy()
df_test = df_all[df_all['split'] == 'test'].copy()

pat_grade = df_cv.groupby('code')['grade_int'].max().reset_index()
pat_grade.columns = ['code', 'strat_grade']
patient_ids, patient_strat = pat_grade['code'].values, pat_grade['strat_grade'].values
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_assignments = {}
for fi, (_, val_idx) in enumerate(skf.split(patient_ids, patient_strat)):
    for pid in patient_ids[val_idx]:
        fold_assignments[pid] = fi
pat_grade['fold'] = pat_grade['code'].map(fold_assignments)
df_cv = df_cv.reset_index(drop=True)
df_cv['cv_idx'] = df_cv.index

val_pats = pat_grade[pat_grade['fold'] == FOLD]['code'].values
df_fold_val = df_cv[df_cv['code'].isin(val_pats)]
print(f'Fold {FOLD}: {len(val_pats)} val patients | {len(df_fold_val)} val images | '
      f'{len(df_test)} test images')

# ── Transforms + dataset ──────────────────────────────────────────────────────
import argparse
from util.datasets import build_transform
from p4_multitask import (P4Dataset, make_records_mt, build_multitask_model)

_aug = argparse.Namespace(input_size=INPUT_SIZE, color_jitter=None,
                          aa='rand-m9-mstd0.5-inc1', reprob=0.25, remode='pixel', recount=1)
eval_transform = build_transform('val', _aug)

ds_val  = P4Dataset(make_records_mt(df_fold_val), eval_transform)
ds_test = P4Dataset(make_records_mt(df_test),     eval_transform)
loader_val  = DataLoader(ds_val,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
loader_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ── Rebuild architecture, then load the trained fold-4 checkpoint ─────────────
import timm
from huggingface_hub import hf_hub_download

def load_backbone(device):
    model = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True,
                              img_size=INPUT_SIZE, num_classes=NUM_CLASSES, drop_path_rate=0.2)
    ckpt = torch.load(hf_hub_download(repo_id=HF_REPO, filename=HF_FILE),
                      map_location='cpu', weights_only=True)['teacher']
    ckpt = {k.replace('backbone.', ''): v for k, v in ckpt.items()}
    ckpt = {k.replace('mlp.w12.', 'mlp.fc1.'): v for k, v in ckpt.items()}
    ckpt = {k.replace('mlp.w3.', 'mlp.fc2.'): v for k, v in ckpt.items()}
    state = model.state_dict()
    for k in [k for k in ckpt if k in state and ckpt[k].shape != state[k].shape]:
        del ckpt[k]
    model.load_state_dict(ckpt, strict=False)
    return model.to(device)

model = build_multitask_model(load_backbone(DEVICE)).to(DEVICE)
ckpt_path = CV_OUTPUT / f'best_fold_{FOLD}.pth'
state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
missing, unexpected = model.load_state_dict(state, strict=True)
print(f'Loaded {ckpt_path.name}  (strict load OK)')

@torch.no_grad()
def eval_fold_mt(model, loader, device):
    model.eval(); labels, probs = [], []
    for imgs, grades, feats in loader:
        with torch.cuda.amp.autocast():
            g_logits, _ = model(imgs.to(device))
        probs.append(torch.softmax(g_logits, dim=1).cpu().float()); labels.append(grades)
    return torch.cat(labels).numpy(), torch.cat(probs).numpy()

def metrics(labels, probs):
    preds = probs.argmax(1)
    pf = probs.astype(np.float64); pf = pf / pf.sum(1, keepdims=True)
    try:
        auroc = roc_auc_score(labels, pf, multi_class='ovr', average='macro',
                              labels=list(range(NUM_CLASSES)))
    except Exception:
        auroc = float('nan')
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    sens = [cm[i, i] / cm[i].sum() if cm[i].sum() else float('nan') for i in range(NUM_CLASSES)]
    return (auroc, accuracy_score(labels, preds),
            cohen_kappa_score(labels, preds, weights='quadratic'),
            float(np.nanmean(sens)), sens)

# ── Run + save (same filenames/shape as folds 0-3) ────────────────────────────
print('Evaluating fold-4 val (OOF)...')
oof_labels, oof_probs = eval_fold_mt(model, loader_val, DEVICE)
print('Evaluating test...')
test_labels, test_probs = eval_fold_mt(model, loader_test, DEVICE)

np.save(CV_OUTPUT / f'fold_{FOLD}_oof_probs.npy',   oof_probs)
np.save(CV_OUTPUT / f'fold_{FOLD}_oof_labels.npy',  oof_labels)
np.save(CV_OUTPUT / f'fold_{FOLD}_test_probs.npy',  test_probs)
np.save(CV_OUTPUT / f'fold_{FOLD}_test_labels.npy', test_labels)

au, acc, kap, ms, sens = metrics(oof_labels, oof_probs)
print(f'\nFOLD {FOLD} OOF: AUROC={au:.4f}  acc={acc:.4f}  kappa={kap:.4f}  macroSens={ms:.4f}')
print('  per-class sens:', {c: round(s, 4) for c, s in zip(CLASSES, sens)})

# Append into a folds-2-4 results json (don't clobber the pilot file)
res_path = CV_OUTPUT / 'fold_results_folds2to4.json'
existing = json.loads(res_path.read_text()) if res_path.exists() else []
existing = [r for r in existing if r.get('fold') != FOLD]
existing.append({'fold': FOLD, 'oof_auroc': au, 'oof_macro_sens': ms,
                 'oof_acc': acc, 'oof_kappa': kap})
res_path.write_text(json.dumps(sorted(existing, key=lambda r: r['fold']), indent=2))
print(f'Saved probs (4 files) + {res_path.name}. Fold 4 eval COMPLETE.')
