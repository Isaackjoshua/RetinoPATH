"""Exp03 resume — finish the interrupted 5-fold run cheaply.

The full re-run died after fold 3's checkpoint was saved but before its eval
probs were written, and fold 4 never started. Folds 0-2 are fully on disk.
This script:
  1. fold 3 — load best_fold_3.pth (NO retrain) -> eval val/test -> save probs
  2. fold 4 — train from scratch (identical recipe to the notebook) -> save
  3. aggregate OOF across all 5 folds -> oof_probs_all/labels_all
  4. test ensemble + Youden + summary + confusion-matrix figures (notebook cells 13/15)

All training config/helpers are copied verbatim from
Exp03_bestmodel_hybrid_class_balancing.ipynb so fold 4 matches folds 0-3.
After this, run exp03_tta_eval.py for the PtMean+TTA comparison vs P2B.
"""
import os, sys, json, math, time, argparse
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import PIL.ImageFile
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, cohen_kappa_score,
                             confusion_matrix, roc_curve)

# ── Config (verbatim from notebook cell 1) ────────────────────────────────────
N_FOLDS, MAX_EPOCHS, PATIENCE = 5, 50, 10
INPUT_SIZE, NUM_CLASSES = 224, 4
SEED = 42
CLASSES = ['R0', 'R1', 'R2', 'R3A']
BASE_LR, MIN_LR = 5e-5, 1e-7
WARMUP_EPOCHS, WEIGHT_DECAY, LLRD_DECAY = 5, 0.05, 0.75
GRAD_CLIP, BATCH_SIZE, ACCUM_STEPS = 1.0, 16, 2
FOCAL_GAMMA = 2.0
TARGET_PER_CLASS = 1000
HF_REPO, HF_FILE = 'YukunZhou/RETFound_dinov2_meh', 'RETFound_dinov2_meh.pth'
CV_OUTPUT = Path('output_dir/exp03_bestmodel_hybrid_class_balancing_cv')
CV_OUTPUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRADE = {'R0': 0, 'R1': 1, 'R2': 2, 'R3A': 3}
print(f'Device: {DEVICE}  ({torch.cuda.get_device_name(0) if DEVICE.type=="cuda" else "cpu"})')


# ── FocalLoss (cell 2) ────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__(); self.gamma = gamma; self.weight = weight
    def forward(self, logits, targets):
        log_p = F.log_softmax(logits, dim=1)
        log_pt = log_p.gather(1, targets.view(-1, 1)).squeeze(1)
        pt = log_pt.exp()
        fw = (1.0 - pt) ** self.gamma
        if self.weight is not None:
            alpha = self.weight[targets]; fw = fw * alpha
            return -(fw * log_pt).sum() / alpha.sum()
        return -(fw * log_pt).mean()


# ── Splits / folds (cell 3) ───────────────────────────────────────────────────
df_all = pd.read_csv('labels/splits.csv')
df_all['grade_int'] = df_all['retinopathy'].map(GRADE)
df_cv = df_all[df_all['split'].isin(['train', 'val'])].copy()
df_test = df_all[df_all['split'] == 'test'].copy()
pat_grade = df_cv.groupby('code')['grade_int'].max().reset_index()
pat_grade.columns = ['code', 'strat_grade']
patient_ids, patient_strat = pat_grade['code'].values, pat_grade['strat_grade'].values
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_assignments = {}
for fi, (_, vidx) in enumerate(skf.split(patient_ids, patient_strat)):
    for pid in patient_ids[vidx]:
        fold_assignments[pid] = fi
pat_grade['fold'] = pat_grade['code'].map(fold_assignments)
df_cv = df_cv.reset_index(drop=True)
df_cv['cv_idx'] = df_cv.index
print(f'CV pool: {len(df_cv)} images | {len(patient_ids)} patients | Test: {len(df_test)} images')


# ── Transforms / dataset (cell 4) ─────────────────────────────────────────────
from util.datasets import build_transform
_aug_args = argparse.Namespace(input_size=INPUT_SIZE, color_jitter=None,
                               aa='rand-m9-mstd0.5-inc1', reprob=0.25, remode='pixel', recount=1)
train_transform = build_transform('train', _aug_args)
eval_transform = build_transform('val', _aug_args)


class RetinopathyDataset(Dataset):
    def __init__(self, records, transform): self.records = records; self.transform = transform
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        path, label = self.records[idx]
        return self.transform(Image.open(path).convert('RGB')), label


def make_records(df_subset):
    return [(row.image_path, row.grade_int) for row in df_subset.itertuples()]


# ── Backbone (cell 6) ─────────────────────────────────────────────────────────
import timm
from huggingface_hub import hf_hub_download
from timm.layers import trunc_normal_


def load_backbone_fft(device, num_classes=NUM_CLASSES, seed=None, pretrained=True):
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed)
    model = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=pretrained,
                              img_size=INPUT_SIZE, num_classes=num_classes, drop_path_rate=0.2)
    if pretrained:
        ckpt_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        cm = ckpt['teacher']
        cm = {k.replace('backbone.', ''): v for k, v in cm.items()}
        cm = {k.replace('mlp.w12.', 'mlp.fc1.'): v for k, v in cm.items()}
        cm = {k.replace('mlp.w3.', 'mlp.fc2.'): v for k, v in cm.items()}
        state = model.state_dict()
        for k in [k for k in cm if k in state and cm[k].shape != state[k].shape]:
            del cm[k]
        model.load_state_dict(cm, strict=False)
        trunc_normal_(model.head.weight, std=2e-5)
        nn.init.zeros_(model.head.bias)
    for p in model.parameters():
        p.requires_grad = True
    model.set_grad_checkpointing(enable=True)
    return model.to(device)


# ── LLRD optimizer (cell 8) ───────────────────────────────────────────────────
def build_llrd_optimizer(model, base_lr, weight_decay, decay=LLRD_DECAY):
    num_blocks = len(model.blocks)
    def get_depth(name):
        if 'head' in name: return 0
        if name.startswith('norm'): return 1
        if 'blocks.' in name:
            return num_blocks - int(name.split('blocks.')[1].split('.')[0]) + 1
        return num_blocks + 2
    def no_decay(name):
        return any(t in name for t in ['bias', 'norm', 'cls_token', 'pos_embed'])
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        groups.setdefault((get_depth(name), no_decay(name)), []).append(param)
    pgs = []
    for (depth, nd), params in sorted(groups.items()):
        lr = base_lr * (decay ** depth)
        pgs.append({'params': params, 'initial_lr': lr, 'lr': lr,
                    'weight_decay': 0.0 if nd else weight_decay})
    return torch.optim.AdamW(pgs)


# ── Training helpers (cell 9) ─────────────────────────────────────────────────
class EarlyStoppingFFT:
    def __init__(self, patience, model, checkpoint_path):
        self.patience = patience; self.best_auroc = -float('inf'); self.counter = 0
        self.checkpoint_path = Path(checkpoint_path)
        torch.save(model.state_dict(), self.checkpoint_path)
    def step(self, auroc, model):
        if auroc != auroc: auroc = 0.0
        if auroc > self.best_auroc:
            self.best_auroc = auroc; self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path); return False
        self.counter += 1
        return self.counter >= self.patience
    def restore(self, model, device):
        model.load_state_dict(torch.load(self.checkpoint_path, map_location=device, weights_only=True))


def get_lr(epoch, warmup, max_ep, base_lr, min_lr):
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    t = (epoch - warmup) / max(1, max_ep - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


def train_epoch_fft(model, loader, optimizer, criterion, device, scaler, epoch):
    model.train()
    head_lr = get_lr(epoch, WARMUP_EPOCHS, MAX_EPOCHS, BASE_LR, MIN_LR)
    lr_scale = head_lr / BASE_LR
    for pg in optimizer.param_groups:
        pg['lr'] = pg['initial_lr'] * lr_scale
    optimizer.zero_grad()
    total_loss, n_samples, step_count = 0.0, 0, 0
    for i, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        is_last = (i + 1 == len(loader))
        should_step = ((step_count + 1) % ACCUM_STEPS == 0) or is_last
        with torch.cuda.amp.autocast():
            loss = criterion(model(imgs), labels) / ACCUM_STEPS
        scaler.scale(loss).backward()
        total_loss += loss.item() * ACCUM_STEPS * len(labels)
        n_samples += len(labels); step_count += 1
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(); step_count = 0
    return total_loss / n_samples, head_lr


@torch.no_grad()
def eval_fold(model, loader, device):
    model.eval(); all_labels, all_probs = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        with torch.cuda.amp.autocast():
            logits = model(imgs)
        all_labels.append(labels)
        all_probs.append(torch.softmax(logits, dim=1).cpu().float())
    return torch.cat(all_labels).numpy(), torch.cat(all_probs).numpy()


def compute_metrics(labels, probs, preds=None):
    if preds is None: preds = probs.argmax(axis=1)
    pf = probs.astype(np.float64); pf = pf / pf.sum(axis=1, keepdims=True)
    try:
        auroc = roc_auc_score(labels, pf, multi_class='ovr', average='macro',
                              labels=list(range(NUM_CLASSES)))
    except Exception:
        auroc = float('nan')
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    sens, spec = [], []
    for i in range(NUM_CLASSES):
        tp = cm[i, i]; fn = cm[i, :].sum() - tp; fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens.append(tp / (tp + fn) if (tp + fn) > 0 else float('nan'))
        spec.append(tn / (tn + fp) if (tn + fp) > 0 else float('nan'))
    return {'auroc': auroc, 'accuracy': accuracy_score(labels, preds),
            'kappa': cohen_kappa_score(labels, preds, weights='quadratic'),
            'macro_sensitivity': np.nanmean(sens), 'macro_specificity': np.nanmean(spec),
            'sensitivity': np.array(sens), 'specificity': np.array(spec)}


criterion_cv = FocalLoss(gamma=FOCAL_GAMMA, weight=None)
df_test_records = make_records(df_test)


def eval_only_fold(fold):
    """Load existing best_fold_{fold}.pth and write its OOF + test probs."""
    print(f'\n=== fold {fold}: EVAL-ONLY from existing checkpoint ===')
    val_pats = pat_grade[pat_grade['fold'] == fold]['code'].values
    df_fold_val = df_cv[df_cv['code'].isin(val_pats)]
    loader_val = DataLoader(RetinopathyDataset(make_records(df_fold_val), eval_transform),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True)
    loader_test = DataLoader(RetinopathyDataset(df_test_records, eval_transform),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True)
    model = load_backbone_fft(DEVICE, pretrained=False)
    model.load_state_dict(torch.load(CV_OUTPUT / f'best_fold_{fold}.pth',
                                     map_location=DEVICE, weights_only=True))
    oof_labels, oof_probs = eval_fold(model, loader_val, DEVICE)
    test_labels, test_probs = eval_fold(model, loader_test, DEVICE)
    np.save(CV_OUTPUT / f'fold_{fold}_oof_probs.npy', oof_probs)
    np.save(CV_OUTPUT / f'fold_{fold}_oof_labels.npy', oof_labels)
    np.save(CV_OUTPUT / f'fold_{fold}_test_probs.npy', test_probs)
    np.save(CV_OUTPUT / f'fold_{fold}_test_labels.npy', test_labels)
    m = compute_metrics(oof_labels, oof_probs)
    print(f'  OOF AUROC {m["auroc"]:.4f} Kappa {m["kappa"]:.4f} mSens {m["macro_sensitivity"]:.4f}')
    del model; torch.cuda.empty_cache()


def train_fold(fold):
    """Train fold from scratch (identical to notebook cell 11)."""
    print(f'\n=== fold {fold}: TRAIN from scratch ===')
    val_pats = pat_grade[pat_grade['fold'] == fold]['code'].values
    train_pats = pat_grade[pat_grade['fold'] != fold]['code'].values
    df_fold_train = df_cv[df_cv['code'].isin(train_pats)]
    df_fold_val = df_cv[df_cv['code'].isin(val_pats)]
    print(f'  Train: {len(df_fold_train)} images | Val: {len(df_fold_val)} images')
    ds_train = RetinopathyDataset(make_records(df_fold_train), train_transform)
    ds_val = RetinopathyDataset(make_records(df_fold_val), eval_transform)
    ds_test = RetinopathyDataset(df_test_records, eval_transform)
    tr_labels = df_fold_train['grade_int'].values
    class_counts = np.bincount(tr_labels, minlength=NUM_CLASSES)
    sample_w = 1.0 / class_counts[tr_labels]
    g_sampler = torch.Generator().manual_seed(SEED + fold)
    sampler = WeightedRandomSampler(weights=torch.as_tensor(sample_w, dtype=torch.double),
                                    num_samples=TARGET_PER_CLASS * NUM_CLASSES,
                                    replacement=True, generator=g_sampler)
    print(f'  Balanced stream: ~{TARGET_PER_CLASS}/class (orig {class_counts.tolist()})')
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=12, pin_memory=True, drop_last=False)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=12, pin_memory=True)
    loader_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=12, pin_memory=True)
    model = load_backbone_fft(DEVICE, seed=SEED + fold)
    optimizer = build_llrd_optimizer(model, BASE_LR, WEIGHT_DECAY, decay=LLRD_DECAY)
    scaler = torch.cuda.amp.GradScaler()
    stopper = EarlyStoppingFFT(PATIENCE, model, CV_OUTPUT / f'best_fold_{fold}.pth')
    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        tr_loss, cur_lr = train_epoch_fft(model, loader_train, optimizer, criterion_cv,
                                          DEVICE, scaler, epoch)
        vl, vp = eval_fold(model, loader_val, DEVICE)
        m = compute_metrics(vl, vp)
        print(f'  ep {epoch:02d} | lr={cur_lr:.2e} | loss={tr_loss:.4f} | '
              f'AUROC={m["auroc"]:.4f} | sens={m["macro_sensitivity"]:.4f} | '
              f'{time.time()-t0:.0f}s')
        if stopper.step(m['auroc'], model):
            print(f'  Early stop ep {epoch} (best AUROC={stopper.best_auroc:.4f})'); break
    stopper.restore(model, DEVICE)
    print(f'  Best val AUROC: {stopper.best_auroc:.4f}')
    oof_labels, oof_probs = eval_fold(model, loader_val, DEVICE)
    test_labels, test_probs = eval_fold(model, loader_test, DEVICE)
    np.save(CV_OUTPUT / f'fold_{fold}_oof_probs.npy', oof_probs)
    np.save(CV_OUTPUT / f'fold_{fold}_oof_labels.npy', oof_labels)
    np.save(CV_OUTPUT / f'fold_{fold}_test_probs.npy', test_probs)
    np.save(CV_OUTPUT / f'fold_{fold}_test_labels.npy', test_labels)
    m = compute_metrics(oof_labels, oof_probs)
    print(f'  OOF AUROC {m["auroc"]:.4f} Kappa {m["kappa"]:.4f} mSens {m["macro_sensitivity"]:.4f}')
    del model; torch.cuda.empty_cache()
    return stopper.best_auroc


# ════════════════════════════════════════════════════════════════════════════
# 1) fold 3 eval-only, 2) fold 4 train
# ════════════════════════════════════════════════════════════════════════════
eval_only_fold(3)
train_fold(4)

# ── 3) aggregate OOF across all 5 folds (rebuild from per-fold files) ──────────
print('\n=== Aggregating OOF across 5 folds ===')
oof_labels_all = np.zeros(len(df_cv), dtype=np.int64)
oof_probs_all = np.zeros((len(df_cv), NUM_CLASSES), dtype=np.float32)
for fold in range(N_FOLDS):
    val_pats = pat_grade[pat_grade['fold'] == fold]['code'].values
    val_cv_indices = df_cv[df_cv['code'].isin(val_pats)]['cv_idx'].values
    oof_probs_all[val_cv_indices] = np.load(CV_OUTPUT / f'fold_{fold}_oof_probs.npy')
    oof_labels_all[val_cv_indices] = np.load(CV_OUTPUT / f'fold_{fold}_oof_labels.npy')
np.save(CV_OUTPUT / 'oof_labels_all.npy', oof_labels_all)
np.save(CV_OUTPUT / 'oof_probs_all.npy', oof_probs_all)
m_oof = compute_metrics(oof_labels_all, oof_probs_all)
print(f'  Aggregate OOF: AUROC {m_oof["auroc"]:.4f} Kappa {m_oof["kappa"]:.4f} '
      f'mSens {m_oof["macro_sensitivity"]:.4f}')

# ── 4) test ensemble + Youden + summary + figures (cells 13/15) ───────────────
print('\n=== Test ensemble + summary (cells 13/15) ===')
test_probs_list = [np.load(CV_OUTPUT / f'fold_{f}_test_probs.npy').astype(np.float64)
                   for f in range(N_FOLDS)]
test_labels_all = np.load(CV_OUTPUT / 'fold_0_test_labels.npy')
test_probs_ens = np.mean(test_probs_list, axis=0)
test_probs_ens = test_probs_ens / test_probs_ens.sum(axis=1, keepdims=True)
np.save(CV_OUTPUT / 'test_ensemble_probs.npy', test_probs_ens)
np.save(CV_OUTPUT / 'test_ensemble_labels.npy', test_labels_all)

oof_prb64 = oof_probs_all.astype(np.float64)
oof_prb64 = oof_prb64 / oof_prb64.sum(axis=1, keepdims=True)
youden_thr = []
for i in range(NUM_CLASSES):
    fpr, tpr, thrs = roc_curve((oof_labels_all == i).astype(int), oof_prb64[:, i])
    youden_thr.append(float(thrs[(tpr - fpr).argmax()]))
print(f'  Youden thresholds: {[f"{t:.4f}" for t in youden_thr]}')


def apply_thresholds(probs, thresholds):
    thresholds = np.array(thresholds)
    return np.where((probs > thresholds).any(axis=1),
                    (probs / thresholds).argmax(axis=1), probs.argmax(axis=1))


def _norm(p):
    p = p.astype(np.float64); return p / p.sum(axis=1, keepdims=True)


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
EXP, EXP_LABEL = 'Exp03_bestmodel_hybrid_class_balancing', 'Exp03 Best+Bal'
RECIPE = ('best-model (focal γ=2, LLRD 0.75) + hybrid under+over sampling to '
          '~1000/class, NO class weights')
FIG_DIR = Path('figures'); FIG_DIR.mkdir(exist_ok=True)


def show_confusion(labels, probs, title, fname):
    cm = confusion_matrix(labels, probs.argmax(1), labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASSES)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title)
    thr_c = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, int(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thr_c else 'black')
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(FIG_DIR / fname, dpi=120); plt.close(fig)
    print(f'  saved {FIG_DIR / fname}')
    return cm


oof_prb = _norm(oof_probs_all)
tst_prb = _norm(test_probs_ens)
cm_oof = show_confusion(oof_labels_all, oof_prb, f'{EXP_LABEL} OOF', f'{EXP}_cm_oof.png')
cm_test = show_confusion(test_labels_all, tst_prb, f'{EXP_LABEL} TEST', f'{EXP}_cm_test.png')


def _pack(lbl, prb, thr=None):
    preds = apply_thresholds(prb, thr) if thr is not None else None
    m = compute_metrics(lbl, prb, preds)
    return {'accuracy': m['accuracy'], 'auroc': m['auroc'], 'kappa': m['kappa'],
            'macro_sensitivity': m['macro_sensitivity'], 'macro_specificity': m['macro_specificity'],
            'sensitivity': m['sensitivity'].tolist(), 'specificity': m['specificity'].tolist()}


summary = {'experiment': EXP, 'recipe': RECIPE, 'classes': CLASSES, 'base_lr': BASE_LR,
           'youden_thresholds': {c: youden_thr[i] for i, c in enumerate(CLASSES)},
           'oof': {'Argmax': _pack(oof_labels_all, oof_prb),
                   'Youden': _pack(oof_labels_all, oof_prb, youden_thr)},
           'test': {'Argmax': _pack(test_labels_all, tst_prb),
                    'Youden': _pack(test_labels_all, tst_prb, youden_thr)},
           'confusion_matrix': {'oof': cm_oof.tolist(), 'test': cm_test.tolist()}}
with open(CV_OUTPUT / 'exp03_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=float)
print(f'\nSaved {CV_OUTPUT}/exp03_summary.json')
print('TEST Argmax:', {k: round(v, 4) for k, v in summary['test']['Argmax'].items()
                       if k in ('accuracy', 'auroc', 'kappa', 'macro_sensitivity')})
print('\nDONE. Next: python exp03_tta_eval.py')
