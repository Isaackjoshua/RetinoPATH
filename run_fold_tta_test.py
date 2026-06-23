"""P4 multi-task — 4-way TTA test inference over all 5 fold checkpoints.

Mirrors the P2D TTA recipe (CLAUDE.md): per image, average prob vectors over
{identity, hflip, vflip, hflip+vflip} of the fixed eval transform. Saves per-fold
TTA test probs, then the 5-fold ensemble, and prints the PtMean+TTA head-to-head
against the P2B recommended config (macro-sens 0.751 / R3A 0.571 / Kappa 0.850).
"""
import os, sys
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np, pandas as pd
import torch, torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import PIL.ImageFile
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import roc_auc_score, accuracy_score, cohen_kappa_score, confusion_matrix

REPO = Path('/home/eth-admin/Desktop/isaack/RETFound-main'); os.chdir(REPO); sys.path.insert(0, str(REPO))
NUM_CLASSES, INPUT_SIZE, BATCH = 4, 224, 16
CLASSES = ['R0', 'R1', 'R2', 'R3A']; GRADE = {'R0':0,'R1':1,'R2':2,'R3A':3}
HF_REPO, HF_FILE = 'YukunZhou/RETFound_dinov2_meh', 'RETFound_dinov2_meh.pth'
CV = REPO / 'output_dir/phase4_mt_cv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE} (CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES","unset")})')

import argparse
from util.datasets import build_transform
from p4_multitask import build_multitask_model
eval_tf = build_transform('val', argparse.Namespace(input_size=INPUT_SIZE, color_jitter=None,
            aa='rand-m9-mstd0.5-inc1', reprob=0.25, remode='pixel', recount=1))
TTA = [lambda im: eval_tf(im), lambda im: eval_tf(TF.hflip(im)),
       lambda im: eval_tf(TF.vflip(im)), lambda im: eval_tf(TF.vflip(TF.hflip(im)))]

df = pd.read_csv('labels/splits.csv'); df['grade_int'] = df['retinopathy'].map(GRADE)
df_test = df[df['split'] == 'test'].copy()
codes = df_test['code'].values; img_grade = df_test['grade_int'].values

class TTADataset(Dataset):
    def __init__(self, paths, grades): self.paths, self.grades = paths, grades
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert('RGB')
        return torch.stack([t(im) for t in TTA]), self.grades[i]   # (4,C,H,W)

loader = DataLoader(TTADataset(df_test['image_path'].values, img_grade),
                    batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

import timm
from huggingface_hub import hf_hub_download
def load_backbone(device):
    m = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True,
                          img_size=INPUT_SIZE, num_classes=NUM_CLASSES, drop_path_rate=0.2)
    ck = torch.load(hf_hub_download(repo_id=HF_REPO, filename=HF_FILE),
                    map_location='cpu', weights_only=True)['teacher']
    ck = {k.replace('backbone.', ''): v for k, v in ck.items()}
    ck = {k.replace('mlp.w12.', 'mlp.fc1.'): v for k, v in ck.items()}
    ck = {k.replace('mlp.w3.', 'mlp.fc2.'): v for k, v in ck.items()}
    st = m.state_dict()
    for k in [k for k in ck if k in st and ck[k].shape != st[k].shape]: del ck[k]
    m.load_state_dict(ck, strict=False); return m.to(device)

@torch.no_grad()
def tta_infer(model):
    model.eval(); out = []
    for stack, _ in loader:                       # stack: (B,4,C,H,W)
        B = stack.shape[0]
        x = stack.view(B * 4, *stack.shape[2:]).to(DEVICE)
        with torch.cuda.amp.autocast():
            g_logits, _ = model(x)
        p = torch.softmax(g_logits, dim=1).cpu().float().view(B, 4, NUM_CLASSES).mean(1)
        out.append(p)
    return torch.cat(out).numpy()

per_fold = []
for f in range(5):
    print(f'[fold {f}] loading + TTA inference...', flush=True)
    model = build_multitask_model(load_backbone(DEVICE)).to(DEVICE)
    model.load_state_dict(torch.load(CV / f'best_fold_{f}.pth', map_location=DEVICE,
                                     weights_only=True), strict=True)
    p = tta_infer(model)
    np.save(CV / f'fold_{f}_test_tta_probs.npy', p)
    per_fold.append(p)
    del model; torch.cuda.empty_cache()
    print(f'[fold {f}] done -> fold_{f}_test_tta_probs.npy', flush=True)

probs = np.mean(per_fold, axis=0); probs = probs / probs.sum(1, keepdims=True)
np.save(CV / 'test_tta_probs.npy', probs)

# PtMean pooling, patient label = worst grade across images
rows, plab = [], []
for c in pd.unique(codes):
    m = codes == c; rows.append(probs[m].mean(0)); plab.append(img_grade[m].max())
pp = np.array(rows); pp = pp / pp.sum(1, keepdims=True); plab = np.array(plab)
preds = pp.argmax(1)
au = roc_auc_score(plab, pp, multi_class='ovr', average='macro', labels=list(range(4)))
acc = accuracy_score(plab, preds); kap = cohen_kappa_score(plab, preds, weights='quadratic')
cm = confusion_matrix(plab, preds, labels=list(range(4)))
sens = [cm[i, i] / cm[i].sum() if cm[i].sum() else float('nan') for i in range(4)]
pca = [roc_auc_score((plab == i).astype(int), pp[:, i]) for i in range(4)]

print('\n' + '=' * 64)
print('P4 multi-task — TEST, 5-fold ensemble, PtMEAN, 4-WAY TTA, argmax')
print('=' * 64)
print(f'  Accuracy    : {acc:.4f}   (P2B 0.8483)')
print(f'  Kappa (quad): {kap:.4f}   (P2B 0.8501)')
print(f'  Macro AUROC : {au:.4f}   (P2B 0.9475)')
print(f'  Macro Sens  : {np.nanmean(sens):.4f}   (P2B 0.7513)')
print(f'  per-class sens : ' + ', '.join(f'{c} {s:.3f}' for c, s in zip(CLASSES, sens))
      + '   (P2B R0 .977 R1 .707 R2 .750 R3A .571)')
print(f'  per-class AUROC: ' + ', '.join(f'{c} {a:.3f}' for c, a in zip(CLASSES, pca)))
print('  confusion (rows=true):'); print(cm)
print('\nSaved per-fold TTA probs + test_tta_probs.npy. TTA head-to-head COMPLETE.')
