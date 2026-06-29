"""Model B — Maculopathy (M0/M1), binary single-task.

Clones the proven P2B recipe (RETFound-DINOv2-MEH full fine-tune, LLRD 0.75,
grad-checkpointing, batch 16×accum 2, cosine+warmup, 5-fold StratifiedKFold seed=42)
with single-variable adaptations for the binary maculopathy task:
  • head → 2 classes; stratify folds on patient-worst maculopathy
  • FocalLoss γ=2, NO class weights, standard shuffled sampling (M1 ~30%, near-balanced)
  • clinical metric is already binary (M1 = referable maculopathy); the M1 operating
    point (sens≥85% / spec≥95%) is chosen on OOF and reported on test — no test peeking.

Usage:
  python modelB_maculopathy.py --folds 0 1      # PILOT (gate on pooled OOF AUROC)
  python modelB_maculopathy.py --folds 2 3 4    # finish remaining folds
  python modelB_maculopathy.py --tta            # 4-way TTA + aggregate + operating point
"""
import os, sys, json, math, time, argparse
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from PIL import Image
import PIL.ImageFile
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

# ── Config (P2B recipe; only NUM_CLASSES + task differ) ───────────────────────
N_FOLDS, MAX_EPOCHS, PATIENCE = 5, 50, 10
INPUT_SIZE, NUM_CLASSES = 224, 2
SEED = 42
CLASSES = ['M0', 'M1']
LABELMAP = {'M0': 0, 'M1': 1}
BASE_LR, MIN_LR = 5e-5, 1e-7
WARMUP_EPOCHS, WEIGHT_DECAY, LLRD_DECAY = 5, 0.05, 0.75
GRAD_CLIP, BATCH_SIZE, ACCUM_STEPS = 1.0, 16, 2
FOCAL_GAMMA = 2.0
HF_REPO, HF_FILE = 'YukunZhou/RETFound_dinov2_meh', 'RETFound_dinov2_meh.pth'
CV = Path('output_dir/modelB_maculopathy_cv'); CV.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__(); self.gamma = gamma; self.weight = weight
    def forward(self, logits, targets):
        log_p = F.log_softmax(logits, dim=1)
        log_pt = log_p.gather(1, targets.view(-1, 1)).squeeze(1)
        pt = log_pt.exp(); fw = (1.0 - pt) ** self.gamma
        if self.weight is not None:
            alpha = self.weight[targets]; fw = fw * alpha
            return -(fw * log_pt).sum() / alpha.sum()
        return -(fw * log_pt).mean()


# ── Splits / folds (stratify on patient-worst maculopathy) ────────────────────
df_all = pd.read_csv('labels/splits.csv')
df_all['grade_int'] = df_all['maculopathy'].map(LABELMAP)
df_cv = df_all[df_all['split'].isin(['train', 'val'])].copy()
df_test = df_all[df_all['split'] == 'test'].copy()
pat = df_cv.groupby('code')['grade_int'].max().reset_index()
pat.columns = ['code', 'strat']
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fa = {}
for fi, (_, vidx) in enumerate(skf.split(pat['code'].values, pat['strat'].values)):
    for pid in pat['code'].values[vidx]:
        fa[pid] = fi
pat['fold'] = pat['code'].map(fa)
df_cv = df_cv.reset_index(drop=True); df_cv['cv_idx'] = df_cv.index

from util.datasets import build_transform
_aug = argparse.Namespace(input_size=INPUT_SIZE, color_jitter=None,
                          aa='rand-m9-mstd0.5-inc1', reprob=0.25, remode='pixel', recount=1)
train_tf = build_transform('train', _aug)
eval_tf = build_transform('val', _aug)


class DS(Dataset):
    def __init__(s, recs, tf): s.r = recs; s.tf = tf
    def __len__(s): return len(s.r)
    def __getitem__(s, i):
        p, l = s.r[i]; return s.tf(Image.open(p).convert('RGB')), l


def recs(d): return [(r.image_path, r.grade_int) for r in d.itertuples()]


import timm
from huggingface_hub import hf_hub_download
from timm.layers import trunc_normal_


def load_backbone(device, seed=None, pretrained=True):
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed)
    m = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=pretrained,
                          img_size=INPUT_SIZE, num_classes=NUM_CLASSES, drop_path_rate=0.2)
    if pretrained:
        ck = torch.load(hf_hub_download(repo_id=HF_REPO, filename=HF_FILE),
                        map_location='cpu', weights_only=True)['teacher']
        ck = {k.replace('backbone.', ''): v for k, v in ck.items()}
        ck = {k.replace('mlp.w12.', 'mlp.fc1.'): v for k, v in ck.items()}
        ck = {k.replace('mlp.w3.', 'mlp.fc2.'): v for k, v in ck.items()}
        st = m.state_dict()
        for k in [k for k in ck if k in st and ck[k].shape != st[k].shape]:
            del ck[k]
        m.load_state_dict(ck, strict=False)
        trunc_normal_(m.head.weight, std=2e-5); nn.init.zeros_(m.head.bias)
    for p in m.parameters(): p.requires_grad = True
    m.set_grad_checkpointing(enable=True)
    return m.to(device)


def llrd_opt(model, base_lr, wd, decay=LLRD_DECAY):
    nb = len(model.blocks)
    def depth(n):
        if 'head' in n: return 0
        if n.startswith('norm'): return 1
        if 'blocks.' in n: return nb - int(n.split('blocks.')[1].split('.')[0]) + 1
        return nb + 2
    def nd(n): return any(t in n for t in ['bias', 'norm', 'cls_token', 'pos_embed'])
    g = {}
    for n, p in model.named_parameters():
        if p.requires_grad: g.setdefault((depth(n), nd(n)), []).append(p)
    pgs = []
    for (d, ndf), ps in sorted(g.items()):
        lr = base_lr * (decay ** d)
        pgs.append({'params': ps, 'initial_lr': lr, 'lr': lr, 'weight_decay': 0.0 if ndf else wd})
    return torch.optim.AdamW(pgs)


def get_lr(ep, wu, mx, base, mn):
    if ep < wu: return base * (ep + 1) / wu
    t = (ep - wu) / max(1, mx - wu)
    return mn + 0.5 * (base - mn) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); L, P = [], []
    for x, y in loader:
        with torch.cuda.amp.autocast():
            o = model(x.to(DEVICE))
        L.append(y); P.append(torch.softmax(o, 1).cpu().float())
    return torch.cat(L).numpy(), torch.cat(P).numpy()


def bin_metrics(y, p):
    """Binary metrics at argmax; p is (N,2)."""
    auc = roc_auc_score(y, p[:, 1])
    pred = p.argmax(1)
    tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    se = tp / (tp + fn) if tp + fn else 0.0
    sp = tn / (tn + fp) if tn + fp else 0.0
    return auc, accuracy_score(y, pred), se, sp


def train_fold(fold):
    print(f'\n{"="*56}\n  FOLD {fold} — Model B maculopathy (focal γ=2, no weights)\n{"="*56}')
    vp = pat[pat['fold'] == fold]['code'].values
    dtr = df_cv[~df_cv['code'].isin(vp)]; dva = df_cv[df_cv['code'].isin(vp)]
    cc = np.bincount(dtr['grade_int'].values, minlength=2)
    print(f'  Train {len(dtr)} imgs {cc.tolist()} | Val {len(dva)} imgs')
    ltr = DataLoader(DS(recs(dtr), train_tf), batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=12, pin_memory=True, drop_last=False)
    lva = DataLoader(DS(recs(dva), eval_tf), batch_size=BATCH_SIZE, shuffle=False,
                     num_workers=12, pin_memory=True)
    lte = DataLoader(DS(recs(df_test), eval_tf), batch_size=BATCH_SIZE, shuffle=False,
                     num_workers=12, pin_memory=True)
    model = load_backbone(DEVICE, seed=SEED + fold)
    opt = llrd_opt(model, BASE_LR, WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()
    crit = FocalLoss(gamma=FOCAL_GAMMA, weight=None)
    ckpt = CV / f'best_fold_{fold}.pth'
    best = -1.0; torch.save(model.state_dict(), ckpt); wait = 0
    t0 = time.time()
    for ep in range(MAX_EPOCHS):
        model.train()
        hlr = get_lr(ep, WARMUP_EPOCHS, MAX_EPOCHS, BASE_LR, MIN_LR)
        for pg in opt.param_groups: pg['lr'] = pg['initial_lr'] * (hlr / BASE_LR)
        opt.zero_grad(); sc = 0
        for i, (x, y) in enumerate(ltr):
            x, y = x.to(DEVICE), y.to(DEVICE)
            step = ((sc + 1) % ACCUM_STEPS == 0) or (i + 1 == len(ltr))
            with torch.cuda.amp.autocast():
                loss = crit(model(x), y) / ACCUM_STEPS
            scaler.scale(loss).backward(); sc += 1
            if step:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sc = 0
        vy, vp = evaluate(model, lva); auc, acc, se, sp = bin_metrics(vy, vp)
        print(f'  ep {ep:02d} | lr={hlr:.2e} | AUROC={auc:.4f} sens={se:.3f} spec={sp:.3f} | {time.time()-t0:.0f}s')
        if auc > best:
            best = auc; wait = 0; torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f'  early stop ep {ep} (best AUROC {best:.4f})'); break
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    oy, op = evaluate(model, lva); ty, tp = evaluate(model, lte)
    np.save(CV / f'fold_{fold}_oof_probs.npy', op); np.save(CV / f'fold_{fold}_oof_labels.npy', oy)
    np.save(CV / f'fold_{fold}_test_probs.npy', tp); np.save(CV / f'fold_{fold}_test_labels.npy', ty)
    auc, acc, se, sp = bin_metrics(oy, op)
    print(f'  OOF: AUROC {auc:.4f} acc {acc:.4f} sens {se:.3f} spec {sp:.3f}')
    del model; torch.cuda.empty_cache()


def pool(dfx, P, how='mean'):
    P = P / P.sum(1, keepdims=True); rec = {}
    for c, p, g in zip(dfx['code'].values, P, dfx['grade_int'].values):
        rec.setdefault(c, {'p': [], 'g': 0}); rec[c]['p'].append(p); rec[c]['g'] = max(rec[c]['g'], int(g))
    ks = sorted(rec)
    A = np.array([(np.mean(rec[k]['p'], 0) if how == 'mean' else np.max(rec[k]['p'], 0)) for k in ks])
    A = A / A.sum(1, keepdims=True)
    return A, np.array([rec[k]['g'] for k in ks])


def report_oof():
    """Pooled OOF over whatever folds are on disk (pilot gate)."""
    have = [f for f in range(N_FOLDS) if (CV / f'fold_{f}_oof_probs.npy').exists()]
    print(f'\n=== Pooled OOF over folds {have} (image-level) ===')
    oof = np.zeros((len(df_cv), NUM_CLASSES)); lab = np.zeros(len(df_cv), int); mask = np.zeros(len(df_cv), bool)
    for f in have:
        vp = pat[pat['fold'] == f]['code'].values
        idx = df_cv[df_cv['code'].isin(vp)]['cv_idx'].values
        oof[idx] = np.load(CV / f'fold_{f}_oof_probs.npy'); lab[idx] = np.load(CV / f'fold_{f}_oof_labels.npy')
        mask[idx] = True
    y, p = lab[mask], oof[mask]
    auc, acc, se, sp = bin_metrics(y, p)
    print(f'  image-level: AUROC {auc:.4f} acc {acc:.4f} sens {se:.3f} spec {sp:.3f} (n={mask.sum()})')
    return auc


def sens_spec(y, s, thr):
    pred = (s >= thr).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    return (tp/(tp+fn) if tp+fn else 0, tn/(tn+fp) if tn+fp else 0,
            tp/(tp+fp) if tp+fp else 0, tn/(tn+fn) if tn+fn else 0, tp, fp, fn, tn)


def tta_and_operating_point():
    """4-way TTA over 5 folds on test, patient pooling, OOF-chosen M1 operating point."""
    assert all((CV / f'best_fold_{f}.pth').exists() for f in range(N_FOLDS)), 'need all 5 folds'
    ttas = [lambda i: eval_tf(i), lambda i: eval_tf(TF.hflip(i)),
            lambda i: eval_tf(TF.vflip(i)), lambda i: eval_tf(TF.vflip(TF.hflip(i)))]

    def tta_predict(model, paths):
        out = []
        for tf in ttas:
            ld = DataLoader(DS([(p, 0) for p in paths], tf), batch_size=32, shuffle=False,
                            num_workers=12, pin_memory=True)
            pr = []
            with torch.no_grad(), torch.cuda.amp.autocast():
                for x, _ in ld: pr.append(torch.softmax(model(x.to(DEVICE)), 1).float().cpu().numpy())
            out.append(np.vstack(pr))
        return np.mean(out, 0)

    # TEST TTA ensemble
    tp = df_test['image_path'].values
    fold_probs = np.zeros((N_FOLDS, len(df_test), NUM_CLASSES))
    for f in range(N_FOLDS):
        m = load_backbone(DEVICE, pretrained=False)
        m.load_state_dict(torch.load(CV / f'best_fold_{f}.pth', map_location=DEVICE, weights_only=True))
        m.eval(); fold_probs[f] = tta_predict(m, tp)
        print(f'  test TTA fold {f} done'); del m; torch.cuda.empty_cache()
    test_tta = fold_probs.mean(0); test_tta /= test_tta.sum(1, keepdims=True)
    np.save(CV / 'test_tta_probs.npy', test_tta)

    # OOF (no TTA needed for threshold pick; use per-fold OOF probs already saved)
    oof = np.zeros((len(df_cv), NUM_CLASSES)); olab = np.zeros(len(df_cv), int)
    for f in range(N_FOLDS):
        vp = pat[pat['fold'] == f]['code'].values
        idx = df_cv[df_cv['code'].isin(vp)]['cv_idx'].values
        oof[idx] = np.load(CV / f'fold_{f}_oof_probs.npy'); olab[idx] = np.load(CV / f'fold_{f}_oof_labels.npy')

    res = {'classes': CLASSES}
    print('\n================  Model B maculopathy — TEST (4-way TTA)  ================')
    for how in ['mean', 'max']:
        Ao, yo = pool(df_cv, oof, how); At, yt = pool(df_test, test_tta, how)
        so, st = Ao[:, 1], At[:, 1]
        auc = roc_auc_score(yt, st)
        line = {'auroc_test': float(auc), 'auroc_oof': float(roc_auc_score(yo, so)),
                'n_test': int(len(yt)), 'n_test_pos': int(yt.sum())}
        print(f'\nPt{how.capitalize()} | TEST AUROC={auc:.4f} | M1 prevalence test {yt.mean():.3f}')
        # operating points chosen on OOF
        def pick(mode, tgt):
            best = None
            for t in np.unique(np.r_[0, so, 1.0]):
                se, sp, *_ = sens_spec(yo, so, t)
                if mode == 'spec' and sp >= tgt and (best is None or se > best[1]): best = (t, se)
                if mode == 'sens' and se >= tgt and (best is None or sp > best[1]): best = (t, sp)
            return best[0] if best else 0.5
        ops = {}
        for name, mode, tgt in [('spec>=95%', 'spec', 0.95), ('sens>=85%', 'sens', 0.85),
                                ('Youden', 'youden', None)]:
            if mode == 'youden':
                fpr, tpr, th = roc_curve(yo, so); thr = float(th[(tpr - fpr).argmax()])
            else:
                thr = pick(mode, tgt)
            se, sp, ppv, npv, tp_, fp_, fn_, tn_ = sens_spec(yt, st, thr)
            print(f'    {name:10} thr={thr:.3f} | TEST sens={se:.3f} spec={sp:.3f} PPV={ppv:.3f} NPV={npv:.3f} '
                  f'(TP{tp_} FP{fp_} FN{fn_} TN{tn_})')
            ops[name] = dict(thr=float(thr), sens=float(se), spec=float(sp), ppv=float(ppv), npv=float(npv))
        line['operating_points'] = ops
        res[f'Pt{how.capitalize()}'] = line
    json.dump(res, open(CV / 'modelB_tta_results.json', 'w'), indent=2)
    print(f'\nSaved {CV}/modelB_tta_results.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, nargs='*', default=None)
    ap.add_argument('--tta', action='store_true')
    a = ap.parse_args()
    print(f'Device {DEVICE} | CV pool {len(df_cv)} imgs {df_cv["grade_int"].value_counts().to_dict()} '
          f'| test {len(df_test)} imgs')
    if a.folds is not None:
        for f in a.folds: train_fold(f)
        auc = report_oof()
        print(f'\nPILOT GATE: pooled OOF image-level AUROC = {auc:.4f}  '
              f'(proceed to remaining folds if healthy)')
    if a.tta:
        tta_and_operating_point()
