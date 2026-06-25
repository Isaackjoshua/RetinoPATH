"""Exp03 — 4-way TTA + patient aggregation on the test set (4-class).

Loads the 5 Exp03 fold checkpoints (224px, 4-class head), runs 4-way TTA over the
full test set (eval transform = build_transform('val'), matching how the model was
validated), ensembles across folds, then reports image-level and patient-pooled
(mean/max) metrics — the apples-to-apples comparison vs P2B's recommended config
(PtMean+TTA: Acc 0.8483, Kappa 0.8501, AUROC 0.9475, mSens 0.7513).
"""
import os, json, argparse
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np, pandas as pd, torch, timm
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import PIL.ImageFile
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import roc_auc_score, cohen_kappa_score, accuracy_score, confusion_matrix

from util.datasets import build_transform

INPUT_SIZE, NUM_CLASSES, N_FOLDS = 224, 4, 5
CVDIR = 'output_dir/exp03_bestmodel_hybrid_class_balancing_cv'
CLASSES = ['R0', 'R1', 'R2', 'R3A']
dev = 'cuda'
GRADE = {'R0': 0, 'R1': 1, 'R2': 2, 'R3A': 3}

df = pd.read_csv('labels/splits.csv')
df['grade_int'] = df['retinopathy'].map(GRADE)
df_te = df[df['split'] == 'test'].reset_index(drop=True)
test_paths = df_te['image_path'].values
print(f'Test set (4-class): {len(df_te)} images | {df_te["code"].nunique()} patients '
      f'| per-class images {df_te["retinopathy"].value_counts().reindex(CLASSES).tolist()}')

_eval_tf = build_transform('val', argparse.Namespace(input_size=INPUT_SIZE))
ttas = [lambda i: _eval_tf(i),
        lambda i: _eval_tf(TF.hflip(i)),
        lambda i: _eval_tf(TF.vflip(i)),
        lambda i: _eval_tf(TF.vflip(TF.hflip(i)))]


class DS(Dataset):
    def __init__(s, paths, tf): s.p = list(paths); s.tf = tf
    def __len__(s): return len(s.p)
    def __getitem__(s, i): return s.tf(Image.open(s.p[i]).convert('RGB'))


def load_fold(k):
    m = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=False,
                          img_size=INPUT_SIZE, num_classes=NUM_CLASSES, drop_path_rate=0.2)
    m.load_state_dict(torch.load(f'{CVDIR}/best_fold_{k}.pth', map_location=dev, weights_only=True))
    return m.eval().to(dev)


def tta_predict(m, paths):
    out = []
    for tf in ttas:
        loader = DataLoader(DS(paths, tf), batch_size=32, shuffle=False, num_workers=12, pin_memory=True)
        pr = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for x in loader:
                pr.append(torch.softmax(m(x.to(dev)), -1).float().cpu().numpy())
        out.append(np.vstack(pr))
    return np.mean(out, axis=0)


print('Running 4-way TTA over 5 folds on the test set...')
fold_probs = np.zeros((N_FOLDS, len(df_te), NUM_CLASSES))
for k in range(N_FOLDS):
    fold_probs[k] = tta_predict(load_fold(k), test_paths)
    np.save(f'{CVDIR}/fold_{k}_test_tta_probs.npy', fold_probs[k])
    print(f'  fold {k} done'); torch.cuda.empty_cache()
test_tta = fold_probs.mean(0)
test_tta = test_tta / test_tta.sum(1, keepdims=True)
np.save(f'{CVDIR}/test_tta_probs.npy', test_tta)
np.save(f'{CVDIR}/test_ensemble_labels.npy', df_te['grade_int'].values)
print('Saved test_tta_probs.npy + per-fold TTA probs')


def metrics(y, P):
    P = P / P.sum(1, keepdims=True); pr = P.argmax(1)
    au = roc_auc_score(y, P, multi_class='ovr', average='macro', labels=list(range(NUM_CLASSES)))
    cm = confusion_matrix(y, pr, labels=list(range(NUM_CLASSES)))
    se = np.array([cm[i, i] / cm[i].sum() if cm[i].sum() else np.nan for i in range(NUM_CLASSES)])
    sp = np.array([(cm.sum() - cm[i].sum() - cm[:, i].sum() + cm[i, i]) /
                   (cm.sum() - cm[i].sum()) for i in range(NUM_CLASSES)])
    return dict(acc=accuracy_score(y, pr), kappa=cohen_kappa_score(y, pr, weights='quadratic'),
                auroc=au, msens=np.nanmean(se), mspec=np.nanmean(sp), sens=se, cm=cm)


def pool(probs, how):
    rec = {}
    for c, p, l in zip(df_te['code'].values, probs, df_te['grade_int'].values):
        rec.setdefault(c, {'p': [], 'g': 0}); rec[c]['p'].append(p); rec[c]['g'] = max(rec[c]['g'], int(l))
    ks = sorted(rec)
    agg = np.array([(np.mean(rec[k]['p'], 0) if how == 'mean' else np.max(rec[k]['p'], 0)) for k in ks])
    agg = agg / agg.sum(1, keepdims=True)
    return agg, np.array([rec[k]['g'] for k in ks])


y = df_te['grade_int'].values
print('\n================  Exp03 4-class TTA RESULTS (test)  ================')
results = {}
def show(tag, yy, P):
    m = metrics(yy, P)
    s = m['sens']
    print(f"{tag:11s}| Acc={m['acc']:.4f} Kappa={m['kappa']:.4f} AUROC={m['auroc']:.4f} "
          f"mSens={m['msens']:.4f} mSpec={m['mspec']:.4f}  "
          f"R0={s[0]:.3f} R1={s[1]:.3f} R2={s[2]:.3f} R3A={s[3]:.3f}")
    results[tag.strip()] = dict(acc=m['acc'], kappa=m['kappa'], auroc=m['auroc'],
                                macro_sensitivity=m['msens'], macro_specificity=m['mspec'],
                                sensitivity=s.tolist(), confusion_matrix=m['cm'].tolist())
    return m

show('Image ', y, test_tta)
for how in ['mean', 'max']:
    P, yp = pool(test_tta, how)
    m = show(f'Pt{how.capitalize()} ', yp, P)
    if how == 'mean':
        print('   PtMean confusion (rows=true R0/R1/R2/R3A):')
        for i, c in enumerate(CLASSES):
            print('     ', c, m['cm'][i].tolist())

print('\nP2B recommended (PtMean+TTA): Acc=0.8483 Kappa=0.8501 AUROC=0.9475 mSens=0.7513  '
      'R0=.977 R1=.707 R2=.750 R3A=.571')
with open(f'{CVDIR}/exp03_tta_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=float)
print(f'Saved {CVDIR}/exp03_tta_results.json')
