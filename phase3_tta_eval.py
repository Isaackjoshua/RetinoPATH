"""Phase 3 — 518px TTA + patient aggregation on the test set.

Loads the 5 fold checkpoints trained at 518, runs 4-way TTA over the test set
(eval transform matched to training: Resize(518)+CenterCrop(518)), ensembles
across folds, then reports image-level and patient-pooled (mean/max) metrics —
the apples-to-apples comparison vs the 224 recommended config (PtMean+TTA=0.7513).
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np, pandas as pd, torch, timm
from torchvision.transforms import functional as TF
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import PIL.ImageFile
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import roc_auc_score, cohen_kappa_score, accuracy_score, confusion_matrix

INPUT_SIZE, NUM_CLASSES, N_FOLDS = 518, 4, 5
CVDIR = 'output_dir/phase3_res518_cv'
dev = 'cuda'
GRADE = {'R0': 0, 'R1': 1, 'R2': 2, 'R3A': 3}

df = pd.read_csv('labels/splits.csv'); df['g'] = df['retinopathy'].map(GRADE)
df_te = df[df['split'] == 'test'].reset_index(drop=True)
test_paths = df_te['image_path'].values

def base_tf(img):
    img = TF.resize(img, INPUT_SIZE, interpolation=TF.InterpolationMode.BICUBIC)
    img = TF.center_crop(img, INPUT_SIZE)
    return TF.normalize(TF.to_tensor(img), IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

ttas = [lambda i: base_tf(i), lambda i: base_tf(TF.hflip(i)),
        lambda i: base_tf(TF.vflip(i)), lambda i: base_tf(TF.vflip(TF.hflip(i)))]

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
        loader = DataLoader(DS(paths, tf), batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        pr = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for x in loader: pr.append(torch.softmax(m(x.to(dev)), -1).float().cpu().numpy())
        out.append(np.vstack(pr))
    return np.mean(out, axis=0)

print('Running 518 TTA over 5 folds on test set...')
fold_probs = np.zeros((N_FOLDS, len(df_te), NUM_CLASSES))
for k in range(N_FOLDS):
    fold_probs[k] = tta_predict(load_fold(k), test_paths)
    print(f'  fold {k} done'); torch.cuda.empty_cache()
test_tta = fold_probs.mean(0)
test_tta = test_tta / test_tta.sum(1, keepdims=True)
np.save(f'{CVDIR}/test_tta_probs.npy', test_tta)
np.save(f'{CVDIR}/test_ensemble_labels.npy', df_te['g'].values)
print('Saved test_tta_probs.npy')

def metrics(y, P):
    P = P / P.sum(1, keepdims=True); pr = P.argmax(1)
    au = roc_auc_score(y, P, multi_class='ovr', average='macro', labels=[0,1,2,3])
    cm = confusion_matrix(y, pr, labels=[0,1,2,3])
    se = np.array([cm[i,i]/cm[i].sum() if cm[i].sum() else np.nan for i in range(4)])
    return dict(acc=accuracy_score(y,pr), kappa=cohen_kappa_score(y,pr,weights='quadratic'),
               auroc=au, msens=np.nanmean(se), sens=se)

def pool(probs, how):
    rec = {}
    for c, p, l in zip(df_te['code'].values, probs, df_te['g'].values):
        rec.setdefault(c, {'p': [], 'g': 0}); rec[c]['p'].append(p); rec[c]['g'] = max(rec[c]['g'], int(l))
    ks = sorted(rec)
    agg = np.array([(np.mean(rec[k]['p'],0) if how=='mean' else np.max(rec[k]['p'],0)) for k in ks])
    agg = agg / agg.sum(1, keepdims=True)
    return agg, np.array([rec[k]['g'] for k in ks])

y = df_te['g'].values
print('\n================  518 TTA RESULTS (test)  ================')
m = metrics(y, test_tta)
print(f"Image  | TTA Argmax : Acc={m['acc']:.4f} Kappa={m['kappa']:.4f} AUROC={m['auroc']:.4f} mSens={m['msens']:.4f}  "
      f"R0={m['sens'][0]:.3f} R1={m['sens'][1]:.3f} R2={m['sens'][2]:.3f} R3A={m['sens'][3]:.3f}")
for how in ['mean', 'max']:
    P, yp = pool(test_tta, how)
    m = metrics(yp, P)
    print(f"Pt{how.capitalize():4}| TTA Argmax : Acc={m['acc']:.4f} Kappa={m['kappa']:.4f} AUROC={m['auroc']:.4f} mSens={m['msens']:.4f}  "
          f"R0={m['sens'][0]:.3f} R1={m['sens'][1]:.3f} R2={m['sens'][2]:.3f} R3A={m['sens'][3]:.3f}")
print('\n224 recommended (PtMean+TTA): Acc=0.8483 Kappa=0.8501 AUROC=0.9475 mSens=0.7513  R0=.977 R1=.707 R2=.750 R3A=.571')
