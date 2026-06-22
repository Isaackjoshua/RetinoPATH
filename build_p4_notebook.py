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

CELL4 = lines('''# ── P4 multi-task dataset + feature pos_weight ─────────────────────────
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

CELL9_APPEND = '''

# ── P4 multi-task train/eval (grade tuple-aware) ─────────────────────
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

CELL11 = lines('''# ── P4 multi-task CV loop (folds 0-1 pilot) ──────────────────────
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

def patch_cell8(src):
    """Make the LLRD optimizer wrapper-aware: params live under backbone.*,
    and the feature_head (contains 'head') routes to head-LR (depth 0)."""
    text = "".join(src)
    assert "num_blocks = len(model.blocks)" in text and "def get_depth(name):" in text
    text = text.replace(
        "num_blocks = len(model.blocks)",
        "num_blocks = len(model.backbone.blocks) if hasattr(model, 'backbone') else len(model.blocks)")
    text = text.replace(
        "    def get_depth(name):\n",
        "    def get_depth(name):\n        name = name.replace('backbone.', '')  # multi-task wrapper prefix\n")
    return lines(text)

def main():
    nb = json.load(open(SRC)); cells = nb["cells"]
    cells[0]["source"] = CELL0
    cells[1]["source"] = patch_cell1(cells[1]["source"])
    cells[4]["source"] = CELL4
    cells[8]["source"] = patch_cell8(cells[8]["source"])
    cells[9]["source"] = lines("".join(cells[9]["source"]) + CELL9_APPEND)
    cells[11]["source"] = CELL11
    nb["cells"] = cells[:12]
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []; c["execution_count"] = None
    json.dump(nb, open(DST, "w"), indent=1)
    print(f"Wrote {DST} ({len(nb['cells'])} cells).")

if __name__ == "__main__":
    main()
