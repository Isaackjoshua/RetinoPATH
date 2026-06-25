"""Shared notebook-cell builders for the ExpNN_* experiment series.

results_cell_source() returns the source (list of keepends lines) for a self-contained
results cell with the full metric suite the experiments require: accuracy, quadratic
kappa, macro sensitivity, macro specificity, macro AUROC, per-class sensitivity, and a
confusion matrix (printed as text AND saved as a PNG under figures/). It depends only on
variables already defined earlier in the cloned P2B notebook: CLASSES, NUM_CLASSES,
CV_OUTPUT, compute_metrics, apply_thresholds, youden_thr_p2b, BASE_LR, np, json, Path."""


def results_cell_source(exp_tag, exp_label, recipe, summary_name):
    """exp_tag: filesystem-safe id (also used for figure filenames);
    exp_label: short human label for table rows; recipe: one-line recipe string;
    summary_name: json filename written under CV_OUTPUT."""
    src = f'''# Self-contained results — full metric suite + confusion matrix. No cross-phase deps.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP       = {exp_tag!r}
EXP_LABEL = {exp_label!r}
RECIPE    = {recipe!r}
FIG_DIR   = Path('figures'); FIG_DIR.mkdir(exist_ok=True)


def _norm(p):
    p = p.astype(np.float64)
    return p / p.sum(axis=1, keepdims=True)


oof_lbl = np.load(CV_OUTPUT / 'oof_labels_all.npy')
oof_prb = _norm(np.load(CV_OUTPUT / 'oof_probs_all.npy'))
tst_lbl = np.load(CV_OUTPUT / 'test_ensemble_labels.npy')
tst_prb = _norm(np.load(CV_OUTPUT / 'test_ensemble_probs.npy'))


def print_table(title, items):
    print('\\n' + '=' * 104)
    print(f'  {{title}}')
    print('=' * 104)
    hdr = (f'{{"Configuration":<22}} | {{"Acc":>6}} | {{"AUROC":>6}} | {{"Kappa":>6}} | '
           f'{{"MacSens":>7}} | {{"MacSpec":>7}} | ' + ' | '.join(f'{{c:>6}}' for c in CLASSES))
    print(hdr)
    print('-' * len(hdr))
    for name, lbl, prb, thr in items:
        preds = apply_thresholds(prb, thr) if thr is not None else None
        m = compute_metrics(lbl, prb, preds)
        s = m['sensitivity']
        print(f'{{name:<22}} | {{m["accuracy"]:>6.4f}} | {{m["auroc"]:>6.4f}} | {{m["kappa"]:>6.4f}} | '
              f'{{m["macro_sensitivity"]:>7.4f}} | {{m["macro_specificity"]:>7.4f}} | '
              + ' | '.join(f'{{s[i]:>6.4f}}' for i in range(NUM_CLASSES)))


print_table(f'{{EXP_LABEL}} — OOF', [
    (f'{{EXP_LABEL}} Argmax', oof_lbl, oof_prb, None),
    (f'{{EXP_LABEL}} Youden', oof_lbl, oof_prb, youden_thr_p2b),
])
print_table(f'{{EXP_LABEL}} — TEST (5-fold ensemble)', [
    (f'{{EXP_LABEL}} Argmax', tst_lbl, tst_prb, None),
    (f'{{EXP_LABEL}} Youden', tst_lbl, tst_prb, youden_thr_p2b),
])


def show_confusion(labels, probs, title, fname):
    preds = probs.argmax(1)
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    print(f'\\nConfusion matrix — {{title}} (rows=true, cols=pred):')
    print('       ' + ' '.join(f'{{c:>6}}' for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f'{{c:>6}} ' + ' '.join(f'{{cm[i, j]:>6d}}' for j in range(NUM_CLASSES)))
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
    print(f'  saved {{FIG_DIR / fname}}')
    return cm


cm_oof  = show_confusion(oof_lbl, oof_prb, f'{{EXP_LABEL}} OOF',  f'{{EXP}}_cm_oof.png')
cm_test = show_confusion(tst_lbl, tst_prb, f'{{EXP_LABEL}} TEST', f'{{EXP}}_cm_test.png')


def _pack(lbl, prb, thr=None):
    preds = apply_thresholds(prb, thr) if thr is not None else None
    m = compute_metrics(lbl, prb, preds)
    return {{'accuracy': m['accuracy'], 'auroc': m['auroc'], 'kappa': m['kappa'],
            'macro_sensitivity': m['macro_sensitivity'], 'macro_specificity': m['macro_specificity'],
            'sensitivity': m['sensitivity'].tolist(), 'specificity': m['specificity'].tolist()}}


summary = {{
    'experiment': EXP,
    'recipe': RECIPE,
    'classes': CLASSES,
    'base_lr': BASE_LR,
    'youden_thresholds': {{c: youden_thr_p2b[i] for i, c in enumerate(CLASSES)}},
    'oof':  {{'Argmax': _pack(oof_lbl, oof_prb), 'Youden': _pack(oof_lbl, oof_prb, youden_thr_p2b)}},
    'test': {{'Argmax': _pack(tst_lbl, tst_prb), 'Youden': _pack(tst_lbl, tst_prb, youden_thr_p2b)}},
    'confusion_matrix': {{'oof': cm_oof.tolist(), 'test': cm_test.tolist()}},
}}
with open(CV_OUTPUT / {summary_name!r}, 'w') as f:
    json.dump(summary, f, indent=2, default=float)
print(f'\\nSummary saved to {{CV_OUTPUT}}/{summary_name}')
'''
    return src.splitlines(keepends=True)
