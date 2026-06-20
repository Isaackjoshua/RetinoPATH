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
