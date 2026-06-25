"""Generate Exp02_Balanced_class_finetuning.ipynb by cloning Exp01_Normal_finetuning.ipynb
and adding ONE axis: hybrid class balancing on the training stream.

Exp02 = Exp01 (vanilla: plain CE, uniform LR) + balanced sampling. Single-variable change
so Exp01 vs Exp02 isolates the effect of class balancing.

Balancing method (user choice): HYBRID under+over to ~TARGET_PER_CLASS images of every class
per epoch, via a WeightedRandomSampler with per-sample weight 1/class_count and num_samples =
TARGET_PER_CLASS * NUM_CLASSES. Majorities (R0/R1) are undersampled (a different random subset
each epoch), minorities (R2/R3A) are oversampled (repeated). Applied to the TRAIN loader only;
val/test stay at the true distribution so all reported metrics are honest.

Requires Exp01_Normal_finetuning.ipynb to exist (run build_exp01_notebook.py first)."""
import copy, json, os
from build_common import results_cell_source

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, 'Exp01_Normal_finetuning.ipynb')
OUT = os.path.join(ROOT, 'Exp02_Balanced_class_finetuning.ipynb')


def rep(cell, a, b):
    s = ''.join(cell['source'])
    assert a in s, f'pattern not found: {a!r}'
    cell['source'] = (s.replace(a, b)).splitlines(keepends=True)


def find(cells, needle):
    return next(i for i, c in enumerate(cells)
               if c['cell_type'] == 'code' and needle in ''.join(c['source']))


def main():
    if not os.path.exists(SRC):
        raise SystemExit('Run build_exp01_notebook.py first (Exp01 is the clone source).')
    nb = copy.deepcopy(json.load(open(SRC)))
    cells = nb['cells']

    # ── Cell 0: title / framing ───────────────────────────────────────────────
    cells[0]['cell_type'] = 'markdown'
    cells[0]['source'] = (
        "# Exp02 — Balanced-Class Full Fine-Tuning (hybrid sampling)\n"
        "\n"
        "**Question:** how does the model perform when the four DR grades are presented in\n"
        "*balanced* proportions during training? Builds directly on the Exp01 vanilla recipe\n"
        "(plain CrossEntropy, uniform LR) and adds exactly **one** axis — a class-balancing\n"
        "sampler — so Exp01 → Exp02 isolates the effect of balancing.\n"
        "\n"
        "**Balancing method — hybrid under + over (train only):**\n"
        "A `WeightedRandomSampler` gives every image weight `1/class_count`, so each class\n"
        "contributes equal total probability. With `num_samples = TARGET_PER_CLASS × 4`, each\n"
        "epoch draws on average ~`TARGET_PER_CLASS` images of **every** class:\n"
        "\n"
        "| Class | CV images | Per epoch | Effect |\n"
        "|---|---|---|---|\n"
        "| R0 | 4393 | ~1000 | undersampled (fresh subset each epoch) |\n"
        "| R1 | 2446 | ~1000 | undersampled |\n"
        "| R2 | 405 | ~1000 | oversampled (~2.5×, repeats) |\n"
        "| R3A | 251 | ~1000 | oversampled (~4×, repeats) |\n"
        "\n"
        "The majority subset is re-drawn every epoch (so R0/R1 data isn't permanently\n"
        "discarded), and **val/test loaders are untouched** — every reported metric reflects\n"
        "the true class distribution.\n"
        "\n"
        "> Prior caution (P2E): a `WeightedRandomSampler` + plain CE once collapsed R1\n"
        "> sensitivity to 0.000 by over-firing on R2. The confusion matrix below is the check\n"
        "> for that failure mode.\n"
        "\n"
        "**Metrics reported:** accuracy, quadratic kappa, macro sensitivity, macro\n"
        "specificity, macro AUROC, per-class sensitivity, and confusion matrices (OOF + test)."
    ).splitlines(keepends=True)

    # ── Config cell: import sampler, add TARGET_PER_CLASS, repoint output dir ──
    cfg = find(cells, 'NUM_CLASSES = 4')
    rep(cells[cfg],
        'from torch.utils.data import Dataset, DataLoader',
        'from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler')
    rep(cells[cfg],
        "output_dir/exp01_normal_finetuning_cv",
        "output_dir/exp02_balanced_class_finetuning_cv")
    rep(cells[cfg],
        "HF_REPO   = 'YukunZhou/RETFound_dinov2_meh'",
        "# ── Hybrid class balancing (Exp02) ─────────────────────────────────\n"
        "TARGET_PER_CLASS = 1000   # WeightedRandomSampler draws ~this many of EACH class/epoch\n"
        "                          # (R0/R1 undersampled, R2/R3A oversampled → balanced ~4000/epoch)\n"
        "\n"
        "HF_REPO   = 'YukunZhou/RETFound_dinov2_meh'")

    # ── CV loop: swap plain shuffled loader for the balanced sampler ───────────
    cv = find(cells, 'for fold in range(N_FOLDS):')
    rep(cells[cv],
        "    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,\n"
        "                              num_workers=12, pin_memory=True, drop_last=False)",
        "    # ── Hybrid class balancing (train only) ─────────────────────────\n"
        "    # Per-sample weight 1/class_count => equal total probability per class; with\n"
        "    # num_samples = TARGET_PER_CLASS*NUM_CLASSES each epoch draws ~TARGET_PER_CLASS of\n"
        "    # EVERY class (majorities subsampled fresh each epoch, minorities repeated).\n"
        "    tr_labels    = df_fold_train['grade_int'].values\n"
        "    class_counts = np.bincount(tr_labels, minlength=NUM_CLASSES)\n"
        "    sample_w     = 1.0 / class_counts[tr_labels]\n"
        "    g_sampler    = torch.Generator().manual_seed(SEED + fold)\n"
        "    balanced_sampler = WeightedRandomSampler(\n"
        "        weights=torch.as_tensor(sample_w, dtype=torch.double),\n"
        "        num_samples=TARGET_PER_CLASS * NUM_CLASSES, replacement=True, generator=g_sampler)\n"
        "    print(f'  Balanced stream: ~{TARGET_PER_CLASS}/class × {NUM_CLASSES} = '\n"
        "          f'{TARGET_PER_CLASS * NUM_CLASSES}/epoch (orig fold counts {class_counts.tolist()})')\n"
        "    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, sampler=balanced_sampler,\n"
        "                              num_workers=12, pin_memory=True, drop_last=False)")
    rep(cells[cv], '  [vanilla full fine-tune, plain CE]', '  [balanced hybrid full fine-tune, plain CE]')

    # ── Results cell: re-emit with Exp02 identity ─────────────────────────────
    res = find(cells, "EXP       = 'Exp01_Normal_finetuning'")
    cells[res]['source'] = results_cell_source(
        exp_tag='Exp02_Balanced_class_finetuning',
        exp_label='Exp02 Balanced',
        recipe='balanced 4-class full fine-tune (plain CE, uniform LR, hybrid under+over to ~1000/class)',
        summary_name='exp02_summary.json')

    # ── Clear stale outputs + keep kernel pin ─────────────────────────────────
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            c['outputs'] = []
            c['execution_count'] = None
    nb.setdefault('metadata', {})['kernelspec'] = {
        'display_name': 'retfound', 'language': 'python', 'name': 'retfound'}

    json.dump(nb, open(OUT, 'w'), indent=1)
    print(f'Wrote {OUT}')


if __name__ == '__main__':
    main()
