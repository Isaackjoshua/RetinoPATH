"""Generate Exp03_bestmodel_hybrid_class_balancing.ipynb by cloning the BEST model
(phase2b_full_finetune.ipynb) and swapping its imbalance handling:

  P2B:   Focal γ=2 + inverse-freq class weights [1,1.8,10.8,17.5]   (weight-balancing)
  Exp03: Focal γ=2 + NO class weights + hybrid balanced sampler     (sampler-balancing)

Rationale: stacking class weights AND a balanced sampler double-corrects the imbalance
(R3A would get ~17× loss weight on top of ~4× oversampling → minority over-fire / R1
collapse, P2E-style). So Exp03 keeps ONE balancing mechanism — the sampler — and drops
the weights. Everything else is held identical to the best model: DINOv2-MEH backbone,
LLRD 0.75, focal γ=2, grad checkpointing, batch 16×accum 2, cosine+warmup, 5-fold CV.
Reported at image-level here; a TTA+PtMean eval (exp03_tta_eval.py) gives the apples-to-
apples vs P2B's recommended config.

Balancing method = same hybrid as Exp02: WeightedRandomSampler, weight 1/class_count,
num_samples = TARGET_PER_CLASS*NUM_CLASSES → ~TARGET_PER_CLASS of every class per epoch
(R0/R1 undersampled fresh each epoch, R2/R3A oversampled). Train loader only."""
import copy, json, os
from build_common import results_cell_source

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, 'phase2b_full_finetune.ipynb')
OUT = os.path.join(ROOT, 'Exp03_bestmodel_hybrid_class_balancing.ipynb')


def rep(cell, a, b):
    s = ''.join(cell['source'])
    assert a in s, f'pattern not found: {a!r}'
    cell['source'] = (s.replace(a, b)).splitlines(keepends=True)


def find(cells, needle):
    return next(i for i, c in enumerate(cells)
               if c['cell_type'] == 'code' and needle in ''.join(c['source']))


def main():
    nb = copy.deepcopy(json.load(open(SRC)))
    cells = nb['cells']

    # ── Cell 0: title / framing ───────────────────────────────────────────────
    cells[0]['cell_type'] = 'markdown'
    cells[0]['source'] = (
        "# Exp03 — Best Model + Hybrid Class Balancing\n"
        "\n"
        "Takes the project's **best architecture** (P2B: RETFound-DINOv2-MEH full fine-tune,\n"
        "focal γ=2, LLRD 0.75, grad checkpointing, 4-way TTA + patient-mean pooling) and swaps\n"
        "its imbalance handling from **inverse-frequency class weights** to the **hybrid\n"
        "balanced sampler** from Exp02.\n"
        "\n"
        "| | P2B (best model) | Exp03 |\n"
        "|---|---|---|\n"
        "| Loss | Focal γ=2 + weights [1, 1.8, 10.8, 17.5] | Focal γ=2, **no class weights** |\n"
        "| Class balance | via loss weights | via **hybrid sampler** (~1000/class/epoch) |\n"
        "| Backbone / LLRD / TTA / PtMean | — | **unchanged** |\n"
        "\n"
        "**Why drop the weights:** keeping inverse-freq weights *and* a balanced sampler\n"
        "double-corrects the imbalance — R3A would get ~17× loss weight on top of ~4×\n"
        "oversampling, over-firing minorities and risking an R1 collapse (the P2E failure).\n"
        "So Exp03 uses a single balancing mechanism (the sampler) and an unweighted focal loss.\n"
        "\n"
        "**Metrics:** accuracy, quadratic kappa, macro sensitivity/specificity, macro AUROC,\n"
        "per-class sensitivity, confusion matrices. Run `exp03_tta_eval.py` afterwards for the\n"
        "TTA + PtMean numbers comparable to P2B's recommended config."
    ).splitlines(keepends=True)

    # ── Config: sampler import, TARGET_PER_CLASS, output dir ──────────────────
    cfg = find(cells, 'NUM_CLASSES = 4')
    rep(cells[cfg],
        'from torch.utils.data import Dataset, DataLoader',
        'from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler')
    rep(cells[cfg], 'output_dir/phase2b_cv', 'output_dir/exp03_bestmodel_hybrid_class_balancing_cv')
    rep(cells[cfg],
        "HF_REPO   = 'YukunZhou/RETFound_dinov2_meh'",
        "# ── Hybrid class balancing (Exp03) ─────────────────────────────────\n"
        "TARGET_PER_CLASS = 1000   # WeightedRandomSampler draws ~this many of EACH class/epoch\n"
        "                          # (R0/R1 undersampled, R2/R3A oversampled → balanced ~4000/epoch)\n"
        "\n"
        "HF_REPO   = 'YukunZhou/RETFound_dinov2_meh'")

    # ── CV loop: drop class weights + add hybrid sampler ──────────────────────
    cv = find(cells, 'for fold in range(N_FOLDS):')
    rep(cells[cv],
        "weight_tensor = torch.tensor(CLASS_WEIGHTS, dtype=torch.float).to(DEVICE)\n"
        "criterion_cv  = FocalLoss(gamma=FOCAL_GAMMA, weight=weight_tensor)",
        "criterion_cv  = FocalLoss(gamma=FOCAL_GAMMA, weight=None)  # Exp03: balancing via sampler, no class weights")
    rep(cells[cv],
        "    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,\n"
        "                              num_workers=4, pin_memory=True, drop_last=False)",
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
    rep(cells[cv], '  [full fine-tune, focal γ={FOCAL_GAMMA}]',
        '  [best-model + hybrid balancing, focal γ={FOCAL_GAMMA} (no weights)]')

    # ── Results cell: full metric suite + confusion matrix ────────────────────
    res = find(cells, 'P1_DIR  = Path')
    cells[res - 1]['source'] = ["## Results — full metric suite (accuracy, kappa, macro sens/spec, AUROC, confusion matrix)"]
    cells[res - 1]['cell_type'] = 'markdown'
    cells[res]['source'] = results_cell_source(
        exp_tag='Exp03_bestmodel_hybrid_class_balancing',
        exp_label='Exp03 Best+Bal',
        recipe='best-model (focal γ=2, LLRD 0.75) + hybrid under+over sampling to ~1000/class, NO class weights',
        summary_name='exp03_summary.json')

    # ── Speed: raise DataLoader workers (result-neutral) ──────────────────────
    for c in nb['cells']:
        if c['cell_type'] == 'code' and 'num_workers=4' in ''.join(c['source']):
            c['source'] = [ln.replace('num_workers=4', 'num_workers=12') for ln in c['source']]

    # ── Clear stale outputs + pin kernel ──────────────────────────────────────
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
