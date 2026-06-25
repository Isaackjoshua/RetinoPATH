"""Generate Exp01_Normal_finetuning.ipynb by cloning phase2b_full_finetune.ipynb into a
TRULY VANILLA 4-class full fine-tune baseline.

Single-purpose: the reference point that all "added infrastructure" experiments compare
against. Exactly TWO functional changes vs P2B:
  1. Loss:      FocalLoss(gamma, class_weights)  ->  plain nn.CrossEntropyLoss()  (no weights)
  2. Optimizer: LLRD (per-layer LR decay 0.75)   ->  uniform LR  (build_llrd_optimizer
                                                       called with decay=1.0 => every layer
                                                       gets BASE_LR; cosine+warmup unchanged)
Everything else identical to P2B: 4 classes, grad checkpointing, batch 16 x accum 2,
cosine LR + 5-epoch warmup, AdamW WD 0.05, early stop on val AUROC, 5-fold CV, test
ensemble + Youden thresholds. New output dir, kernel pinned, cross-phase comparison cell
replaced with a self-contained Exp01 results+summary."""
import copy, json, os
from build_common import results_cell_source

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, 'phase2b_full_finetune.ipynb')
OUT = os.path.join(ROOT, 'Exp01_Normal_finetuning.ipynb')

OUT_DIR = 'output_dir/exp01_normal_finetuning_cv'


def rep(cell, a, b):
    """Replace substring a->b across the cell's full source (handles multi-line)."""
    s = ''.join(cell['source'])
    assert a in s, f'pattern not found: {a!r}'
    cell['source'] = (s.replace(a, b)).splitlines(keepends=True)


def main():
    nb = copy.deepcopy(json.load(open(SRC)))
    cells = nb['cells']

    # ── Cell 0: title / framing ───────────────────────────────────────────────
    cells[0]['cell_type'] = 'markdown'
    cells[0]['source'] = (
        "# Exp01 — Normal (Vanilla) Full Fine-Tuning\n"
        "\n"
        "**Baseline / primary comparison** for all later experiments that add infrastructure.\n"
        "Full fine-tune of RETFound-DINOv2-MEH on the 4 DR grades (R0/R1/R2/R3A) with the\n"
        "plainest possible recipe — no imbalance handling, no per-layer LR tricks.\n"
        "\n"
        "**Exactly two functional changes vs the P2B recipe:**\n"
        "\n"
        "| | P2B (recommended) | Exp01 (vanilla baseline) |\n"
        "|---|---|---|\n"
        "| Loss | Focal γ=2 + inverse-freq class weights | plain CrossEntropy (no weights) |\n"
        "| Per-layer LR | LLRD decay 0.75 (early layers barely move) | uniform — every layer at BASE_LR |\n"
        "\n"
        "Everything else is held identical to P2B for a clean comparison: 4 classes,\n"
        "gradient checkpointing, batch 16 × accum 2, AdamW (WD 0.05), cosine LR + 5-epoch\n"
        "warmup, early stopping on val AUROC, 5-fold patient-stratified CV, 5-model test\n"
        "ensemble + Youden thresholds.\n"
        "\n"
        "> NOTE: `FOCAL_GAMMA`, `LLRD_DECAY`, and `CLASS_WEIGHTS` remain defined in the config\n"
        "> cell (P2B inheritance) but are **unused** here — the loss is plain CE and the\n"
        "> optimizer is built with `decay=1.0` (uniform LR)."
    ).splitlines(keepends=True)

    # ── Cell 1: config — only the output dir changes ──────────────────────────
    rep(cells[1], 'output_dir/phase2b_cv', OUT_DIR)

    # ── Cell 2: drop FocalLoss definition + sanity check (vanilla uses plain CE) ─
    cells[2]['source'] = (
        "# Vanilla baseline: loss is plain nn.CrossEntropyLoss (defined in the CV loop).\n"
        "# No focal loss and no class weights — that is the whole point of this baseline.\n"
        "print('Loss: plain CrossEntropyLoss (no focal, no class weights).')"
    ).splitlines(keepends=True)

    # ── Cell 7 (markdown): note LLRD is disabled in this baseline ──────────────
    md7_idx = next(i for i, c in enumerate(cells)
                   if c['cell_type'] == 'markdown' and 'Layer-wise learning rate decay' in ''.join(c['source']))
    cells[md7_idx]['source'] = (
        "## Optimizer — uniform learning rate (LLRD disabled)\n"
        "\n"
        "P2B uses layer-wise LR decay (LLRD): the head trains at `BASE_LR` and each layer\n"
        "toward the input is scaled by `0.75`, so early blocks barely move. This vanilla\n"
        "baseline **disables** that — we reuse the same `build_llrd_optimizer` helper but call\n"
        "it with `decay=1.0`, so `BASE_LR × 1.0^depth = BASE_LR` for every parameter group.\n"
        "All layers therefore train at the same learning rate.\n"
        "\n"
        "The cosine schedule + 5-epoch warmup still scale every group together each epoch —\n"
        "only the *per-layer* differentiation is removed. Bias/norm/positional params keep\n"
        "weight-decay = 0 (standard ViT practice; not part of the baseline's two changes)."
    ).splitlines(keepends=True)

    # ── Cell 8: keep build_llrd_optimizer as-is, but its demo print table is LLRD-
    #            specific. Replace the trailing print block with a uniform-LR note. ─
    s8 = ''.join(cells[8]['source'])
    head8 = s8.split("# Print LLRD LR table")[0]
    cells[8]['source'] = (
        head8 +
        "# Uniform-LR baseline: called below with decay=1.0, so every group = BASE_LR.\n"
        "print(f'Uniform LR (no LLRD): all parameter groups at BASE_LR = {BASE_LR:.2e}')"
    ).splitlines(keepends=True)

    # ── Cell 11 (CV loop): plain CE + uniform optimizer + label tweaks ─────────
    rep(cells[11],
        "weight_tensor = torch.tensor(CLASS_WEIGHTS, dtype=torch.float).to(DEVICE)\n"
        "criterion_cv  = FocalLoss(gamma=FOCAL_GAMMA, weight=weight_tensor)",
        "criterion_cv  = nn.CrossEntropyLoss()  # vanilla baseline: no focal, no class weights")
    rep(cells[11],
        "  [full fine-tune, focal γ={FOCAL_GAMMA}]",
        "  [vanilla full fine-tune, plain CE]")
    rep(cells[11],
        "optimizer = build_llrd_optimizer(model, BASE_LR, WEIGHT_DECAY, decay=LLRD_DECAY)",
        "optimizer = build_llrd_optimizer(model, BASE_LR, WEIGHT_DECAY, decay=1.0)  # uniform LR (no LLRD)")

    # ── Cell 13: rename P2B-specific Youden print (var names kept; internal) ────
    rep(cells[13], 'Phase 2B Youden thresholds', 'Exp01 Youden thresholds')

    # ── Cells 14-15: replace cross-phase comparison with self-contained results ─
    cells[14]['cell_type'] = 'markdown'
    cells[14]['source'] = ["## Results — full metric suite (accuracy, kappa, macro sens/spec, AUROC, confusion matrix)"]
    cells[15]['source'] = results_cell_source(
        exp_tag='Exp01_Normal_finetuning',
        exp_label='Exp01 Vanilla',
        recipe='vanilla 4-class full fine-tune (plain CE, uniform LR, no class weights)',
        summary_name='exp01_summary.json')

    # ── Speed: raise DataLoader workers (32 idle CPUs; result-neutral) ────────
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
