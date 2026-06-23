"""Generate phase6_3class_pilot.ipynb by cloning phase2b_full_finetune.ipynb for a
3-class (R0/R1/R2) task: NUM_CLASSES 4->3, inject the R3A-patient filter + class-weight
recompute, fix the focal sanity-check arity, new output dir, pilot fold range, kernel
pin, and truncate trailing 5-fold aggregation cells."""
import copy, json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, 'phase2b_full_finetune.ipynb')
OUT = os.path.join(ROOT, 'phase6_3class_pilot.ipynb')


def sub(cell, a, b):
    cell['source'] = [ln.replace(a, b) for ln in cell['source']]


def main():
    nb = copy.deepcopy(json.load(open(SRC)))
    cells = nb['cells']

    # cell 1: config — 3 classes, provisional weights (recomputed in cell 3), output dir
    sub(cells[1], 'NUM_CLASSES = 4', 'NUM_CLASSES = 3')
    sub(cells[1], "CLASSES     = ['R0', 'R1', 'R2', 'R3A']", "CLASSES     = ['R0', 'R1', 'R2']")
    sub(cells[1], 'CLASS_WEIGHTS = [1.0, 1.796, 10.8469, 17.502]   # recomputed inverse-freq on NEW cohort',
        'CLASS_WEIGHTS = [1.0, 1.0, 1.0]   # provisional (cell 2 sanity); recomputed on 3-class cohort in cell 3')
    sub(cells[1], 'output_dir/phase2b_cv', 'output_dir/phase6_3class_cv')

    # cell 2: focal-loss gamma=0 vs CE sanity check — fix arity 4 -> 3
    sub(cells[2], 'torch.randn(16, 4)', 'torch.randn(16, 3)')
    sub(cells[2], 'torch.randint(0, 4, (16,))', 'torch.randint(0, 3, (16,))')

    # cell 3: inject cohort filter (after grade map) + weight recompute (after cv_idx)
    sub(cells[3], "df_all['grade_int'] = df_all['retinopathy'].map(GRADE)",
        "df_all['grade_int'] = df_all['retinopathy'].map(GRADE)\n"
        "from p6_cohort import filter_r0r2_patients, inverse_freq_weights\n"
        "df_all = filter_r0r2_patients(df_all)  # P6: drop patients whose max grade is R3A")
    sub(cells[3], "df_cv['cv_idx'] = df_cv.index",
        "df_cv['cv_idx'] = df_cv.index\n"
        "CLASS_WEIGHTS = inverse_freq_weights(df_cv['grade_int'].value_counts().to_dict())\n"
        "print('Recomputed 3-class CLASS_WEIGHTS:', CLASS_WEIGHTS)")
    sub(cells[3], 'Same folds as Phase 1/2A (SEED=42) — direct comparison valid.',
        'Folds re-run on 3-class cohort (R3A patients dropped) — NOT comparable to P2B folds.')

    # CV-loop cell (found by content): pilot range + pilot results filename
    cv_idx = next(i for i, c in enumerate(cells)
                  if c['cell_type'] == 'code' and 'for fold in range(N_FOLDS)' in ''.join(c['source']))
    sub(cells[cv_idx], 'for fold in range(N_FOLDS):', 'for fold in range(2):  # PILOT folds 0-1')
    sub(cells[cv_idx], 'fold_results.json', 'fold_results_pilot.json')

    # Truncate trailing 5-fold aggregation/eval cells so the pilot runs clean + saves outputs
    nb['cells'] = cells[:cv_idx + 1]

    # Clear stale outputs + pin kernel
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
