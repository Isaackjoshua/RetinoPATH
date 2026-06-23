import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_generated_notebook_markers():
    out = os.path.join(ROOT, 'phase6_3class_pilot.ipynb')
    if os.path.exists(out):
        os.remove(out)
    subprocess.check_call([sys.executable, os.path.join(ROOT, 'build_p6_notebook.py')], cwd=ROOT)
    nb = json.load(open(out))
    src = '\n'.join('\n'.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    assert 'NUM_CLASSES = 3' in src
    assert "CLASSES     = ['R0', 'R1', 'R2']" in src
    assert 'output_dir/phase6_3class_cv' in src
    assert 'from p6_cohort import filter_r0r2_patients, inverse_freq_weights' in src
    assert 'df_all = filter_r0r2_patients(df_all)' in src
    assert 'CLASS_WEIGHTS = inverse_freq_weights(' in src
    assert 'for fold in range(2)' in src
    assert 'torch.randn(16, 3)' in src            # focal sanity arity fixed
    assert 'NUM_CLASSES = 4' not in src
    assert 'output_dir/phase2b_cv' not in src
    assert "['R0', 'R1', 'R2', 'R3A']" not in src
    assert nb['metadata']['kernelspec']['name'] == 'retfound'
    # trailing 5-fold aggregation cells truncated: last code cell is the CV loop
    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    assert 'for fold in range(2)' in '\n'.join(code_cells[-1]['source'])
    assert all(c.get('outputs', []) == [] for c in nb['cells'] if c['cell_type'] == 'code')


if __name__ == '__main__':
    test_generated_notebook_markers()
    print('ALL TESTS PASSED')
