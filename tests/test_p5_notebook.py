import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_generated_notebook_markers():
    out = os.path.join(ROOT, 'phase5_mae_pilot.ipynb')
    if os.path.exists(out):
        os.remove(out)
    subprocess.check_call([sys.executable, os.path.join(ROOT, 'build_p5_notebook.py')], cwd=ROOT)
    nb = json.load(open(out))               # valid JSON
    src = '\n'.join('\n'.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    assert 'output_dir/phase5_mae_cv' in src
    assert 'YukunZhou/RETFound_mae_meh' in src
    assert 'RETFound_mae_meh.pth' in src
    assert 'from p5_mae import' in src
    assert 'build_mae_llrd_optimizer' in src
    assert 'for fold in range(2)' in src
    assert 'output_dir/phase2b_cv' not in src      # no leftover P2B paths
    assert 'RETFound_dinov2_meh' not in src
    # all code cells cleared of stale outputs
    assert all(c.get('outputs', []) == [] for c in nb['cells'] if c['cell_type'] == 'code')


if __name__ == '__main__':
    test_generated_notebook_markers()
    print('ALL TESTS PASSED')
