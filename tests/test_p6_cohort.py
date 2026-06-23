import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from p6_cohort import filter_r0r2_patients, inverse_freq_weights


def test_filter_drops_max_r3a_patients():
    df = pd.DataFrame({
        'code':      ['A', 'A', 'B', 'C', 'C'],
        'grade_int': [ 0,   3,   1,   2,   2 ],   # A max=3(R3A)->drop; B max=1; C max=2
    })
    out = filter_r0r2_patients(df)
    assert set(out['code']) == {'B', 'C'}
    assert len(out) == 3                          # B(1) + C(2 rows), A fully removed
    assert out['grade_int'].max() <= 2


def test_inverse_freq_weights_majority_is_one():
    w = inverse_freq_weights({0: 100, 1: 50, 2: 10})
    assert w[0] == 1.0
    assert abs(w[1] - 2.0) < 1e-9
    assert abs(w[2] - 10.0) < 1e-9


if __name__ == '__main__':
    test_filter_drops_max_r3a_patients()
    test_inverse_freq_weights_majority_is_one()
    print('ALL TESTS PASSED')
