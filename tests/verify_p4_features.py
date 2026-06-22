"""Verify P4 lesion-feature columns in splits.csv carry the validated signal.
Run: /home/eth-admin/miniconda3/envs/retfound/bin/python tests/verify_p4_features.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

FEATS = ['haem', 'exud', 'cws', 'nvd']

def main():
    df = pd.read_csv('labels/splits.csv')
    for c in FEATS:
        assert c in df.columns, f'missing feature column: {c}'
        assert set(df[c].dropna().unique()) <= {0, 1}, f'{c} not binary: {df[c].unique()}'
    # R0 must be 0% on every feature (validated: blank == absent)
    r0 = df[df['retinopathy'] == 'R0']
    for c in FEATS:
        assert r0[c].sum() == 0, f'R0 has nonzero {c} ({r0[c].sum()})'
    # haemorrhage prevalence must rise R1 -> R2 (severity gradient)
    pv = df.groupby('retinopathy')['haem'].mean()
    assert pv.get('R2', 0) > pv.get('R1', 0) > 0, f'haem gradient wrong: {pv.to_dict()}'
    # nvd must be R3A-specific (present in R3A, ~0 in R0/R1/R2)
    nvd = df.groupby('retinopathy')['nvd'].mean()
    assert nvd.get('R3A', 0) > 0.1, f'nvd not present in R3A: {nvd.to_dict()}'
    assert nvd.get('R1', 0) == 0 and nvd.get('R2', 0) == 0, f'nvd leaked to R1/R2: {nvd.to_dict()}'
    print('PASS: feature columns present, binary, R0=0, haem gradient, nvd R3A-specific.')

if __name__ == '__main__':
    main()
