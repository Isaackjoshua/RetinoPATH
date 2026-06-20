"""Smoke test: P2G data path on real images.
Run: /home/eth-admin/miniconda3/envs/retfound/bin/python tests/smoke_p2g_datapath.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from p2g_augmentation import (
    build_standard_train_transform, build_eval_transform,
    build_minority_train_transform, P2GDataset, MINORITY_LABELS,
)

GRADE = {"R0": 0, "R1": 1, "R2": 2, "R3A": 3}

def main():
    df = pd.read_csv("labels/splits.csv")
    df["grade_int"] = df["retinopathy"].map(GRADE)
    df = df[df["split"].isin(["train", "val"])]

    # Tiny balanced-ish subset incl. minority classes.
    recs = []
    for g in (0, 1, 2, 3):
        sub = df[df["grade_int"] == g].head(8)
        recs += [(r.image_path, r.grade_int) for r in sub.itertuples()]
    assert any(lbl in MINORITY_LABELS for _, lbl in recs), "subset has no R2/R3A"
    print(f"Subset: {len(recs)} images, labels = {sorted({l for _,l in recs})}")

    std = build_standard_train_transform()
    minp = build_minority_train_transform()
    ev = build_eval_transform()

    # Train loader yields correct shape.
    ds_tr = P2GDataset(recs, std, minp, eval_tf=ev, train=True)
    xb, yb = next(iter(DataLoader(ds_tr, batch_size=8, shuffle=True)))
    assert xb.shape == torch.Size([8, 3, 224, 224]), xb.shape
    assert xb.dtype == torch.float32
    print("Train batch OK:", tuple(xb.shape))

    # Find a minority record; train mode is stochastic, eval mode is deterministic.
    mi = next(i for i, (_, l) in enumerate(recs) if l in MINORITY_LABELS)
    a, _ = ds_tr[mi]; b, _ = ds_tr[mi]
    assert not torch.allclose(a, b), "minority aug should be stochastic"
    ds_ev = P2GDataset(recs, std, minp, eval_tf=ev, train=False)
    c, _ = ds_ev[mi]; d, _ = ds_ev[mi]
    assert torch.allclose(c, d), "eval mode must be deterministic"
    print("Routing OK: minority train stochastic, eval deterministic.")
    print("\nSmoke test passed.")

if __name__ == "__main__":
    main()
