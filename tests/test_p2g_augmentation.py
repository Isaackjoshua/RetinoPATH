"""Plain-assert tests for P2G augmentation routing (no pytest available).
Run: /home/eth-admin/miniconda3/envs/retfound/bin/python tests/test_p2g_augmentation.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from p2g_augmentation import (
    MINORITY_LABELS,
    build_standard_train_transform,
    build_eval_transform,
    build_minority_train_transform,
    P2GDataset,
)

def _synthetic_image():
    arr = (np.random.rand(300, 300, 3) * 255).astype("uint8")
    return Image.fromarray(arr)

def _contains_random_erasing(compose):
    return any(type(t).__name__ == "RandomErasing" for t in compose.transforms)

def test_minority_labels_are_r2_r3a():
    assert MINORITY_LABELS == frozenset({2, 3})

def test_routing_selects_correct_transform():
    std = build_standard_train_transform()
    min_tf = build_minority_train_transform()
    eval_tf = build_eval_transform()
    ds = P2GDataset(records=[], standard_tf=std, minority_tf=min_tf,
                    eval_tf=eval_tf, train=True)
    assert ds.transform_for(0) is std
    assert ds.transform_for(1) is std
    assert ds.transform_for(2) is min_tf
    assert ds.transform_for(3) is min_tf

def test_eval_mode_always_uses_eval_transform():
    std = build_standard_train_transform()
    min_tf = build_minority_train_transform()
    eval_tf = build_eval_transform()
    ds = P2GDataset(records=[], standard_tf=std, minority_tf=min_tf,
                    eval_tf=eval_tf, train=False)
    for lbl in (0, 1, 2, 3):
        assert ds.transform_for(lbl) is eval_tf

def test_minority_transform_has_no_random_erasing():
    # The whole point of the geometry-heavy choice: protect tiny lesions.
    assert not _contains_random_erasing(build_minority_train_transform())

def test_standard_transform_does_have_random_erasing():
    # Sanity: baseline (P2B) really does carry erasing, so the difference is real.
    assert _contains_random_erasing(build_standard_train_transform())

def test_minority_transform_output_shape_and_normalized():
    t = build_minority_train_transform()(_synthetic_image())
    assert t.shape == torch.Size([3, 224, 224])
    assert t.dtype == torch.float32
    # Normalized images leave [0,1]: at least some negative values expected.
    assert t.min() < 0.0

def test_eval_transform_is_deterministic():
    img = _synthetic_image()
    t = build_eval_transform()
    assert torch.allclose(t(img), t(img))

def test_minority_transform_is_stochastic():
    img = _synthetic_image()
    t = build_minority_train_transform()
    assert not torch.allclose(t(img), t(img))

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
