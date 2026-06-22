"""CPU unit tests for the P4 multi-task module (no GPU, no HF download).
Run: /home/eth-admin/miniconda3/envs/retfound/bin/python tests/test_p4_multitask.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, timm
from PIL import Image
import torchvision.transforms as T

from p4_multitask import (FEATURE_NAMES, MultiTaskRETFound, build_multitask_model,
                          MultiTaskLoss, P4Dataset, compute_feature_pos_weight)

def _backbone():
    return timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=False,
                             img_size=224, num_classes=4)

def test_feature_names_order():
    assert FEATURE_NAMES == ['haem', 'exud', 'cws', 'nvd']

def test_forward_shapes():
    m = build_multitask_model(_backbone()).eval()
    g, f = m(torch.randn(2, 3, 224, 224))
    assert g.shape == (2, 4) and f.shape == (2, 4)

def test_grade_path_unchanged():
    bb = _backbone().eval()
    m = MultiTaskRETFound(bb).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(bb(x), m(x)[0], atol=1e-5)

def test_loss_is_finite_and_lambda_zero_equals_focal():
    focal = nn.CrossEntropyLoss()
    pw = torch.ones(4)
    gl = torch.randn(3, 4); gt = torch.tensor([0, 1, 2])
    fl = torch.randn(3, 4); ft = torch.randint(0, 2, (3, 4)).float()
    full = MultiTaskLoss(focal, pw, lam=0.5)(gl, gt, fl, ft)
    assert torch.isfinite(full)
    only = MultiTaskLoss(focal, pw, lam=0.0)(gl, gt, fl, ft)
    assert torch.allclose(only, focal(gl, gt))

def test_dataset_returns_triplet():
    arr = (np.random.rand(40, 40, 3) * 255).astype('uint8')
    p = '/tmp/_p4_test.png'; Image.fromarray(arr).save(p)
    tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()])
    ds = P4Dataset([(p, 2, [1, 0, 1, 0])], tf)
    img, g, fv = ds[0]
    assert img.shape == torch.Size([3, 224, 224])
    assert g == 2 and fv.tolist() == [1.0, 0.0, 1.0, 0.0] and fv.dtype == torch.float32

def test_pos_weight_rewards_rare_features():
    import pandas as pd
    df = pd.DataFrame({'haem': [1, 1, 0, 0], 'exud': [1, 0, 0, 0],
                       'cws': [0, 0, 0, 0], 'nvd': [1, 0, 0, 0]})
    pw = compute_feature_pos_weight(df)
    assert pw.shape == (4,)
    assert pw[1] > pw[0]   # exud rarer (1 pos) than haem (2 pos) -> higher weight
    assert (pw >= 1).all() # clamped

if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn(); print(f'PASS {fn.__name__}')
    print(f'\nAll {len(fns)} tests passed.')
