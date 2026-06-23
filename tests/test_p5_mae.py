import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from p5_mae import build_classifier, mae_get_depth, build_mae_llrd_optimizer


def test_forward_shape():
    m = build_classifier(num_classes=4)
    out = m(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 4), out.shape


def test_depth():
    nb = 24
    assert mae_get_depth('backbone.head.weight', nb) == 0
    assert mae_get_depth('backbone.fc_norm.weight', nb) == 1
    assert mae_get_depth('backbone.blocks.23.attn.qkv.weight', nb) == 2
    assert mae_get_depth('backbone.blocks.0.norm1.weight', nb) == 25
    assert mae_get_depth('backbone.patch_embed.proj.weight', nb) == 26
    assert mae_get_depth('backbone.pos_embed', nb) == 26


def test_llrd_groups():
    m = build_classifier(num_classes=4)
    opt = build_mae_llrd_optimizer(m, base_lr=5e-5, weight_decay=0.05, decay=0.75)
    seen = sum(len(g['params']) for g in opt.param_groups)
    n_train = sum(1 for p in m.parameters() if p.requires_grad)
    assert seen == n_train, (seen, n_train)
    fc = dict(m.named_parameters())['backbone.fc_norm.weight']
    for g in opt.param_groups:
        if any(p is fc for p in g['params']):
            assert g['weight_decay'] == 0.0
            assert abs(g['lr'] - 5e-5 * 0.75 ** 1) < 1e-12
            break
    else:
        raise AssertionError('fc_norm param not assigned to any group')


if __name__ == '__main__':
    test_forward_shape(); test_depth(); test_llrd_groups()
    print('ALL TESTS PASSED')
