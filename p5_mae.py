"""P5 — RETFound-MAE backbone (ViT-L/16, global-pooled) for the P2B pipeline.

Drop-in replacement for the DINOv2 loader. Kept importable so the forward
contract and LLRD grouping are unit-tested off-GPU (no weights download).
"""
import torch
import torch.nn as nn

import models_vit
from timm.layers import trunc_normal_


class MAEClassifier(nn.Module):
    """Wrap the RETFound-MAE ViT so forward() returns (B, num_classes) logits.

    models_vit.VisionTransformer.forward_features (global_pool) returns a
    (B, 1, embed) tensor already passed through fc_norm; we squeeze the token
    dim and apply the classifier head directly (bypassing timm's forward_head,
    which expects a string global_pool flag).
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        if feats.dim() == 3:
            feats = feats[:, 0]
        return self.backbone.head(feats)

    def set_grad_checkpointing(self, enable=True):
        # No-op: the custom forward_features does not honor timm checkpointing.
        pass


def build_mae_backbone(num_classes=4, img_size=224, drop_path_rate=0.2, global_pool=True):
    return models_vit.RETFound_mae(
        img_size=img_size, num_classes=num_classes,
        drop_path_rate=drop_path_rate, global_pool=global_pool)


def build_classifier(num_classes=4, img_size=224, drop_path_rate=0.2):
    return MAEClassifier(build_mae_backbone(num_classes, img_size, drop_path_rate))


def load_pretrained_mae_(backbone, repo_id, filename, device='cpu'):
    """In-place load of RETFound-MAE weights, mirroring main_finetune.py.

    Loaded with weights_only=True (no arbitrary unpickling / code execution).
    RETFound MAE checkpoints wrap the state_dict alongside an argparse.Namespace
    of training args, so that one benign type is allowlisted for the safe loader.
    If a checkpoint needs another global, torch raises naming it — allowlist it
    explicitly rather than disabling weights_only.
    """
    import argparse
    from huggingface_hub import hf_hub_download
    from util.pos_embed import interpolate_pos_embed
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    with torch.serialization.safe_globals([argparse.Namespace]):
        ck = torch.load(path, map_location='cpu', weights_only=True)
    cm = ck['model'] if 'model' in ck else ck
    cm = {k.replace('backbone.', ''): v for k, v in cm.items()}
    cm = {k.replace('mlp.w12.', 'mlp.fc1.'): v for k, v in cm.items()}
    cm = {k.replace('mlp.w3.', 'mlp.fc2.'): v for k, v in cm.items()}
    sd = backbone.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in cm and cm[k].shape != sd[k].shape:
            del cm[k]
    interpolate_pos_embed(backbone, cm)
    backbone.load_state_dict(cm, strict=False)
    trunc_normal_(backbone.head.weight, std=2e-5)
    nn.init.zeros_(backbone.head.bias)
    return backbone


def mae_get_depth(name, num_blocks):
    """LLRD depth from the head. fc_norm -> 1 (MAE deletes model.norm)."""
    name = name.replace('backbone.', '')
    if 'head' in name:
        return 0
    if 'fc_norm' in name or name.startswith('norm'):
        return 1
    if 'blocks.' in name:
        return num_blocks - int(name.split('blocks.')[1].split('.')[0]) + 1
    return num_blocks + 2


def build_mae_llrd_optimizer(model, base_lr, weight_decay, decay=0.75):
    num_blocks = len(model.backbone.blocks)

    def no_decay(n):
        return any(t in n for t in ['bias', 'norm', 'cls_token', 'pos_embed'])

    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        key = (mae_get_depth(name, num_blocks), no_decay(name))
        groups.setdefault(key, []).append(param)

    param_groups = []
    for (depth, nd), params in sorted(groups.items()):
        lr = base_lr * (decay ** depth)
        param_groups.append({'params': params, 'initial_lr': lr, 'lr': lr,
                             'weight_decay': 0.0 if nd else weight_decay})
    return torch.optim.AdamW(param_groups)
