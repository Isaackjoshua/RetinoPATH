"""P4 — lesion-feature multi-task: shared-backbone model, loss, dataset.

The grade path is untouched (backbone.head); a parallel Linear head predicts
4 binary lesion features (haem/exud/cws/nvd) from the same pooled embedding.
Kept importable so the wiring is unit-tested off-GPU.
"""
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

FEATURE_NAMES = ['haem', 'exud', 'cws', 'nvd']


class MultiTaskRETFound(nn.Module):
    """Wrap a timm ViT (with a grade head) and add a lesion-feature head that
    shares the pooled pre-logits embedding."""

    def __init__(self, backbone, n_features=4):
        super().__init__()
        self.backbone = backbone
        self.feature_head = nn.Linear(backbone.num_features, n_features)

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        pooled = self.backbone.forward_head(feats, pre_logits=True)
        return self.backbone.head(pooled), self.feature_head(pooled)

    def set_grad_checkpointing(self, enable=True):
        self.backbone.set_grad_checkpointing(enable)


def build_multitask_model(backbone, n_features=4):
    return MultiTaskRETFound(backbone, n_features)


class MultiTaskLoss(nn.Module):
    """L = focal(grade) + lam * BCEWithLogits(features, pos_weight)."""

    def __init__(self, focal, feature_pos_weight, lam=0.5):
        super().__init__()
        self.focal = focal
        self.bce = nn.BCEWithLogitsLoss(pos_weight=feature_pos_weight)
        self.lam = lam

    def forward(self, grade_logits, grade_targets, feat_logits, feat_targets):
        return self.focal(grade_logits, grade_targets) + self.lam * self.bce(feat_logits, feat_targets)


class P4Dataset(Dataset):
    """records: list of (image_path, grade_int, feat_list[4 floats])."""

    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, grade, feats = self.records[idx]
        img = self.transform(Image.open(path).convert('RGB'))
        return img, grade, torch.tensor(feats, dtype=torch.float32)


def make_records_mt(df_subset):
    return [(r.image_path, r.grade_int, [getattr(r, f) for f in FEATURE_NAMES])
            for r in df_subset.itertuples()]


def compute_feature_pos_weight(df):
    """neg/pos per feature on the given split, clamped to >=1 (rare feature => higher weight)."""
    w = []
    for f in FEATURE_NAMES:
        pos = float(df[f].sum()); neg = float(len(df) - pos)
        w.append(max(1.0, neg / pos) if pos > 0 else 1.0)
    return torch.tensor(w, dtype=torch.float)
