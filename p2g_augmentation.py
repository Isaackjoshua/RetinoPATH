"""P2G — class-conditional augmentation for minority DR grades (R2, R3A).

The only novel piece of Phase 2G. R0/R1 images use the exact P2B timm
transform; R2/R3A images use a geometry-heavy, lesion-safe transform with
Random Erasing turned OFF (erasing can blank out the tiny microaneurysms /
exudates / neovascularisation that define the minority grades).

Kept importable so the routing is unit-tested independently of the
multi-hour GPU training in phase2g_minority_augmentation.ipynb.
"""
import argparse

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from util.datasets import build_transform

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MINORITY_LABELS = frozenset({2, 3})  # R2, R3A


def _p2b_aug_args(input_size):
    # Identical to phase2b_full_finetune.ipynb cell 4.
    return argparse.Namespace(
        input_size=input_size, color_jitter=None,
        aa="rand-m9-mstd0.5-inc1", reprob=0.25, remode="pixel", recount=1,
    )


def build_standard_train_transform(input_size=224):
    """R0/R1 train transform — timm RandAugment m9 + RandomErasing p=0.25."""
    return build_transform("train", _p2b_aug_args(input_size))


def build_eval_transform(input_size=224):
    """Val/test transform — deterministic resize + center crop (no aug)."""
    return build_transform("val", _p2b_aug_args(input_size))


def build_minority_train_transform(input_size=224):
    """R2/R3A train transform — geometry-heavy, lesion-safe, NO erasing."""
    return T.Compose([
        T.RandomResizedCrop(input_size, scale=(0.5, 1.0),
                            interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(20, interpolation=T.InterpolationMode.BICUBIC),
        T.ColorJitter(brightness=0.1, contrast=0.1),  # mild only
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        # NO RandomErasing — protects tiny grade-defining lesions.
    ])


class P2GDataset(Dataset):
    """Routes each image to a transform by label.

    records: list of (image_path, int_label).
    train=True  -> minority_tf for label in minority_labels, else standard_tf.
    train=False -> eval_tf for every label.
    """

    def __init__(self, records, standard_tf, minority_tf, eval_tf=None,
                 train=True, minority_labels=MINORITY_LABELS):
        self.records = records
        self.standard_tf = standard_tf
        self.minority_tf = minority_tf
        self.eval_tf = eval_tf
        self.train = train
        self.minority_labels = minority_labels

        if not self.train and self.eval_tf is None:
            raise ValueError("eval_tf is required when train=False (eval mode needs a transform)")

    def __len__(self):
        return len(self.records)

    def transform_for(self, label):
        if not self.train:
            return self.eval_tf
        return self.minority_tf if label in self.minority_labels else self.standard_tf

    def __getitem__(self, idx):
        path, label = self.records[idx]
        img = Image.open(path).convert("RGB")
        return self.transform_for(label)(img), label
