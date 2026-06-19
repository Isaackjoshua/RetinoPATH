"""
Build two symlinked ImageFolder trees, one per model.

Why symlinks (not copies)?  The raw images total several GB.  Symlinks let
PyTorch's ImageFolder walk a clean train/val/test/<class>/ structure without
duplicating disk space.  If an image is used in both Model A and Model B
(same physical file, different label trees), only one copy sits on disk.

Tree layout:
  image_trees/modelA/train/R0/  ...  /R3A/
  image_trees/modelA/val/R0/    ...
  image_trees/modelA/test/R0/   ...
  image_trees/modelB/train/M0/  /M1/
  image_trees/modelB/val/M0/    /M1/
  image_trees/modelB/test/M0/   /M1/

Symlink naming: <code>_<eye>_<original_filename> — keeps names unique
across patients while staying human-readable.
"""

import os
from pathlib import Path
import pandas as pd

ROOT       = Path(__file__).parent.parent
SPLITS_CSV = ROOT / "labels/splits.csv"
TREE_ROOT  = str(ROOT / "image_trees")

TASKS = {
    "modelA": {
        "grade_col":    "retinopathy",
        "valid_grades": ["R0", "R1", "R2", "R3A"],
    },
    "modelB": {
        "grade_col":    "maculopathy",
        "valid_grades": ["M0", "M1"],
    },
}
SPLITS = ["train", "val", "test"]

df = pd.read_csv(SPLITS_CSV)

for model, cfg in TASKS.items():
    col    = cfg["grade_col"]
    grades = cfg["valid_grades"]

    # Filter to rows valid for this model (no U, no R3S, correct quality already done)
    subset = df[df[col].isin(grades)].copy()

    print(f"\n── {model} ── {len(subset)} image rows")
    created = skipped = 0

    for split in SPLITS:
        split_rows = subset[subset["split"] == split]
        for _, row in split_rows.iterrows():
            grade   = row[col]
            src     = os.path.abspath(row["image_path"])
            fname   = f"{row['code']}_{row['eye']}_{os.path.basename(src)}"
            dst_dir = os.path.join(TREE_ROOT, model, split, grade)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, fname)

            if os.path.lexists(dst):
                skipped += 1
                continue
            if not os.path.exists(src):
                print(f"  [WARN] source missing: {src}")
                continue
            os.symlink(src, dst)
            created += 1

    print(f"  Symlinks created: {created}  |  already existed: {skipped}")

    # Per-class image counts per split (sanity check)
    print(f"  {'split':<6} " + "  ".join(f"{g:>6}" for g in grades))
    for split in SPLITS:
        counts = []
        for grade in grades:
            d = os.path.join(TREE_ROOT, model, split, grade)
            n = len(os.listdir(d)) if os.path.isdir(d) else 0
            counts.append(n)
        print(f"  {split:<6} " + "  ".join(f"{n:>6}" for n in counts))

print("\nDone.  Directories ready for ImageFolder:")
for model in TASKS:
    print(f"  {os.path.abspath(os.path.join(TREE_ROOT, model))}")
