"""
Patient-level stratified 70/15/15 train/val/test split.

Why patient-level?  A patient's LE and RE images are correlated — same disease
progression, same lighting conditions, same screening session.  If we split at
the image or eye level, the same patient's images could appear in both train and
test, causing data leakage that inflates test metrics.

Why 70/15/15?  We need val for early stopping / hyperparameter choice, and test
for final unbiased evaluation.  15% each gives ~360 patients per held-out set,
with ~25 R3A eyes in test — enough for a stable AUROC on the rarest class.

Why stratify?  R3A (165 eyes) and R2 (211 eyes) are rare.  A random split might
put most of them in train and leave test with too few to evaluate reliably.
Stratification ensures every split gets a proportional slice of each class.

Stratification key: (worst_retinopathy_severity, has_any_M1) per patient —
this preserves the rarest grade combinations across splits.

Exclusions applied here (not in build_label_table.py, so the raw CSV stays clean):
  - Inadequate image quality eyes
  - U retinopathy (per Model A) and U maculopathy (per Model B)
  - R3S (only 1 eye; not a target class)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT      = Path(__file__).parent.parent
LABEL_CSV = ROOT / "labels/per_eye_labels.csv"
OUT_CSV   = ROOT / "labels/splits.csv"

RETINOPATHY_ORDER = {"R0": 0, "R1": 1, "R2": 2, "R3A": 3}  # severity ranking
SEED = 42  # fix for reproducibility

# ── 1. Load and apply quality filter ─────────────────────────────────────────
df = pd.read_csv(LABEL_CSV)

# Drop Inadequate eyes entirely — decision confirmed by user.
before = len(df["code"].drop_duplicates())  # patient count before
df = df[df["image_quality"] == "Adequate"].copy()
after_q = len(df["code"].drop_duplicates())
print(f"[FILTER] Inadequate quality: {before - after_q} patients lost "
      f"(some eyes dropped, patient kept if other eye is Adequate)")

# Keep only Model A target retinopathy grades: drop U (ungradable) and R3S
# (stable proliferative — not one of R0/R1/R2/R3A). The docstring above always
# claimed this exclusion, but the original script only applied it in reporting,
# not to the saved frame; applying it here makes splits.csv reproduce the
# intended R0/R1/R2/R3A cohort exactly. A patient is kept if >=1 eye survives.
before_g = len(df["code"].drop_duplicates())
df = df[df["retinopathy"].isin(["R0", "R1", "R2", "R3A"])].copy()
after_g = len(df["code"].drop_duplicates())
print(f"[FILTER] Non-target retinopathy (U/R3S): {before_g - after_g} patients lost")

# ── 2. Build a per-patient stratum for stratified splitting ───────────────────
# We want to preserve rare class combinations. Strategy:
#   - worst_dr: highest retinopathy severity across usable (non-U, non-R3S) eyes
#   - has_m1:   True if any Adequate eye has M1 (non-U)
# Patients whose ALL Adequate eyes are U for retinopathy still participate
# (they contribute to Model B), so worst_dr → "U_only" for them.

per_patient = (
    df.drop_duplicates(["code", "eye"])
      .groupby("code")
      .agg(
          worst_dr  = ("retinopathy",
                       lambda g: max(
                           (RETINOPATHY_ORDER.get(v, -1) for v in g),
                           default=-1
                       )),
          has_m1    = ("maculopathy", lambda g: (g == "M1").any()),
      )
      .reset_index()
)

# Convert numeric severity back to label (or "U_only" if no valid DR grade)
per_patient["dr_label"] = per_patient["worst_dr"].map(
    {v: k for k, v in RETINOPATHY_ORDER.items()}
).fillna("U_only")

per_patient["stratum"] = (
    per_patient["dr_label"] + "_M" + per_patient["has_m1"].astype(int).astype(str)
)

print("\nPatient stratum distribution before split:")
print(per_patient["stratum"].value_counts().to_string())

# ── 3. Merge tiny strata so sklearn doesn't error on <2-sample classes ────────
# Any stratum with fewer than 7 patients can't be split 70/15/15 cleanly.
# We merge rare combinations into a catch-all "rare" bucket.
STRAT_MIN = 7
counts = per_patient["stratum"].value_counts()
small  = counts[counts < STRAT_MIN].index.tolist()
if small:
    print(f"\n[MERGE] Strata with <{STRAT_MIN} patients → 'rare': {small}")
    per_patient["stratum"] = per_patient["stratum"].replace(
        {s: "rare" for s in small}
    )

# ── 4. Patient-level stratified split ─────────────────────────────────────────
# Two-step: first cut off 30% (val+test), then split that 50/50 → 15/15.
patients     = per_patient["code"].values
strata       = per_patient["stratum"].values

train_pts, valtest_pts, _, valtest_strat = train_test_split(
    patients, strata,
    test_size=0.30, random_state=SEED, stratify=strata
)
val_pts, test_pts = train_test_split(
    valtest_pts,
    test_size=0.50, random_state=SEED, stratify=valtest_strat
)

split_map = (
    {p: "train" for p in train_pts} |
    {p: "val"   for p in val_pts}   |
    {p: "test"  for p in test_pts}
)

# ── 5. Assert disjoint ────────────────────────────────────────────────────────
assert set(train_pts).isdisjoint(val_pts),  "LEAK: train ∩ val non-empty"
assert set(train_pts).isdisjoint(test_pts), "LEAK: train ∩ test non-empty"
assert set(val_pts).isdisjoint(test_pts),   "LEAK: val ∩ test non-empty"
print(f"\n[OK] Disjoint check passed — no patient appears in more than one split.")
print(f"     Train: {len(train_pts)} patients | "
      f"Val: {len(val_pts)} patients | Test: {len(test_pts)} patients")

# ── 6. Attach split labels to image rows ──────────────────────────────────────
df["split"] = df["code"].map(split_map)

# ── 7. Print per-class counts per split, per task ────────────────────────────
def report_task(df, grade_col, valid_grades, task_name):
    subset = df[df[grade_col].isin(valid_grades)].drop_duplicates(["code", "eye"])
    print(f"\n── {task_name} ── (eyes, after U/quality exclusion)")
    pivot = (
        subset.groupby(["split", grade_col])
              .size()
              .unstack(fill_value=0)
              .reindex(["train", "val", "test"])
    )
    print(pivot.to_string())
    total = subset.groupby("split").size().reindex(["train", "val", "test"])
    print(f"  Total: {total.to_dict()}")

report_task(df, "retinopathy", ["R0","R1","R2","R3A"], "Model A — Retinopathy (4-class)")
report_task(df, "maculopathy", ["M0","M1"],            "Model B — Maculopathy (binary)")

# ── 8. Save ───────────────────────────────────────────────────────────────────
df.to_csv(OUT_CSV, index=False)
print(f"\n[SAVED] {OUT_CSV}  ({len(df)} image rows with split column)")
print("Next step: build_image_trees.py — create symlinked ImageFolder directories.")
