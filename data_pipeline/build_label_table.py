"""
Build a clean per-eye label table from the raw Excel grades.

Decisions applied (confirmed by user):
  - Use ranking==1 rows only: these are the single authoritative final grade
    per patient (Arbitration when a dispute existed, Secondary otherwise).
  - Each EYE is one sample: creates two rows per patient (LE, RE).
  - LE folder = Left Eye grade columns; RE folder = Right Eye grade columns.
  - Use ALL images in each eye folder (they're multi-field CFP of the same eye).
  - Exclude U grades per task (not per eye globally: an eye can be valid for
    one model and excluded from the other).
  - R3S (1 row, stable proliferative) is excluded alongside U — Model A only
    targets R0/R1/R2/R3A; R3S is a distinct clinical category.

Output: labels/per_eye_labels.csv — one row per (patient × eye × image).
"""

import os
from pathlib import Path
import pandas as pd

ROOT    = Path(__file__).parent.parent
EXCEL   = ROOT / "Data/Homterton_Reading_Centre_Grades.xlsx"
IMG_DIR = ROOT / "Data/Diabetic Retinopathy IMAGES"
OUT_CSV = ROOT / "labels/per_eye_labels.csv"

# ── 1. Load and filter to the definitive grade row per patient ────────────────
df = pd.read_excel(EXCEL, sheet_name="Sheet1")
df = df[df["ranking"] == 1].copy()
assert len(df) == df["code"].nunique(), "Expected exactly one rank-1 row per patient"

# Strip the trailing '_T' from the code to get the image folder name
df["folder"] = df["code"].str.replace(r"_T$", "", regex=True)

# ── 2. Unpivot: one row per patient becomes two rows (LE + RE) ────────────────
# We reshape so downstream code works uniformly on (patient, eye) pairs.
records = []
for _, row in df.iterrows():
    for eye, side in [("LE", "Left Eye"), ("RE", "Right Eye")]:
        records.append({
            "code":             row["code"],
            "folder":           row["folder"],
            "eye":              eye,
            "retinopathy":      row[f"Retinopathy ({side})"],
            "maculopathy":      row[f"Maculopathy ({side})"],
            "image_quality":    row[f"Image quality ({side})"],
        })

eyes = pd.DataFrame(records)
total_eyes = len(eyes)

# ── 3. Link each eye to its image folder; collect image file paths ────────────
def list_images(folder, eye):
    """Return sorted list of image paths for one (patient, eye) pair."""
    eye_dir = os.path.join(IMG_DIR, folder, eye)
    if not os.path.isdir(eye_dir):
        return None          # folder itself is missing
    # Skip zero-byte files — some source files are empty placeholders that PIL
    # cannot open and would crash the DataLoader.
    files = sorted(
        f for f in os.listdir(eye_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        and not f.startswith("._")   # macOS resource fork sidecars — not images
        and os.path.getsize(os.path.join(eye_dir, f)) > 0
    )
    return files             # may be empty list

image_lists = [list_images(r.folder, r.eye) for r in eyes.itertuples()]
eyes["image_dir"]   = [
    os.path.join(IMG_DIR, r.folder, r.eye) if lst is not None else None
    for r, lst in zip(eyes.itertuples(), image_lists)
]
eyes["image_files"] = image_lists

# ── 4. Diagnostics: report coverage problems ──────────────────────────────────

# 4a. Patient folders in images/ that have NO row in the spreadsheet
all_img_folders = set(os.listdir(IMG_DIR))
spreadsheet_folders = set(eyes["folder"].unique())
folders_no_metadata = all_img_folders - spreadsheet_folders
if folders_no_metadata:
    print(f"[WARN] {len(folders_no_metadata)} image folder(s) have no spreadsheet entry:")
    for f in sorted(folders_no_metadata):
        print(f"       {f}")
else:
    print("[OK] Every image folder has a spreadsheet entry.")

# 4b. Spreadsheet rows whose patient folder doesn't exist at all
folder_missing = eyes[eyes["image_dir"].isna()]
if len(folder_missing):
    print(f"\n[WARN] {len(folder_missing)} eye(s) have no image folder:")
    print(folder_missing[["code", "folder", "eye"]].to_string(index=False))
else:
    print("[OK] Every spreadsheet eye has an image folder.")

# 4c. Folders that exist but are empty (no image files)
eyes_with_dir   = eyes[eyes["image_dir"].notna()].copy()
empty_dirs      = eyes_with_dir[eyes_with_dir["image_files"].apply(lambda x: len(x) == 0)]
nonempty_dirs   = eyes_with_dir[eyes_with_dir["image_files"].apply(lambda x: len(x) > 0)]
print(f"\n[INFO] Eye folders with 0 images : {len(empty_dirs)}")
print(f"[INFO] Eye folders with ≥1 image : {len(nonempty_dirs)}")

# ── 5. Explode to one row per image ──────────────────────────────────────────
# Each image gets the grade of its eye. All images from an eye stay together
# (same patient → same split later), so there is no leakage risk.
usable = nonempty_dirs.copy()
usable["image_path"] = usable.apply(
    lambda r: [os.path.join(r["image_dir"], f) for f in r["image_files"]], axis=1
)
usable_exploded = usable.explode("image_path").drop(columns=["image_files", "image_dir"])
usable_exploded = usable_exploded.reset_index(drop=True)

# ── 6. Report grade distributions (all usable eyes, before U exclusion) ───────
print("\n── Grade distributions for usable eyes (before U exclusion) ─────────")
for task, col in [("Model A (retinopathy)", "retinopathy"), ("Model B (maculopathy)", "maculopathy")]:
    counts = usable_exploded.drop_duplicates(["code", "eye"])[col].value_counts()
    print(f"\n  {task}:")
    for grade, n in counts.items():
        print(f"    {grade:>5}  {n:>5} eyes")

# ── 7. Flag R3S explicitly ────────────────────────────────────────────────────
r3s_count = (usable_exploded.drop_duplicates(["code", "eye"])["retinopathy"] == "R3S").sum()
if r3s_count:
    print(f"\n[NOTE] {r3s_count} eye(s) graded R3S (stable proliferative).")
    print("       R3S is NOT in Model A's 4 target classes (R0/R1/R2/R3A).")
    print("       These will be excluded alongside U grades.")

# ── 8. Save full table (U grades still present — downstream scripts filter) ───
usable_exploded.to_csv(OUT_CSV, index=False)
print(f"\n[SAVED] {OUT_CSV}  ({len(usable_exploded)} rows = images; "
      f"{usable_exploded.drop_duplicates(['code','eye']).shape[0]} unique eyes)")

print("\nDone. Next step: patient-level stratified split (build_splits.py).")
