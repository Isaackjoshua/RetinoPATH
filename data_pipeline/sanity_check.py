"""
Visual sanity check: sample a few images from each class and save a labelled
grid so you can eyeball that:
  1. The images are colour fundus photos (CFP) — round, orange/red, retina visible.
  2. Labels look clinically plausible (R0 = clean; R3A = heavy lesions).

Saves: labels/sanity_check_modelA.png and labels/sanity_check_modelB.png
Open them with any image viewer.
"""

import os
from pathlib import Path
import random
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT       = Path(__file__).parent.parent
SPLITS_CSV = ROOT / "labels/splits.csv"
TREE_ROOT  = str(ROOT / "image_trees")
THUMB_W    = 224   # thumbnail width
THUMB_H    = 224
COLS       = 4     # images per class row
FONT_SIZE  = 14
SEED       = 42
random.seed(SEED)

def pick_samples(model_dir, grades, split="train", n=COLS):
    """Return n random image paths per grade from the symlinked tree."""
    samples = {}
    for grade in grades:
        folder = os.path.join(model_dir, split, grade)
        if not os.path.isdir(folder):
            samples[grade] = []
            continue
        files = os.listdir(folder)
        picked = random.sample(files, min(n, len(files)))
        samples[grade] = [os.path.join(folder, f) for f in picked]
    return samples

def make_grid(samples, grades, title, out_path):
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font = small_font = ImageFont.load_default()

    LABEL_H = 30
    n_rows  = len(grades)
    n_cols  = COLS
    w = n_cols * THUMB_W
    h = n_rows * (THUMB_H + LABEL_H) + 40  # 40px title bar

    canvas = Image.new("RGB", (w, h), (30, 30, 30))
    draw   = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill=(255, 255, 255), font=font)

    for row_i, grade in enumerate(grades):
        y_top = 40 + row_i * (THUMB_H + LABEL_H)
        paths = samples.get(grade, [])
        for col_i in range(n_cols):
            x_left = col_i * THUMB_W
            if col_i >= len(paths):
                # grey placeholder
                placeholder = Image.new("RGB", (THUMB_W, THUMB_H), (60, 60, 60))
                canvas.paste(placeholder, (x_left, y_top))
            else:
                try:
                    img = Image.open(paths[col_i]).convert("RGB")
                    img.thumbnail((THUMB_W, THUMB_H))
                    # Centre in cell
                    offset_x = (THUMB_W - img.width) // 2
                    offset_y = (THUMB_H - img.height) // 2
                    canvas.paste(img, (x_left + offset_x, y_top + offset_y))
                    # Tiny filename label (patient code + eye extracted from symlink name)
                    fname = os.path.basename(paths[col_i])
                    parts = fname.split("_")
                    label_txt = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else fname[:12]
                    draw.text((x_left + 3, y_top + THUMB_H - 14),
                              label_txt, fill=(255, 255, 0), font=small_font)
                except Exception as e:
                    print(f"  [WARN] Could not open {paths[col_i]}: {e}")

            # Grade label on left of each row
            if col_i == 0:
                draw.rectangle([x_left, y_top + THUMB_H, x_left + THUMB_W, y_top + THUMB_H + LABEL_H],
                               fill=(20, 60, 100))
                draw.text((x_left + 5, y_top + THUMB_H + 6), f"Grade: {grade}",
                          fill=(200, 220, 255), font=font)

    canvas.save(out_path)
    print(f"[SAVED] {out_path}")

# ── Model A ───────────────────────────────────────────────────────────────────
modelA_dir = os.path.join(TREE_ROOT, "modelA")
grades_A   = ["R0", "R1", "R2", "R3A"]
samples_A  = pick_samples(modelA_dir, grades_A)
make_grid(samples_A, grades_A,
          "Model A — Retinopathy (train set, 4 images per class)",
          str(ROOT / "figures/sanity_check_modelA.png"))

# ── Model B ───────────────────────────────────────────────────────────────────
modelB_dir = os.path.join(TREE_ROOT, "modelB")
grades_B   = ["M0", "M1"]
samples_B  = pick_samples(modelB_dir, grades_B)
make_grid(samples_B, grades_B,
          "Model B — Maculopathy (train set, 4 images per class)",
          str(ROOT / "figures/sanity_check_modelB.png"))

# ── Final per-class counts ────────────────────────────────────────────────────
print("\nFinal image counts in tree:")
for model, grades in [("modelA", grades_A), ("modelB", grades_B)]:
    print(f"\n  {model}:")
    print(f"  {'split':<6} " + "  ".join(f"{g:>6}" for g in grades) + "   total")
    for split in ["train", "val", "test"]:
        counts = []
        for grade in grades:
            d = os.path.join(TREE_ROOT, model, split, grade)
            n = len(os.listdir(d)) if os.path.isdir(d) else 0
            counts.append(n)
        print(f"  {split:<6} " + "  ".join(f"{n:>6}" for n in counts) +
              f"   {sum(counts):>6}")
