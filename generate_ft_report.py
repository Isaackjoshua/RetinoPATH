"""
Detailed PDF report for Model A — Full Fine-Tune.
Output: labels/modelA_ft_report.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from PIL import Image
import pandas as pd

ROOT      = os.path.dirname(__file__)
LOG_DIR   = os.path.join(ROOT, "output_logs/retfound_dinov2_modelA_finetune/retfound_dinov2_modelA_finetune")
VAL_CSV   = os.path.join(ROOT, "output_dir/retfound_dinov2_modelA_finetune/retfound_dinov2_modelA_finetune/metrics_val.csv")
TEST_CSV  = os.path.join(ROOT, "output_dir/retfound_dinov2_modelA_finetune/retfound_dinov2_modelA_finetune/metrics_test.csv")
CONF_MAT  = os.path.join(ROOT, "output_dir/retfound_dinov2_modelA_finetune/retfound_dinov2_modelA_finetune/confusion_matrix_test.jpg")
SPLITS    = os.path.join(ROOT, "labels/splits.csv")
OUT_PDF   = os.path.join(ROOT, "labels/modelA_ft_report.pdf")

# ── Palette ────────────────────────────────────────────────────────────────────
BG    = "#F7F9FC"
PANEL = "#FFFFFF"
NAVY  = "#1A2B4A"
ORANGE= "#E07B39"
ORG2  = "#F5A05A"
GRAY  = "#6B7C93"
LGRAY = "#FDF3EC"
DKGRAY= "#3D4F66"
GREEN = "#2E8B57"
TEAL  = "#1B7B8A"
RED   = "#C0392B"
CLASSES = ["R0", "R1", "R2", "R3A"]
CLASS_C = ["#4A90D9", "#5BB85D", "#F0A030", "#D9534F"]

# ── Load data ──────────────────────────────────────────────────────────────────
ea = EventAccumulator(LOG_DIR); ea.Reload()

def curve(tag):
    seen = {}
    for e in ea.Scalars(tag):
        seen[e.step] = e.value
    steps = sorted(seen)
    return np.array(steps, dtype=float), np.array([seen[s] for s in steps])

ep_lr, lr   = curve("lr")
ep_lt, lt   = curve("loss/train")

# Val metrics from CSV (cleaner; drop the 2 duplicate init rows)
val_df      = pd.read_csv(VAL_CSV).iloc[2:].reset_index(drop=True)
ep          = np.arange(len(val_df))
auroc       = val_df["roc_auc"].values
loss_v      = val_df["val_loss"].values
acc         = val_df["accuracy"].values
kappa       = val_df["kappa"].values
f1          = val_df["f1"].values
prec        = val_df["precision"].values
rec         = val_df["recall"].values
ap          = val_df["average_precision"].values

test        = pd.read_csv(TEST_CSV).iloc[0].to_dict()
t_auroc     = float(test["roc_auc"])
t_acc       = float(test["accuracy"])
t_kappa     = float(test["kappa"])
t_f1        = float(test["f1"])
t_prec      = float(test["precision"])
t_rec       = float(test["recall"])
t_ap        = float(test["average_precision"])
t_loss      = float(test["val_loss"])

best_i      = int(auroc.argmax())
best_epoch  = int(ep[best_i])
best_auroc  = float(auroc[best_i])

# Dataset
df_sp  = pd.read_csv(SPLITS)
mA     = df_sp[df_sp["retinopathy"].isin(CLASSES)]
splt   = {}
for sp in ["train", "val", "test"]:
    vc = mA[mA["split"] == sp]["retinopathy"].value_counts()
    splt[sp] = {c: int(vc.get(c, 0)) for c in CLASSES}

LP_BEST_AUROC = 0.8401   # from linear probe best checkpoint

# ── Helpers ────────────────────────────────────────────────────────────────────
def page_bg(fig):
    fig.patch.set_facecolor(BG)

def ax_style(ax, xlabel="Epoch", ylabel="", ylim=None, title="", tc=ORANGE):
    ax.set_facecolor(PANEL)
    ax.grid(True, linestyle="--", lw=0.5, color="#D0DDE8", alpha=0.9)
    ax.set_xlabel(xlabel, fontsize=9, color=GRAY)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9, color=GRAY)
    ax.tick_params(colors=GRAY, labelsize=8)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_color("#C8D8E8")
    if ylim: ax.set_ylim(ylim)
    if title: ax.set_title(title, fontsize=10, color=tc, fontweight="bold", pad=7)

def banner(fig, text, sub="", y=0.92, h=0.08, bg=NAVY):
    ax = fig.add_axes([0.0, y, 1.0, h])
    ax.set_facecolor(bg); ax.axis("off")
    ax.text(0.5, 0.65, text, ha="center", va="center", fontsize=14,
            fontweight="bold", color="white", transform=ax.transAxes)
    if sub:
        ax.text(0.5, 0.22, sub, ha="center", va="center", fontsize=9,
                color="#F5C8A0", transform=ax.transAxes)

def footer(fig, page, total=4):
    fig.text(0.5, 0.016,
             f"RETFound · Model A · Full Fine-Tune  ·  Homerton Reading Centre Data  ·  Page {page} of {total}",
             ha="center", fontsize=7.5, color=GRAY)

def stat_box(ax, x, y, w, h, label, value, bg=LGRAY, vc=ORANGE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                                facecolor=bg, edgecolor=ORANGE, lw=1.5,
                                transform=ax.transAxes, clip_on=False))
    ax.text(x + w/2, y + h * 0.65, value, ha="center", va="center",
            fontsize=19, fontweight="bold", color=vc, transform=ax.transAxes)
    ax.text(x + w/2, y + h * 0.22, label, ha="center", va="center",
            fontsize=8.5, color=GRAY, transform=ax.transAxes)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview & Setup
# ══════════════════════════════════════════════════════════════════════════════
def page1(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    page_bg(fig)

    banner(fig,
           "Model A  ·  Diabetic Retinopathy  ·  Full Fine-Tune",
           "All RETFound-DINOv2 layers updated  ·  Layer-wise LR decay (0.65)  ·  50 epochs")

    # ── Concept box ───────────────────────────────────────────────────────────
    ax_c = fig.add_axes([0.03, 0.70, 0.45, 0.20])
    ax_c.set_facecolor(LGRAY); ax_c.axis("off")
    ax_c.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                                  facecolor=LGRAY, edgecolor=ORANGE, lw=1.5,
                                  transform=ax_c.transAxes, clip_on=False))
    ax_c.text(0.04, 0.88, "What is Full Fine-Tuning?", fontsize=11,
              fontweight="bold", color=NAVY, transform=ax_c.transAxes, va="top")
    concept = (
        "All ~307M backbone parameters are updated alongside the classification head.\n"
        "A layer-wise LR decay (factor 0.65 per depth level) prevents destroying the\n"
        "general retinal features learned during pretraining: the deepest layers use\n"
        "the full LR while the earliest layers receive LR × 0.65²³ ≈ 0.02% of full LR.\n"
        "This adapts high-level representations to diabetic retinopathy grading while\n"
        "preserving low-level feature detectors that transfer across retinal tasks."
    )
    ax_c.text(0.04, 0.68, concept, fontsize=8.8, color=DKGRAY,
              transform=ax_c.transAxes, va="top", linespacing=1.6)

    # ── Key result stats ──────────────────────────────────────────────────────
    ax_s = fig.add_axes([0.52, 0.70, 0.45, 0.20])
    ax_s.set_facecolor(BG); ax_s.axis("off")
    stat_box(ax_s, 0.01, 0.08, 0.30, 0.84, "Test AUROC",    f"{t_auroc:.3f}",  vc=GREEN)
    stat_box(ax_s, 0.35, 0.08, 0.30, 0.84, "Test Accuracy", f"{t_acc:.1%}",    vc=ORANGE)
    stat_box(ax_s, 0.69, 0.08, 0.30, 0.84, "Test κ",        f"{t_kappa:.3f}",  vc=ORANGE)

    # ── Training config ───────────────────────────────────────────────────────
    ax_t = fig.add_axes([0.03, 0.38, 0.44, 0.28])
    ax_t.axis("off")
    ax_t.text(0, 1.04, "Training Configuration", fontsize=11, fontweight="bold",
              color=NAVY, transform=ax_t.transAxes)

    config = [
        ["Backbone",       "RETFound-DINOv2 ViT-Large"],
        ["Backbone status","ALL layers updated (full fine-tune)"],
        ["LR decay",       "Layer-wise × 0.65 per depth  (24 ViT blocks)"],
        ["Deepest LR",     "Full base LR  (classification head)"],
        ["Shallowest LR",  "Base LR × 0.65²³ ≈ 0.0002×  (patch embed)"],
        ["Epochs",         "50"],
        ["Batch size",     "24  (ViT-L gradient memory requires smaller batch)"],
        ["Input size",     "224 × 224 px"],
        ["Loss",           "CrossEntropyLoss with class weights"],
        ["Class weights",  "R0=1.00 · R1=1.79 · R2=9.53 · R3A=15.68"],
        ["Selection",      "Best val AUROC checkpoint → test evaluation"],
        ["Best ckpt epoch",f"{best_epoch}  (val AUROC {best_auroc:.3f})"],
    ]

    tbl = ax_t.table(cellText=config, cellLoc="left", loc="center",
                     bbox=[0.0, 0.0, 1.0, 0.94])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D8E4EE")
        cell.set_facecolor("#FBF3EC" if r % 2 == 0 else PANEL)
        if c == 0: cell.set_text_props(color=ORANGE, fontweight="bold")
        else:      cell.set_text_props(color=DKGRAY)

    # ── Layer-wise LR visualisation ───────────────────────────────────────────
    ax_lr = fig.add_axes([0.52, 0.38, 0.44, 0.28])
    ax_lr.set_facecolor(PANEL)
    ax_lr.text(0, 1.04, "Layer-wise LR Decay  (relative to head LR = 1.0)",
               fontsize=10, fontweight="bold", color=NAVY, transform=ax_lr.transAxes)

    n_layers = 25
    layer_names = ["Patch\nembed"] + [f"Block\n{i}" for i in range(1, 24)] + ["Head"]
    lrs = [0.65 ** (n_layers - 1 - i) for i in range(n_layers)]
    colors_lr = [plt.cm.YlOrRd(v) for v in lrs]
    bars = ax_lr.barh(range(n_layers), lrs, color=colors_lr,
                      edgecolor="white", lw=0.4, height=0.75)
    ax_lr.set_yticks([0, 6, 12, 18, 23, 24])
    ax_lr.set_yticklabels(["Patch embed", "Block 6", "Block 12",
                            "Block 18", "Block 23", "Head"],
                          fontsize=7.5, color=DKGRAY)
    ax_lr.set_xlabel("Relative LR multiplier", fontsize=9, color=GRAY)
    ax_lr.tick_params(colors=GRAY, labelsize=7.5)
    for s in ["top", "right"]: ax_lr.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax_lr.spines[s].set_color("#C8D8E8")
    ax_lr.grid(axis="x", linestyle="--", lw=0.5, color="#D0DDE8", alpha=0.8)
    ax_lr.set_xlim(0, 1.15)
    ax_lr.text(lrs[-1] + 0.02, 24, "1.000", fontsize=7.5, color=ORANGE,
               va="center", fontweight="bold")
    ax_lr.text(lrs[0] + 0.001, 0, f"{lrs[0]:.4f}", fontsize=7.5, color=GRAY, va="center")

    # ── LP vs FT comparison strip ─────────────────────────────────────────────
    ax_cmp = fig.add_axes([0.03, 0.09, 0.92, 0.24])
    ax_cmp.set_facecolor(PANEL); ax_cmp.axis("off")
    ax_cmp.text(0, 1.06, "Comparison: Linear Probe  →  Full Fine-Tune", fontsize=11,
                fontweight="bold", color=NAVY, transform=ax_cmp.transAxes)

    metrics   = ["AUROC", "Accuracy", "Kappa", "F1", "Precision", "Recall", "Avg Precision"]
    lp_vals   = [LP_BEST_AUROC, 0.702, 0.457, None, None, None, None]
    ft_vals   = [t_auroc, t_acc, t_kappa, t_f1, t_prec, t_rec, t_ap]
    x_pos     = np.linspace(0.07, 0.93, len(metrics))

    for xi, m, lv, fv in zip(x_pos, metrics, lp_vals, ft_vals):
        ax_cmp.text(xi, 0.96, m, ha="center", va="top", fontsize=8.5,
                    color=GRAY, transform=ax_cmp.transAxes)
        if lv is not None:
            ax_cmp.text(xi, 0.72, f"{lv:.3f}", ha="center", va="top", fontsize=11,
                        color=TEAL, fontweight="bold", transform=ax_cmp.transAxes)
            delta = fv - lv
            ax_cmp.text(xi, 0.43, f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}",
                        ha="center", va="top", fontsize=9,
                        color=GREEN if delta > 0 else RED, fontweight="bold",
                        transform=ax_cmp.transAxes)
        else:
            ax_cmp.text(xi, 0.72, "—", ha="center", va="top", fontsize=11,
                        color=GRAY, transform=ax_cmp.transAxes)
            ax_cmp.text(xi, 0.43, "—", ha="center", va="top", fontsize=9,
                        color=GRAY, transform=ax_cmp.transAxes)
        ax_cmp.text(xi, 0.18, f"{fv:.3f}", ha="center", va="top", fontsize=11,
                    color=ORANGE, fontweight="bold", transform=ax_cmp.transAxes)

    ax_cmp.text(0.01, 0.72, "LP val:", ha="left", va="top", fontsize=8.5,
                color=TEAL, transform=ax_cmp.transAxes)
    ax_cmp.text(0.01, 0.43, "Δ:", ha="left", va="top", fontsize=8.5,
                color=GREEN, transform=ax_cmp.transAxes)
    ax_cmp.text(0.01, 0.18, "FT test:", ha="left", va="top", fontsize=8.5,
                color=ORANGE, transform=ax_cmp.transAxes)
    for y_l in [0.87, 0.60, 0.34, 0.09]:
        ax_cmp.plot([0, 1], [y_l, y_l], color="#D8E4EE", lw=0.8,
                    transform=ax_cmp.transAxes, clip_on=False)

    footer(fig, 1)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Training Curves
# ══════════════════════════════════════════════════════════════════════════════
def page2(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    page_bg(fig)
    banner(fig, "Training Dynamics  —  50 Epochs of Full Fine-Tuning",
           "All metrics evaluated on the held-out validation set after each epoch")

    def plot_curve(rect, x, y, ylabel, ylim, title,
                   ref=None, ref_label=None, ref_color=GREEN,
                   test_val=None, test_label=None):
        ax = fig.add_axes(rect)
        ax_style(ax, ylabel=ylabel, ylim=ylim, title=title, tc=ORANGE)
        ax.plot(x, y, color=ORANGE, lw=2)
        ax.fill_between(x, y, alpha=0.13, color=ORANGE)

        bi = int(y.argmax()) if "Loss" not in ylabel else int(y.argmin())
        marker_val = float(y[bi])
        ax.scatter(x[bi], marker_val, color=ORANGE, s=60, zorder=5)
        tag = "Best" if "Loss" not in ylabel else "Min"
        ax.annotate(f" {tag}: {marker_val:.3f}  (ep {int(x[bi])})",
                    xy=(x[bi], marker_val), fontsize=7.8, color=ORANGE)
        ax.axvline(x[bi], color=ORANGE, lw=0.8, ls=":", alpha=0.5)

        if ref is not None:
            ax.axhline(ref, color=TEAL, lw=1.2, ls="--", alpha=0.75, label=ref_label)
            ax.legend(fontsize=7.5, framealpha=0.8, edgecolor="#D0DDE8", loc="lower right")
        if test_val is not None:
            ax.axhline(test_val, color=GREEN, lw=1.5, ls="-.", alpha=0.85)
            ax.text(x[-1] * 0.6, test_val + (ylim[1] - ylim[0]) * 0.02,
                    f"{test_label}: {test_val:.3f}", fontsize=7.8, color=GREEN)

    # Row 1: AUROC and Loss
    plot_curve([0.07, 0.59, 0.40, 0.27], ep, auroc, "Val AUROC",
               (0.76, 0.95), "Val AUROC per Epoch",
               ref=LP_BEST_AUROC, ref_label=f"LP best val {LP_BEST_AUROC:.3f}",
               test_val=t_auroc, test_label="Test AUROC")

    ax_lt = fig.add_axes([0.57, 0.59, 0.40, 0.27])
    ax_style(ax_lt, ylabel="Loss", title="Train Loss vs Val Loss", tc=ORANGE)
    # Align train loss steps to match epochs (train steps are per-batch)
    # Downsample train loss: take last step per epoch
    max_ep = int(ep_lt[-1])
    train_ep, train_loss_per_epoch = [], []
    for e in range(len(ep)):
        lo = int(ep_lt[ep_lt >= (e * (max_ep / len(ep)))][0]) if e > 0 else 0
        hi_mask = ep_lt < ((e + 1) * (max_ep / len(ep)))
        if hi_mask.any():
            train_epoch_loss = lt[hi_mask].mean()
            train_ep.append(e)
            train_loss_per_epoch.append(train_epoch_loss)
    ax_lt.plot(train_ep, train_loss_per_epoch, color=DKGRAY, lw=1.8, label="Train loss", alpha=0.7)
    ax_lt.plot(ep, loss_v, color=ORANGE, lw=2, label="Val loss")
    ax_lt.fill_between(ep, loss_v, alpha=0.10, color=ORANGE)
    ax_lt.legend(fontsize=8, framealpha=0.8, edgecolor="#D0DDE8")

    # Row 2: Accuracy and Kappa
    plot_curve([0.07, 0.34, 0.40, 0.21], ep, acc, "Val Accuracy",
               (0.60, 0.85), "Val Accuracy per Epoch",
               test_val=t_acc, test_label="Test acc")

    plot_curve([0.57, 0.34, 0.40, 0.21], ep, kappa, "Cohen's κ",
               (0.30, 0.72), "Cohen's κ per Epoch",
               ref=0.457, ref_label="LP best κ 0.457",
               test_val=t_kappa, test_label="Test κ")

    # Row 3: F1 / Precision / Recall and LR
    ax_prf = fig.add_axes([0.07, 0.09, 0.40, 0.20])
    ax_style(ax_prf, ylabel="Score", ylim=(0.35, 0.80),
             title="Precision / Recall / F1 per Epoch", tc=ORANGE)
    ax_prf.plot(ep, prec, color="#4A90D9", lw=1.8, label="Precision")
    ax_prf.plot(ep, rec,  color="#D9534F", lw=1.8, label="Recall", ls="--")
    ax_prf.plot(ep, f1,   color=ORANGE,    lw=2.2, label="F1")
    ax_prf.legend(fontsize=8, framealpha=0.8, edgecolor="#D0DDE8")

    ax_lr2 = fig.add_axes([0.57, 0.09, 0.40, 0.20])
    ax_style(ax_lr2, ylabel="LR", title="Learning Rate Schedule", tc=DKGRAY)
    ax_lr2.plot(ep_lr, lr, color=DKGRAY, lw=1.8)
    ax_lr2.fill_between(ep_lr, lr, alpha=0.10, color=DKGRAY)
    ax_lr2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1e}"))

    footer(fig, 2)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Test Results & Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════
def page3(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    page_bg(fig)
    banner(fig, "Test Set Evaluation  —  Best Checkpoint (Epoch {})".format(best_epoch),
           "Evaluated on 702 held-out test images never seen during training or validation")

    # ── Metric bar chart ──────────────────────────────────────────────────────
    ax_bar = fig.add_axes([0.04, 0.55, 0.40, 0.33])
    metric_names = ["AUROC", "Avg Precision", "F1", "Kappa", "Accuracy", "Precision", "Recall"]
    metric_vals  = [t_auroc, t_ap, t_f1, t_kappa, t_acc, t_prec, t_rec]
    bar_colors   = [GREEN if v >= 0.85 else ORANGE if v >= 0.65 else RED for v in metric_vals]

    bars = ax_bar.barh(metric_names, metric_vals, color=bar_colors, height=0.58,
                       edgecolor="white", lw=0.8)
    for bar, val in zip(bars, metric_vals):
        ax_bar.text(min(val + 0.012, 1.05), bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9.5, fontweight="bold", color=NAVY)
    ax_bar.set_xlim(0, 1.18)
    ax_bar.set_xlabel("Score", fontsize=9, color=GRAY)
    ax_bar.tick_params(colors=GRAY, labelsize=9)
    ax_bar.set_facecolor(PANEL)
    ax_bar.grid(axis="x", linestyle="--", lw=0.5, color="#D0DDE8", alpha=0.8)
    for s in ["top", "right", "left"]: ax_bar.spines[s].set_visible(False)
    ax_bar.spines["bottom"].set_color("#C8D8E8")
    ax_bar.axvline(1.0, color="#B0C4D8", lw=0.8, ls="--")
    ax_bar.set_title("Test Set Metrics", fontsize=11, color=NAVY,
                     fontweight="bold", pad=8)

    # Threshold legend
    for val, label, color in [(0.85, "≥0.85 (strong)", GREEN),
                               (0.65, "≥0.65 (good)", ORANGE),
                               (0.0,  "<0.65 (weak)", RED)]:
        ax_bar.add_patch(plt.Rectangle((0, 0), 0, 0, color=color, label=label))
    ax_bar.legend(fontsize=7.5, framealpha=0.85, edgecolor="#D0DDE8",
                  loc="lower right", title="Range", title_fontsize=7.5)

    # ── Detailed metrics table ────────────────────────────────────────────────
    ax_mt = fig.add_axes([0.04, 0.14, 0.40, 0.36])
    ax_mt.axis("off")
    ax_mt.text(0, 1.04, "Detailed Test Metrics with Clinical Context", fontsize=10.5,
               fontweight="bold", color=NAVY, transform=ax_mt.transAxes)

    rows_d = [
        ["AUROC",            f"{t_auroc:.4f}", "Excellent — strong class separability"],
        ["Avg Precision",    f"{t_ap:.4f}",    "Good — precision-recall area under curve"],
        ["Cohen's κ",        f"{t_kappa:.4f}", "Moderate-good — grader-equivalent agreement"],
        ["F1 (macro)",       f"{t_f1:.4f}",    "Reduced by low F1 on rare R2/R3A classes"],
        ["Accuracy",         f"{t_acc:.4f}",   "79.5% — against class-imbalanced test set"],
        ["Precision (macro)",f"{t_prec:.4f}",  "When model predicts a grade, 62.6% correct"],
        ["Recall (macro)",   f"{t_rec:.4f}",   "Model detects 61.6% of true grade instances"],
        ["Test loss",        f"{t_loss:.4f}",  "Cross-entropy on test set (lower is better)"],
    ]

    tbl = ax_mt.table(cellText=rows_d,
                      colLabels=["Metric", "Value", "Interpretation"],
                      cellLoc="left", loc="center", bbox=[0.0, 0.0, 1.0, 0.92])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    col_widths = [0.25, 0.12, 0.63]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D8E4EE")
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#FBF3EC")
        else:
            cell.set_facecolor(PANEL)
        if c == 1 and r > 0:
            cell.set_text_props(fontweight="bold", color=ORANGE)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    ax_cm = fig.add_axes([0.50, 0.14, 0.47, 0.76])
    ax_cm.axis("off")
    ax_cm.text(0, 1.03, "Confusion Matrix  —  Test Set (702 images)",
               fontsize=10.5, fontweight="bold", color=NAVY, transform=ax_cm.transAxes)

    img = np.array(Image.open(CONF_MAT))
    ax_img = fig.add_axes([0.50, 0.14, 0.47, 0.72])
    ax_img.imshow(img); ax_img.axis("off")

    footer(fig, 3)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Clinical Interpretation & Next Steps
# ══════════════════════════════════════════════════════════════════════════════
def page4(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    page_bg(fig)
    banner(fig, "Clinical Interpretation  &  Next Steps",
           "What do the numbers mean?  What should be done next?")

    # ── AUROC interpretation diagram ──────────────────────────────────────────
    ax_auc = fig.add_axes([0.04, 0.60, 0.42, 0.28])
    ax_auc.set_facecolor(PANEL)
    theta = np.linspace(0, np.pi / 2, 100)
    # Stylised ROC curve (actual AUROC not re-computed here — uses reported value)
    # Draw reference lines + annotation zones
    ax_auc.fill_between([0, 1], [0, 1], alpha=0.07, color=GRAY)
    ax_auc.plot([0, 1], [0, 1], "--", color=GRAY, lw=1.2, label="Random (0.50)")
    # Approximate convex curve at 0.929
    t = np.linspace(0, 1, 200)
    roc_approx = 1 - (1 - t) ** (1 / (1 - t_auroc + 0.001 + 1e-6))
    ax_auc.plot(t, roc_approx, color=GREEN, lw=2.5, label=f"Model A FT ({t_auroc:.3f})")
    ax_auc.fill_between(t, roc_approx, alpha=0.15, color=GREEN)
    lp_approx = 1 - (1 - t) ** (1 / (1 - LP_BEST_AUROC + 0.001 + 1e-6))
    ax_auc.plot(t, lp_approx, color=TEAL, lw=1.8, ls="--",
                label=f"Linear Probe val ({LP_BEST_AUROC:.3f})")
    ax_auc.set_xlim(0, 1); ax_auc.set_ylim(0, 1)
    ax_auc.set_xlabel("False Positive Rate", fontsize=9, color=GRAY)
    ax_auc.set_ylabel("True Positive Rate", fontsize=9, color=GRAY)
    ax_auc.set_title("Illustrative ROC Curve Comparison", fontsize=10,
                     color=NAVY, fontweight="bold", pad=7)
    ax_auc.tick_params(colors=GRAY, labelsize=8)
    ax_auc.legend(fontsize=8, framealpha=0.85, edgecolor="#D0DDE8")
    for s in ["top", "right"]: ax_auc.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax_auc.spines[s].set_color("#C8D8E8")
    ax_auc.grid(linestyle="--", lw=0.5, color="#D0DDE8", alpha=0.8)

    # ── Kappa interpretation bar ──────────────────────────────────────────────
    ax_k = fig.add_axes([0.54, 0.60, 0.42, 0.28])
    ax_k.set_facecolor(PANEL)
    bands = [(0.0, 0.20, "Poor",         "#D9534F"),
             (0.20, 0.40, "Fair",         "#E07B39"),
             (0.40, 0.60, "Moderate",     "#F0C040"),
             (0.60, 0.80, "Substantial",  "#5BB85D"),
             (0.80, 1.00, "Almost Perfect","#2E8B57")]

    for lo, hi, label, color in bands:
        ax_k.barh(0, hi - lo, left=lo, height=0.45, color=color, alpha=0.80,
                  edgecolor="white", lw=1)
        ax_k.text((lo + hi) / 2, 0.35, label, ha="center", va="bottom",
                  fontsize=8, color="white", fontweight="bold")

    ax_k.axvline(t_kappa, color=ORANGE, lw=3, zorder=5)
    ax_k.text(t_kappa, -0.38, f"FT κ={t_kappa:.3f}", ha="center",
              fontsize=9.5, color=ORANGE, fontweight="bold")
    ax_k.axvline(0.457, color=TEAL, lw=2, ls="--", zorder=5)
    ax_k.text(0.457, -0.55, f"LP κ=0.457", ha="center", fontsize=8.5, color=TEAL)
    ax_k.set_xlim(0, 1); ax_k.set_ylim(-0.7, 0.85)
    ax_k.set_xlabel("Cohen's κ", fontsize=9, color=GRAY)
    ax_k.set_title("Cohen's κ Interpretation Band  (Landis & Koch 1977)",
                   fontsize=10, color=NAVY, fontweight="bold", pad=7)
    ax_k.tick_params(colors=GRAY, labelsize=8)
    ax_k.set_yticks([])
    for s in ax_k.spines.values(): s.set_visible(False)

    # ── Observations box ──────────────────────────────────────────────────────
    ax_o = fig.add_axes([0.04, 0.34, 0.92, 0.21])
    ax_o.set_facecolor("#F0F8F0"); ax_o.axis("off")
    ax_o.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                                  facecolor="#F0F8F0", edgecolor=GREEN, lw=1.3,
                                  transform=ax_o.transAxes, clip_on=False))
    ax_o.text(0.02, 0.93, "Key Findings", fontsize=10.5, fontweight="bold",
              color=NAVY, transform=ax_o.transAxes, va="top")

    findings = [
        ("AUROC 0.929",
         "Strong discrimination: the model reliably ranks a higher-grade eye above a lower-grade eye in 92.9% of paired comparisons (macro OvR)."),
        ("AUROC gain +0.091",
         "Fine-tuning the full backbone lifts AUROC by 9.1 points over the linear probe, confirming that backbone adaptation to retinopathy is essential."),
        ("κ = 0.639 (Substantial)",
         "Equivalent to 'substantial' inter-grader agreement (Landis & Koch 1977). Clinically, this is the agreement range of experienced human graders."),
        ("Early best-epoch (ep 7)",
         "Val AUROC peaked at epoch 7 then slowly decayed. With ~1,400 training eyes this is typical — the model rapidly over-specialises to augmentation patterns. "
         "For future work: stronger regularisation, stochastic depth, or more data would sustain gains longer."),
        ("F1 / Recall gap",
         "Macro F1 (0.60) and recall (0.62) are lower than AUROC because rare classes R2/R3A are harder to recall correctly. "
         "The confusion matrix will show R2→R1 and R3A→R2 confusions — the model tends to under-grade."),
    ]
    y0 = 0.80
    for title_f, body in findings:
        ax_o.text(0.015, y0, f"▸ {title_f}:", fontsize=9, fontweight="bold",
                  color=GREEN, transform=ax_o.transAxes, va="top")
        ax_o.text(0.18, y0, body, fontsize=8.5, color=DKGRAY,
                  transform=ax_o.transAxes, va="top", linespacing=1.35,
                  wrap=True)
        y0 -= 0.175

    # ── Next steps ────────────────────────────────────────────────────────────
    ax_n = fig.add_axes([0.04, 0.06, 0.92, 0.24])
    ax_n.set_facecolor(LGRAY); ax_n.axis("off")
    ax_n.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                                  facecolor=LGRAY, edgecolor=ORANGE, lw=1.3,
                                  transform=ax_n.transAxes, clip_on=False))
    ax_n.text(0.02, 0.91, "Recommended Next Steps", fontsize=10.5, fontweight="bold",
              color=NAVY, transform=ax_n.transAxes, va="top")

    steps = [
        "1. Train Model B (maculopathy, binary M0/M1) — LP then full fine-tune — to complete the two-model system.",
        "2. Per-class sensitivity / specificity analysis at clinically-motivated operating points "
             "(e.g. 90% sensitivity for R2+ to ensure proliferative cases are flagged).",
        "3. Consider ordinal loss (e.g. coral-loss or ordinal CE) — R2→R1 confusions are clinically safer than R0→R3A, "
             "and standard CE does not penalise large ordinal errors more than small ones.",
        "4. Investigate early stopping at epoch 7–10 for future runs; the validation AUROC plateau starts early, "
             "so running 50 epochs wastes compute and may introduce slight overfitting.",
        "5. Qualitative review: visualise GradCAM / attention maps on misclassified R2/R3A cases to understand "
             "whether errors are in lesion detection or image quality issues.",
    ]
    y0 = 0.77
    for step in steps:
        ax_n.text(0.015, y0, step, fontsize=8.7, color=DKGRAY,
                  transform=ax_n.transAxes, va="top", linespacing=1.4)
        y0 -= 0.165

    footer(fig, 4)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ── Build ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
with PdfPages(OUT_PDF) as pdf:
    pdf.infodict().update({
        "Title":   "RETFound Model A — Full Fine-Tune Report",
        "Author":  "Isaack Joshua",
        "Subject": "Test set evaluation for full fine-tune of RETFound-DINOv2 on retinopathy grading",
    })
    page1(pdf); page2(pdf); page3(pdf); page4(pdf)

print(f"[DONE] {OUT_PDF}")
