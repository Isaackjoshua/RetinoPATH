"""
Generate a summary PDF report for Model A (retinopathy) — linear probe + fine-tune.
Output: labels/modelA_report.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LP_LOG_DIR  = os.path.join(ROOT, "output_logs/retfound_dinov2_modelA_lp/retfound_dinov2_modelA_lp")
FT_VAL_CSV  = os.path.join(ROOT, "output_dir/retfound_dinov2_modelA_finetune/retfound_dinov2_modelA_finetune/metrics_val.csv")
FT_TEST_CSV = os.path.join(ROOT, "output_dir/retfound_dinov2_modelA_finetune/retfound_dinov2_modelA_finetune/metrics_test.csv")
CONF_MAT    = os.path.join(ROOT, "output_dir/retfound_dinov2_modelA_finetune/retfound_dinov2_modelA_finetune/confusion_matrix_test.jpg")
SPLITS_CSV  = os.path.join(ROOT, "labels/splits.csv")
OUT_PDF     = os.path.join(ROOT, "reports/modelA_report.pdf")

# ── Palette ────────────────────────────────────────────────────────────────────
C_BG       = "#F7F9FC"   # page background
C_PANEL    = "#FFFFFF"   # card background
C_NAVY     = "#1A2B4A"   # headings / borders
C_TEAL     = "#1B7B8A"   # LP accent
C_ORANGE   = "#E07B39"   # FT accent
C_GREEN    = "#2E8B57"   # good result highlight
C_LIGHT    = "#EAF2F8"   # soft fill
C_GRAY     = "#6B7C93"   # secondary text
C_RED      = "#C0392B"   # warning / poor metric

CLASSES    = ["R0", "R1", "R2", "R3A"]
CLASS_COLS = ["#4A90D9", "#5BB85D", "#F0A030", "#D9534F"]

# ── Data loading ───────────────────────────────────────────────────────────────
def load_lp_curves():
    ea = EventAccumulator(LP_LOG_DIR)
    ea.Reload()
    def series(tag):
        seen = {}
        for e in ea.Scalars(tag):
            seen[e.step] = e.value
        steps = sorted(seen)
        return np.array(steps), np.array([seen[s] for s in steps])
    return {t: series(t) for t in ["perf/roc_auc", "loss/val", "perf/kappa"]}

def load_ft_curves():
    df = pd.read_csv(FT_VAL_CSV)
    # Drop the duplicated initialization rows (first two are identical pre-train evals)
    df = df.iloc[2:].reset_index(drop=True)
    epochs = np.arange(len(df))
    return {"epochs": epochs, "roc_auc": df["roc_auc"].values,
            "loss": df["val_loss"].values, "kappa": df["kappa"].values,
            "accuracy": df["accuracy"].values}

def load_test_metrics():
    df = pd.read_csv(FT_TEST_CSV)
    return df.iloc[0].to_dict()

def load_splits():
    df = pd.read_csv(SPLITS_CSV)
    mA = df[df["retinopathy"].isin(CLASSES)]
    result = {}
    for sp in ["train", "val", "test"]:
        counts = mA[mA["split"] == sp]["retinopathy"].value_counts()
        result[sp] = {c: int(counts.get(c, 0)) for c in CLASSES}
    return result

lp   = load_lp_curves()
ft   = load_ft_curves()
test = load_test_metrics()
splt = load_splits()

# Derived values
lp_best_auroc   = float(lp["perf/roc_auc"][1].max())
lp_best_epoch   = int(lp["perf/roc_auc"][0][lp["perf/roc_auc"][1].argmax()])
ft_best_epoch   = int(ft["epochs"][ft["roc_auc"].argmax()])
ft_best_val_auc = float(ft["roc_auc"].max())
ft_test_auc     = float(test["roc_auc"])
ft_test_acc     = float(test["accuracy"])
ft_test_kappa   = float(test["kappa"])
ft_test_f1      = float(test["f1"])
ft_test_prec    = float(test["precision"])
ft_test_rec     = float(test["recall"])
ft_test_ap      = float(test["average_precision"])

# ── Helper functions ──────────────────────────────────────────────────────────
def page_bg(fig):
    fig.patch.set_facecolor(C_BG)

def card(ax, facecolor=C_PANEL, edgecolor=C_NAVY, lw=0.8, radius=0.04):
    ax.set_facecolor(facecolor)
    for spine in ax.spines.values():
        spine.set_visible(False)

def section_header(ax, text, y=1.04, fontsize=11, color=C_NAVY):
    ax.text(0, y, text, transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", color=color, va="bottom")

def metric_badge(ax, x, y, label, value, color, fontsize_val=22, fontsize_lab=9):
    """Draw a metric chip: big coloured value + small label below."""
    ax.text(x, y + 0.06, value, transform=ax.transAxes,
            fontsize=fontsize_val, fontweight="bold", color=color,
            ha="center", va="center")
    ax.text(x, y - 0.06, label, transform=ax.transAxes,
            fontsize=fontsize_lab, color=C_GRAY,
            ha="center", va="center")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Title + Key Results
# ══════════════════════════════════════════════════════════════════════════════
def page1(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    page_bg(fig)

    # ── Hero banner ───────────────────────────────────────────────────────────
    ax_title = fig.add_axes([0.0, 0.82, 1.0, 0.18])
    ax_title.set_xlim(0, 1); ax_title.set_ylim(0, 1)
    ax_title.set_facecolor(C_NAVY)
    for spine in ax_title.spines.values():
        spine.set_visible(False)
    ax_title.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    ax_title.text(0.5, 0.72, "RETFound  ·  Model A  ·  Diabetic Retinopathy Grading",
                  ha="center", va="center", fontsize=20, fontweight="bold",
                  color="white", transform=ax_title.transAxes)
    ax_title.text(0.5, 0.34,
                  "4-class ordinal grading  (R0 · R1 · R2 · R3A)  |  Colour fundus photography  |  RETFound-DINOv2 ViT-L backbone",
                  ha="center", va="center", fontsize=10, color="#A8C8E0",
                  transform=ax_title.transAxes)

    # ── LP summary card ───────────────────────────────────────────────────────
    ax_lp = fig.add_axes([0.03, 0.52, 0.44, 0.27])
    card(ax_lp, facecolor=C_PANEL)
    ax_lp.set_xlim(0, 1); ax_lp.set_ylim(0, 1)
    ax_lp.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    ax_lp.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                                   facecolor=C_LIGHT, edgecolor=C_TEAL, lw=2,
                                   transform=ax_lp.transAxes, clip_on=False))
    ax_lp.text(0.5, 0.88, "LINEAR PROBE  (frozen backbone)",
               ha="center", va="center", fontsize=11, fontweight="bold",
               color=C_TEAL, transform=ax_lp.transAxes)
    ax_lp.text(0.5, 0.72, "Only classification head trained  ·  50 epochs",
               ha="center", va="center", fontsize=8.5, color=C_GRAY,
               transform=ax_lp.transAxes)

    metric_badge(ax_lp, 0.25, 0.38, "Best Val AUROC", f"{lp_best_auroc:.3f}", C_TEAL)
    metric_badge(ax_lp, 0.75, 0.38, "Best Val Epoch", str(lp_best_epoch), C_TEAL)
    ax_lp.axhline(0.22, 0.07, 0.93, color=C_TEAL, lw=0.5, alpha=0.4)
    ax_lp.text(0.5, 0.10,
               "Baseline — no test evaluation (interrupted at epoch 48, best ckpt saved)",
               ha="center", va="center", fontsize=8, color=C_GRAY,
               transform=ax_lp.transAxes, style="italic")

    # ── FT summary card ───────────────────────────────────────────────────────
    ax_ft = fig.add_axes([0.53, 0.52, 0.44, 0.27])
    card(ax_ft, facecolor=C_PANEL)
    ax_ft.set_xlim(0, 1); ax_ft.set_ylim(0, 1)
    ax_ft.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    ax_ft.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                                   facecolor="#FDF3EC", edgecolor=C_ORANGE, lw=2,
                                   transform=ax_ft.transAxes, clip_on=False))
    ax_ft.text(0.5, 0.88, "FULL FINE-TUNE  (all layers updated)",
               ha="center", va="center", fontsize=11, fontweight="bold",
               color=C_ORANGE, transform=ax_ft.transAxes)
    ax_ft.text(0.5, 0.72, "Layer-wise LR decay (0.65)  ·  50 epochs",
               ha="center", va="center", fontsize=8.5, color=C_GRAY,
               transform=ax_ft.transAxes)

    cols = [0.17, 0.5, 0.83]
    metric_badge(ax_ft, cols[0], 0.38, "Test AUROC",    f"{ft_test_auc:.3f}", C_GREEN)
    metric_badge(ax_ft, cols[1], 0.38, "Test Accuracy", f"{ft_test_acc:.1%}", C_ORANGE)
    metric_badge(ax_ft, cols[2], 0.38, "Test κ",        f"{ft_test_kappa:.3f}", C_ORANGE)
    ax_ft.axhline(0.22, 0.07, 0.93, color=C_ORANGE, lw=0.5, alpha=0.4)
    ax_ft.text(0.5, 0.10, f"Best val epoch: {ft_best_epoch}  ·  Best val AUROC: {ft_best_val_auc:.3f}",
               ha="center", va="center", fontsize=8, color=C_GRAY,
               transform=ax_ft.transAxes, style="italic")

    # ── Delta callout ─────────────────────────────────────────────────────────
    ax_d = fig.add_axes([0.44, 0.565, 0.115, 0.18])
    ax_d.set_xlim(0, 1); ax_d.set_ylim(0, 1)
    ax_d.set_facecolor(C_BG)
    for spine in ax_d.spines.values():
        spine.set_visible(False)
    ax_d.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    delta = ft_test_auc - lp_best_auroc
    ax_d.text(0.5, 0.60, f"+{delta:.3f}", ha="center", va="center",
              fontsize=16, fontweight="bold", color=C_GREEN, transform=ax_d.transAxes)
    ax_d.text(0.5, 0.30, "AUROC gain\nLP → FT", ha="center", va="center",
              fontsize=8, color=C_GRAY, transform=ax_d.transAxes)

    # ── Dataset table ─────────────────────────────────────────────────────────
    ax_ds = fig.add_axes([0.03, 0.13, 0.44, 0.34])
    card(ax_ds)
    ax_ds.axis("off")
    section_header(ax_ds, "Dataset  ·  Model A splits (images per class)", y=1.03)

    col_labels = ["Split", "R0", "R1", "R2", "R3A", "Total"]
    rows_data  = []
    for sp in ["train", "val", "test"]:
        counts = splt[sp]
        total  = sum(counts.values())
        rows_data.append([sp.capitalize(),
                          str(counts["R0"]), str(counts["R1"]),
                          str(counts["R2"]), str(counts["R3A"]),
                          str(total)])

    tbl = ax_ds.table(cellText=rows_data, colLabels=col_labels,
                      cellLoc="center", loc="center",
                      bbox=[0.0, 0.05, 1.0, 0.82])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D0D8E4")
        if r == 0:
            cell.set_facecolor(C_NAVY)
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F0F4F8")
        else:
            cell.set_facecolor(C_PANEL)
        if c == 5 and r > 0:
            cell.set_text_props(fontweight="bold")

    # ── Architecture note ─────────────────────────────────────────────────────
    ax_arch = fig.add_axes([0.53, 0.13, 0.44, 0.34])
    card(ax_arch)
    ax_arch.axis("off")
    section_header(ax_arch, "Model & Training Setup", y=1.03)

    entries = [
        ("Backbone",         "RETFound-DINOv2 ViT-Large (~307M params)"),
        ("Pretrained on",    "UK fundus cohort (MEH, 736k images)"),
        ("Input size",       "224 × 224 px"),
        ("LP head",          "Linear, frozen backbone · batch 64"),
        ("FT LR decay",      "Layer-wise × 0.65 per depth level"),
        ("FT batch size",    "24  (VRAM-safe for ViT-L)"),
        ("Loss",             "CrossEntropyLoss with class weights"),
        ("Class weights",    "R0=1.0 · R1=1.79 · R2=9.53 · R3A=15.68"),
        ("Epochs",           "50 for both LP and FT"),
        ("Selection",        "Best checkpoint by val AUROC"),
    ]
    y0 = 0.93
    for label, val in entries:
        ax_arch.text(0.02, y0, f"  {label}:", fontsize=8.5, color=C_GRAY,
                     transform=ax_arch.transAxes, va="top")
        ax_arch.text(0.35, y0, val, fontsize=8.5, color=C_NAVY, fontweight="bold",
                     transform=ax_arch.transAxes, va="top")
        y0 -= 0.088

    # ── Footer ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.03, "RETFound · Model A · Diabetic Retinopathy  ·  Homerton Reading Centre Data  ·  Page 1 of 3",
             ha="center", fontsize=7.5, color=C_GRAY)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Training Curves
# ══════════════════════════════════════════════════════════════════════════════
def page2(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    page_bg(fig)

    # Banner
    ax_banner = fig.add_axes([0.0, 0.92, 1.0, 0.08])
    ax_banner.set_facecolor(C_NAVY); ax_banner.axis("off")
    ax_banner.text(0.5, 0.5, "Training Dynamics — Validation Metrics over Epochs",
                   ha="center", va="center", fontsize=14, fontweight="bold",
                   color="white", transform=ax_banner.transAxes)

    lp_epochs = lp["perf/roc_auc"][0]
    lp_auroc  = lp["perf/roc_auc"][1]
    lp_loss   = lp["loss/val"][1]
    lp_kappa  = lp["perf/kappa"][1]

    ft_epochs = ft["epochs"]
    ft_auroc  = ft["roc_auc"]
    ft_loss   = ft["loss"]
    ft_kappa  = ft["kappa"]

    def styled_ax(ax, ylabel, ylim=None):
        ax.set_facecolor(C_PANEL)
        ax.grid(True, linestyle="--", linewidth=0.5, color="#D8E4EE", alpha=0.8)
        ax.set_xlabel("Epoch", fontsize=9, color=C_GRAY)
        ax.set_ylabel(ylabel, fontsize=9, color=C_GRAY)
        ax.tick_params(colors=C_GRAY, labelsize=8)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#C8D8E8")
        if ylim:
            ax.set_ylim(ylim)

    # ── Row 1: AUROC ─────────────────────────────────────────────────────────
    ax1 = fig.add_axes([0.07, 0.60, 0.38, 0.26])
    styled_ax(ax1, "Val AUROC")
    ax1.plot(lp_epochs, lp_auroc, color=C_TEAL, lw=2, label="Val AUROC")
    ax1.fill_between(lp_epochs, lp_auroc, alpha=0.12, color=C_TEAL)
    best_i = lp_auroc.argmax()
    ax1.scatter(lp_epochs[best_i], lp_auroc[best_i], color=C_TEAL, s=60, zorder=5)
    ax1.annotate(f"  Best: {lp_auroc[best_i]:.3f}\n  @ epoch {lp_epochs[best_i]}",
                 xy=(lp_epochs[best_i], lp_auroc[best_i]),
                 fontsize=7.5, color=C_TEAL)
    ax1.set_title("Linear Probe  —  Val AUROC", fontsize=10, color=C_TEAL,
                  fontweight="bold", pad=6)
    ax1.set_ylim(0.68, 0.88)
    ax1.axhline(lp_auroc[best_i], color=C_TEAL, lw=0.7, linestyle=":", alpha=0.5)

    ax2 = fig.add_axes([0.57, 0.60, 0.38, 0.26])
    styled_ax(ax2, "Val AUROC")
    ax2.plot(ft_epochs, ft_auroc, color=C_ORANGE, lw=2, label="Val AUROC")
    ax2.fill_between(ft_epochs, ft_auroc, alpha=0.12, color=C_ORANGE)
    best_i_ft = ft_auroc.argmax()
    ax2.scatter(ft_epochs[best_i_ft], ft_auroc[best_i_ft], color=C_ORANGE, s=60, zorder=5)
    ax2.annotate(f"  Best: {ft_auroc[best_i_ft]:.3f}\n  @ epoch {ft_epochs[best_i_ft]}",
                 xy=(ft_epochs[best_i_ft], ft_auroc[best_i_ft]),
                 fontsize=7.5, color=C_ORANGE)
    # Mark test AUROC as horizontal dashed line
    ax2.axhline(ft_test_auc, color=C_GREEN, lw=1.2, linestyle="--", alpha=0.8,
                label=f"Test AUROC {ft_test_auc:.3f}")
    ax2.text(ft_epochs[-1] * 0.75, ft_test_auc + 0.003, f"Test: {ft_test_auc:.3f}",
             fontsize=7.5, color=C_GREEN)
    ax2.set_title("Full Fine-Tune  —  Val AUROC", fontsize=10, color=C_ORANGE,
                  fontweight="bold", pad=6)
    ax2.set_ylim(0.76, 0.95)

    # ── Row 2: Val Loss ───────────────────────────────────────────────────────
    ax3 = fig.add_axes([0.07, 0.31, 0.38, 0.23])
    styled_ax(ax3, "Val Loss")
    ax3.plot(lp_epochs, lp_loss, color=C_TEAL, lw=1.8)
    ax3.fill_between(lp_epochs, lp_loss, alpha=0.10, color=C_TEAL)
    ax3.set_title("Linear Probe  —  Val Loss", fontsize=10, color=C_TEAL,
                  fontweight="bold", pad=6)

    ax4 = fig.add_axes([0.57, 0.31, 0.38, 0.23])
    styled_ax(ax4, "Val Loss")
    ax4.plot(ft_epochs, ft_loss, color=C_ORANGE, lw=1.8)
    ax4.fill_between(ft_epochs, ft_loss, alpha=0.10, color=C_ORANGE)
    ax4.set_title("Full Fine-Tune  —  Val Loss", fontsize=10, color=C_ORANGE,
                  fontweight="bold", pad=6)

    # ── Row 3: Cohen's Kappa ─────────────────────────────────────────────────
    ax5 = fig.add_axes([0.07, 0.07, 0.38, 0.19])
    styled_ax(ax5, "Cohen's κ")
    ax5.plot(lp_epochs, lp_kappa, color=C_TEAL, lw=1.8)
    ax5.fill_between(lp_epochs, lp_kappa, alpha=0.10, color=C_TEAL)
    ax5.set_title("Linear Probe  —  Cohen's κ", fontsize=10, color=C_TEAL,
                  fontweight="bold", pad=6)
    ax5.set_ylim(0.22, 0.52)

    ax6 = fig.add_axes([0.57, 0.07, 0.38, 0.19])
    styled_ax(ax6, "Cohen's κ")
    ax6.plot(ft_epochs, ft_kappa, color=C_ORANGE, lw=1.8)
    ax6.fill_between(ft_epochs, ft_kappa, alpha=0.10, color=C_ORANGE)
    ax6.axhline(ft_test_kappa, color=C_GREEN, lw=1.2, linestyle="--", alpha=0.8)
    ax6.text(ft_epochs[-1] * 0.65, ft_test_kappa + 0.008, f"Test κ: {ft_test_kappa:.3f}",
             fontsize=7.5, color=C_GREEN)
    ax6.set_title("Full Fine-Tune  —  Cohen's κ", fontsize=10, color=C_ORANGE,
                  fontweight="bold", pad=6)

    # Divider label
    fig.text(0.5, 0.58, "·" * 80, ha="center", color="#C8D8E8", fontsize=6)

    fig.text(0.5, 0.03, "RETFound · Model A · Diabetic Retinopathy  ·  Homerton Reading Centre Data  ·  Page 2 of 3",
             ha="center", fontsize=7.5, color=C_GRAY)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Test Results + Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════
def page3(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    page_bg(fig)

    # Banner
    ax_banner = fig.add_axes([0.0, 0.92, 1.0, 0.08])
    ax_banner.set_facecolor(C_NAVY); ax_banner.axis("off")
    ax_banner.text(0.5, 0.5, "Test Set Evaluation  —  Full Fine-Tune (Best Checkpoint)",
                   ha="center", va="center", fontsize=14, fontweight="bold",
                   color="white", transform=ax_banner.transAxes)

    # ── Metrics bar chart ─────────────────────────────────────────────────────
    ax_bar = fig.add_axes([0.05, 0.58, 0.42, 0.30])
    card(ax_bar)

    metric_names  = ["AUROC", "Avg Precision", "Accuracy", "F1", "Precision", "Recall", "Kappa"]
    metric_vals   = [ft_test_auc, ft_test_ap, ft_test_acc, ft_test_f1,
                     ft_test_prec, ft_test_rec, ft_test_kappa]
    colors_bar    = [C_GREEN if v >= 0.85 else C_ORANGE if v >= 0.70 else C_RED
                     for v in metric_vals]

    bars = ax_bar.barh(metric_names, metric_vals, color=colors_bar, height=0.55,
                       edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, metric_vals):
        ax_bar.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9, color=C_NAVY,
                    fontweight="bold")
    ax_bar.set_xlim(0, 1.12)
    ax_bar.set_xlabel("Score", fontsize=9, color=C_GRAY)
    ax_bar.tick_params(labelsize=9, colors=C_GRAY)
    ax_bar.set_facecolor(C_PANEL)
    ax_bar.grid(axis="x", linestyle="--", lw=0.5, color="#D8E4EE", alpha=0.8)
    for spine in ["top", "right", "left"]:
        ax_bar.spines[spine].set_visible(False)
    ax_bar.spines["bottom"].set_color("#C8D8E8")
    ax_bar.set_title("Test Set Metrics  (Fine-Tune)", fontsize=10.5, color=C_NAVY,
                      fontweight="bold", pad=8)
    ax_bar.axvline(1.0, color="#C8D8E8", lw=0.8, linestyle="--")

    # ── LP vs FT comparison table ─────────────────────────────────────────────
    ax_tbl = fig.add_axes([0.05, 0.13, 0.42, 0.38])
    ax_tbl.axis("off")
    section_header(ax_tbl, "Linear Probe  vs  Full Fine-Tune  —  Key Metrics", y=1.04)

    col_labels = ["Metric", "Linear Probe\n(val, best ckpt)", "Fine-Tune\n(test set)", "Δ"]
    rows_data  = [
        ["AUROC",       f"{lp_best_auroc:.3f}",  f"{ft_test_auc:.3f}",
         f"+{ft_test_auc - lp_best_auroc:.3f}"],
        ["Accuracy",    "~70%",                   f"{ft_test_acc:.1%}",   "~+9.5pp"],
        ["Kappa (κ)",   "~0.457",                 f"{ft_test_kappa:.3f}", f"+{ft_test_kappa-0.457:.3f}"],
        ["F1",          "—",                       f"{ft_test_f1:.3f}",    "—"],
        ["Precision",   "—",                       f"{ft_test_prec:.3f}",  "—"],
        ["Recall",      "—",                       f"{ft_test_rec:.3f}",   "—"],
        ["Avg Precision","—",                      f"{ft_test_ap:.3f}",    "—"],
    ]

    tbl = ax_tbl.table(cellText=rows_data, colLabels=col_labels,
                       cellLoc="center", loc="center",
                       bbox=[0.0, 0.0, 1.0, 0.94])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D0D8E4")
        if r == 0:
            cell.set_facecolor(C_NAVY)
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F0F4F8")
        else:
            cell.set_facecolor(C_PANEL)
        if c == 3 and r > 0:
            txt = rows_data[r - 1][3]
            cell.set_text_props(color=C_GREEN if "+" in txt else C_GRAY,
                                fontweight="bold")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    ax_cm = fig.add_axes([0.53, 0.13, 0.44, 0.76])
    ax_cm.axis("off")
    section_header(ax_cm, "Confusion Matrix  —  Test Set  (Fine-Tune, Best Checkpoint)", y=1.015)

    cm_img = np.array(Image.open(CONF_MAT))
    ax_cm_img = fig.add_axes([0.53, 0.13, 0.44, 0.74])
    ax_cm_img.imshow(cm_img)
    ax_cm_img.axis("off")

    # ── Interpretation note ───────────────────────────────────────────────────
    note_text = (
        "AUROC 0.929 indicates strong class-separability across all 4 grades.  "
        "κ = 0.639 (moderate-good) is clinically meaningful for a 4-class ordinal task.  "
        "Lower F1 (0.60) reflects class imbalance: R2/R3A are rare and harder to recall."
    )
    fig.text(0.05, 0.06, note_text, fontsize=8.5, color=C_GRAY,
             wrap=True, ha="left", style="italic",
             bbox=dict(facecolor=C_LIGHT, edgecolor="#C8D8E8", boxstyle="round,pad=0.4"))

    fig.text(0.5, 0.02, "RETFound · Model A · Diabetic Retinopathy  ·  Homerton Reading Centre Data  ·  Page 3 of 3",
             ha="center", fontsize=7.5, color=C_GRAY)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Build PDF
# ══════════════════════════════════════════════════════════════════════════════
os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
with PdfPages(OUT_PDF) as pdf:
    meta = pdf.infodict()
    meta["Title"]   = "RETFound Model A — Diabetic Retinopathy Grading Report"
    meta["Author"]  = "Isaack Joshua"
    meta["Subject"] = "Linear Probe and Full Fine-Tune Evaluation"

    page1(pdf)
    page2(pdf)
    page3(pdf)

print(f"[DONE] PDF saved to: {OUT_PDF}")
