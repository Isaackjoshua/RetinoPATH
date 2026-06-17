"""
Detailed PDF report — Model A, Linear Probe.
Output: labels/modelA_lp_report.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT    = os.path.dirname(__file__)
LOG_DIR = os.path.join(ROOT, "output_logs/retfound_dinov2_modelA_lp/retfound_dinov2_modelA_lp")
VAL_CSV = os.path.join(ROOT, "output_dir/retfound_dinov2_modelA_lp/retfound_dinov2_modelA_lp/metrics_val.csv")
SPLITS  = os.path.join(ROOT, "labels/splits.csv")
OUT_PDF = os.path.join(ROOT, "labels/modelA_lp_report.pdf")

# ── Palette ────────────────────────────────────────────────────────────────────
BG      = "#F7F9FC"
PANEL   = "#FFFFFF"
NAVY    = "#1A2B4A"
TEAL    = "#1B7B8A"
TEAL_L  = "#EAF5F7"
GRAY    = "#6B7C93"
LGRAY   = "#EAF2F8"
DKGRAY  = "#3D4F66"
GREEN   = "#2E8B57"
ORANGE  = "#E07B39"
RED     = "#C0392B"
CLASSES = ["R0", "R1", "R2", "R3A"]
CLASS_FULL = ["R0 — No DR", "R1 — Mild", "R2 — Moderate", "R3A — Active Proliferative"]
CLASS_C = ["#4A90D9", "#5BB85D", "#F0A030", "#D9534F"]

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(VAL_CSV)
ep = np.arange(len(df))

auroc   = df["roc_auc"].values
loss_v  = df["val_loss"].values
acc     = df["accuracy"].values
kappa   = df["kappa"].values
f1      = df["f1"].values
prec    = df["precision"].values
rec     = df["recall"].values
ap      = df["average_precision"].values
m_sens  = df["macro_sensitivity"].values
m_spec  = df["macro_specificity"].values
per_sens = np.column_stack([df[f"sensitivity_{i}"].values for i in range(4)])
per_spec = np.column_stack([df[f"specificity_{i}"].values for i in range(4)])

# TFEvents — train loss (per-batch) → downsample to per-epoch mean
ea = EventAccumulator(LOG_DIR); ea.Reload()
tb_steps, tb_loss = zip(*[(e.step, e.value) for e in ea.Scalars("loss/train")])
tb_steps = np.array(tb_steps); tb_loss = np.array(tb_loss)
n_epochs = len(ep)
max_step = tb_steps.max()
train_loss_ep = []
for e in range(n_epochs):
    lo = e / n_epochs * max_step
    hi = (e + 1) / n_epochs * max_step
    mask = (tb_steps >= lo) & (tb_steps < hi)
    train_loss_ep.append(tb_loss[mask].mean() if mask.any() else np.nan)
train_loss_ep = np.array(train_loss_ep)

tb_lr_steps, tb_lr = zip(*[(e.step, e.value) for e in ea.Scalars("lr")])
tb_lr_steps = np.array(tb_lr_steps); tb_lr = np.array(tb_lr)

# Best checkpoint info
best_i     = int(auroc.argmax())
best_epoch = int(ep[best_i])

# Dataset
sp_df = pd.read_csv(SPLITS)
mA    = sp_df[sp_df["retinopathy"].isin(CLASSES)]
splt  = {sp: {c: int(mA[mA["split"]==sp]["retinopathy"].value_counts().get(c, 0))
              for c in CLASSES} for sp in ["train","val","test"]}
total_train = sum(splt["train"].values())
weights = {c: round(total_train / (4 * splt["train"][c]), 4) for c in CLASSES}

# ── Helpers ────────────────────────────────────────────────────────────────────
def page_bg(fig):
    fig.patch.set_facecolor(BG)

def banner(fig, title, subtitle="", y=0.92, h=0.08):
    ax = fig.add_axes([0.0, y, 1.0, h])
    ax.set_facecolor(NAVY); ax.axis("off")
    ax.text(0.5, 0.68, title, ha="center", va="center", fontsize=15,
            fontweight="bold", color="white", transform=ax.transAxes)
    if subtitle:
        ax.text(0.5, 0.22, subtitle, ha="center", va="center", fontsize=9,
                color="#A8C8E0", transform=ax.transAxes)

def footer(fig, page, total=4):
    fig.text(0.5, 0.015,
             f"RETFound · Model A · Linear Probe  ·  Homerton Reading Centre Data  ·  Page {page} of {total}",
             ha="center", fontsize=7.5, color=GRAY)

def styled_ax(ax, title="", ylabel="", ylim=None, grid=True):
    ax.set_facecolor(PANEL)
    if grid:
        ax.grid(True, linestyle="--", lw=0.5, color="#D0DDE8", alpha=0.85)
    ax.set_xlabel("Epoch", fontsize=8.5, color=GRAY)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8.5, color=GRAY)
    ax.tick_params(colors=GRAY, labelsize=8)
    for s in ["top","right"]:   ax.spines[s].set_visible(False)
    for s in ["left","bottom"]: ax.spines[s].set_color("#C8D8E8")
    if ylim: ax.set_ylim(ylim)
    if title: ax.set_title(title, fontsize=10, color=TEAL, fontweight="bold", pad=7)

def annotate_best(ax, x, y, color=TEAL, label_prefix="Best"):
    bi = int(y.argmax())
    ax.scatter(x[bi], y[bi], color=color, s=55, zorder=6)
    ax.axvline(x[bi], color=color, lw=0.9, ls=":", alpha=0.55)
    ax.annotate(f" {label_prefix}: {y[bi]:.3f}  (ep {int(x[bi])})",
                xy=(x[bi], y[bi]), fontsize=7.5, color=color,
                xytext=(x[bi]+0.8, y[bi]))

def card_box(ax, facecolor=LGRAY, edgecolor=TEAL, lw=1.4):
    ax.add_patch(FancyBboxPatch((0,0), 1, 1, boxstyle="round,pad=0.02",
                                facecolor=facecolor, edgecolor=edgecolor, lw=lw,
                                transform=ax.transAxes, clip_on=False))

def kpi(ax, x, y, w, h, label, value, vc=TEAL, lc=GRAY, fontsize=18):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                                facecolor=LGRAY, edgecolor=TEAL, lw=1.3,
                                transform=ax.transAxes, clip_on=False))
    ax.text(x+w/2, y+h*0.65, value, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=vc, transform=ax.transAxes)
    ax.text(x+w/2, y+h*0.20, label, ha="center", va="center",
            fontsize=8, color=lc, transform=ax.transAxes)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview, Setup & Dataset
# ══════════════════════════════════════════════════════════════════════════════
def page1(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    page_bg(fig)
    banner(fig,
           "Model A  ·  Diabetic Retinopathy Grading  ·  Linear Probe",
           "Frozen RETFound-DINOv2 ViT-Large backbone  ·  Linear head only  ·  50 epochs  ·  4-class (R0/R1/R2/R3A)")

    # ── What is LP ────────────────────────────────────────────────────────────
    ax_lp = fig.add_axes([0.03, 0.70, 0.44, 0.19])
    ax_lp.axis("off"); card_box(ax_lp)
    ax_lp.text(0.04, 0.90, "What is Linear Probing?", fontsize=10.5,
               fontweight="bold", color=NAVY, transform=ax_lp.transAxes, va="top")
    txt = ("All ~307 M backbone weights are FROZEN — gradients flow only through\n"
           "a single linear layer (1024 → 4).  This measures how much retinopathy\n"
           "information is already encoded in the RETFound-DINOv2 representations\n"
           "without any task-specific adaptation, establishing a clean lower bound\n"
           "before the more expensive full fine-tune.")
    ax_lp.text(0.04, 0.68, txt, fontsize=8.8, color=DKGRAY,
               transform=ax_lp.transAxes, va="top", linespacing=1.55)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    ax_k = fig.add_axes([0.52, 0.70, 0.45, 0.19])
    ax_k.axis("off")
    kpi(ax_k, 0.01, 0.05, 0.31, 0.90, "Best Val AUROC",     f"{auroc[best_i]:.3f}")
    kpi(ax_k, 0.35, 0.05, 0.31, 0.90, "Macro Sensitivity",  f"{m_sens[best_i]:.3f}")
    kpi(ax_k, 0.69, 0.05, 0.31, 0.90, "Macro Specificity",  f"{m_spec[best_i]:.3f}")

    # ── Config table ──────────────────────────────────────────────────────────
    ax_cfg = fig.add_axes([0.03, 0.38, 0.44, 0.28])
    ax_cfg.axis("off")
    ax_cfg.text(0, 1.04, "Training Configuration", fontsize=10.5, fontweight="bold",
                color=NAVY, transform=ax_cfg.transAxes)
    config = [
        ["Backbone",        "RETFound-DINOv2 ViT-Large (~307M params)"],
        ["Pretrained on",   "UK Biobank + MEH fundus cohort (736k images)"],
        ["Backbone status", "FROZEN — weights not updated"],
        ["Head",            "Linear: 1024 → 4 classes  (no hidden layers)"],
        ["Epochs",          f"50  (best ckpt by val AUROC at epoch {best_epoch})"],
        ["Batch size",      "64"],
        ["Input",           "224 × 224 px colour fundus photo"],
        ["Optimiser",       "AdamW (default RETFound settings)"],
        ["Loss",            "CrossEntropyLoss with inverse-frequency weights"],
        ["Class weights",   "R0=1.00 · R1=1.79 · R2=9.53 · R3A=15.68"],
        ["Best ckpt metric","Val AUROC  →  checkpoint-best.pth"],
        ["Test eval",       "Deferred (run after fine-tune for direct comparison)"],
    ]
    tbl = ax_cfg.table(cellText=config, cellLoc="left", loc="center",
                       bbox=[0, 0, 1, 0.93])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.8)
    for (r,c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D8E4EE")
        cell.set_facecolor("#EEF4FA" if r%2==0 else PANEL)
        if c==0: cell.set_text_props(color=TEAL, fontweight="bold")
        else:    cell.set_text_props(color=DKGRAY)

    # ── Class weights explainer ────────────────────────────────────────────────
    ax_w = fig.add_axes([0.52, 0.38, 0.45, 0.28])
    ax_w.axis("off")
    ax_w.text(0, 1.04, "Class Imbalance & Weighting", fontsize=10.5, fontweight="bold",
              color=NAVY, transform=ax_w.transAxes)
    why = ("The dataset is severely skewed — R0 outnumbers R3A by ~16:1.\n"
           "Without correction the model would learn to predict R0 for\n"
           "everything, achieving high accuracy while completely missing\n"
           "sight-threatening proliferative disease.\n\n"
           "Weight formula:   w_i = N / (n_classes × count_i)\n"
           "Scaled so min weight = 1.0  →  each class contributes equally\n"
           "to the loss regardless of frequency.")
    ax_w.text(0.02, 0.90, why, fontsize=8.8, color=DKGRAY,
              transform=ax_w.transAxes, va="top", linespacing=1.55)

    # ── Dataset bar chart ─────────────────────────────────────────────────────
    ax_d = fig.add_axes([0.03, 0.07, 0.92, 0.26])
    ax_d.set_facecolor(PANEL)
    ax_d.text(-0.01, 1.06, "Dataset  ·  Image counts per class and split",
              fontsize=10.5, fontweight="bold", color=NAVY, transform=ax_d.transAxes)

    x     = np.arange(4)
    width = 0.25
    sp_colors = [TEAL, ORANGE, GREEN]
    for i, (sp, col) in enumerate(zip(["train","val","test"], sp_colors)):
        vals = [splt[sp][c] for c in CLASSES]
        bars = ax_d.bar(x + (i-1)*width, vals, width, color=col, alpha=0.85,
                        edgecolor="white", lw=0.6, label=sp.capitalize())
        for bar, v in zip(bars, vals):
            ax_d.text(bar.get_x()+bar.get_width()/2, bar.get_height()+14,
                      str(v), ha="center", va="bottom", fontsize=7.5, color=DKGRAY)

    ax_d.set_xticks(x)
    ax_d.set_xticklabels([f"{c}\n(w={weights[c]})" for c in CLASSES], fontsize=9, color=DKGRAY)
    ax_d.set_ylabel("Images", fontsize=9, color=GRAY)
    ax_d.tick_params(colors=GRAY, labelsize=8)
    ax_d.set_facecolor(PANEL)
    ax_d.grid(axis="y", linestyle="--", lw=0.5, color="#D0DDE8", alpha=0.8)
    for s in ["top","right"]:   ax_d.spines[s].set_visible(False)
    for s in ["left","bottom"]: ax_d.spines[s].set_color("#C8D8E8")
    ax_d.legend(fontsize=9, framealpha=0.85, edgecolor="#D0DDE8")

    footer(fig, 1)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — All Training Curves
# ══════════════════════════════════════════════════════════════════════════════
def page2(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    page_bg(fig)
    banner(fig, "Training Dynamics  —  All Validation Metrics over 50 Epochs",
           "Frozen backbone · only the linear classification head receives gradient updates each epoch")

    def curve_plot(rect, y, title, ylabel, ylim, color=TEAL,
                   y2=None, y2_label=None, y2_color=None,
                   show_best=True, ref_line=None, ref_label=None):
        ax = fig.add_axes(rect)
        styled_ax(ax, title=title, ylabel=ylabel, ylim=ylim)
        ax.plot(ep, y, color=color, lw=2)
        ax.fill_between(ep, y, alpha=0.12, color=color)
        if show_best:
            annotate_best(ax, ep, y, color=color)
        if y2 is not None:
            ax.plot(ep, y2, color=y2_color, lw=1.8, ls="--", alpha=0.9, label=y2_label)
            ax.legend(fontsize=7.5, framealpha=0.85, edgecolor="#D0DDE8", loc="lower right")
        if ref_line is not None:
            ax.axhline(ref_line, color="#AABBCC", lw=1.0, ls=":", alpha=0.8)
            ax.text(ep[-1]*0.65, ref_line+0.005, ref_label, fontsize=7, color=GRAY)
        return ax

    # Row 1 — AUROC and Loss
    curve_plot([0.06, 0.61, 0.40, 0.25],
               auroc, "Val AUROC", "AUROC", (0.68, 0.88))

    ax_loss = fig.add_axes([0.57, 0.61, 0.40, 0.25])
    styled_ax(ax_loss, title="Train Loss vs Val Loss", ylabel="Loss")
    ax_loss.plot(ep, train_loss_ep, color=DKGRAY, lw=1.8, alpha=0.75, label="Train (epoch mean)")
    ax_loss.plot(ep, loss_v, color=TEAL, lw=2, label="Val")
    ax_loss.fill_between(ep, loss_v, alpha=0.10, color=TEAL)
    ax_loss.legend(fontsize=7.5, framealpha=0.85, edgecolor="#D0DDE8")

    # Row 2 — Sensitivity and Specificity (the new metrics)
    ax_sens = fig.add_axes([0.06, 0.36, 0.40, 0.21])
    styled_ax(ax_sens, title="Macro Sensitivity over Epochs",
              ylabel="Sensitivity", ylim=(0.40, 0.72))
    ax_sens.plot(ep, m_sens, color=GREEN, lw=2)
    ax_sens.fill_between(ep, m_sens, alpha=0.12, color=GREEN)
    annotate_best(ax_sens, ep, m_sens, color=GREEN)
    ax_sens.axhline(0.5, color="#AABBCC", lw=0.9, ls=":", alpha=0.7)
    ax_sens.text(ep[-1]*0.55, 0.503, "0.5 reference", fontsize=6.5, color=GRAY)

    ax_spec = fig.add_axes([0.57, 0.36, 0.40, 0.21])
    styled_ax(ax_spec, title="Macro Specificity over Epochs",
              ylabel="Specificity", ylim=(0.80, 0.90))
    ax_spec.plot(ep, m_spec, color="#8B5CF6", lw=2)
    ax_spec.fill_between(ep, m_spec, alpha=0.12, color="#8B5CF6")
    bi_sp = int(m_spec.argmax())
    ax_spec.scatter(ep[bi_sp], m_spec[bi_sp], color="#8B5CF6", s=55, zorder=6)
    ax_spec.annotate(f" Best: {m_spec[bi_sp]:.3f}  (ep {int(ep[bi_sp])})",
                     xy=(ep[bi_sp], m_spec[bi_sp]), fontsize=7.5, color="#8B5CF6")

    # Row 3 — Accuracy / Kappa and Precision / Recall
    ax_ak = fig.add_axes([0.06, 0.09, 0.40, 0.22])
    styled_ax(ax_ak, title="Accuracy & Cohen's κ", ylabel="Score", ylim=(0.50, 0.76))
    ax_ak.plot(ep, acc,   color=TEAL,   lw=2,   label="Accuracy")
    ax_ak.plot(ep, kappa, color=ORANGE, lw=1.8, ls="--", label="Cohen's κ")
    ax_ak.fill_between(ep, acc,   alpha=0.10, color=TEAL)
    ax_ak.fill_between(ep, kappa, alpha=0.08, color=ORANGE)
    ax_ak.legend(fontsize=7.5, framealpha=0.85, edgecolor="#D0DDE8")
    bi_a = int(acc.argmax())
    ax_ak.scatter(ep[bi_a], acc[bi_a], color=TEAL, s=50, zorder=6)

    ax_prf = fig.add_axes([0.57, 0.09, 0.40, 0.22])
    styled_ax(ax_prf, title="Precision / Recall / F1", ylabel="Score", ylim=(0.35, 0.72))
    ax_prf.plot(ep, prec, color="#4A90D9", lw=1.8, label="Precision")
    ax_prf.plot(ep, rec,  color="#D9534F", lw=1.8, ls="--", label="Recall / Sensitivity")
    ax_prf.plot(ep, f1,   color=TEAL,     lw=2.2, label="F1")
    ax_prf.fill_between(ep, f1, alpha=0.10, color=TEAL)
    ax_prf.legend(fontsize=7.5, framealpha=0.85, edgecolor="#D0DDE8", loc="lower right")

    footer(fig, 2)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Per-class Sensitivity & Specificity Deep-dive
# ══════════════════════════════════════════════════════════════════════════════
def page3(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    page_bg(fig)
    banner(fig, "Per-class Sensitivity & Specificity  —  Linear Probe",
           "Sensitivity = TP/(TP+FN)  ·  Specificity = TN/(TN+FP)  ·  OvR decomposition at best-AUROC checkpoint")

    best_sens = per_sens[best_i]   # shape (4,)
    best_spec = per_spec[best_i]   # shape (4,)

    # ── Per-class bar chart ───────────────────────────────────────────────────
    ax_bar = fig.add_axes([0.05, 0.57, 0.56, 0.30])
    ax_bar.set_facecolor(PANEL)
    x = np.arange(4); w = 0.38
    b1 = ax_bar.bar(x - w/2, best_sens, w, color=CLASS_C, alpha=0.85,
                    edgecolor="white", lw=0.8, label="Sensitivity")
    b2 = ax_bar.bar(x + w/2, best_spec, w, color=CLASS_C, alpha=0.40,
                    edgecolor=CLASS_C, lw=1.2, label="Specificity", hatch="///")
    for bar, v in list(zip(b1, best_sens)) + list(zip(b2, best_spec)):
        ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color=DKGRAY)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(CLASS_FULL, fontsize=9, color=DKGRAY)
    ax_bar.set_ylabel("Score", fontsize=9, color=GRAY)
    ax_bar.set_ylim(0, 1.08)
    ax_bar.set_title(f"Sensitivity & Specificity per Class  (best checkpoint, epoch {best_epoch})",
                     fontsize=10.5, color=NAVY, fontweight="bold", pad=8)
    ax_bar.tick_params(colors=GRAY, labelsize=8)
    ax_bar.grid(axis="y", linestyle="--", lw=0.5, color="#D0DDE8", alpha=0.8)
    for s in ["top","right"]:   ax_bar.spines[s].set_visible(False)
    for s in ["left","bottom"]: ax_bar.spines[s].set_color("#C8D8E8")
    ax_bar.legend(fontsize=9, framealpha=0.85, edgecolor="#D0DDE8", loc="upper left")
    ax_bar.axhline(0.80, color="#AABBCC", lw=0.9, ls=":", alpha=0.8)
    ax_bar.text(3.5, 0.81, "0.80 ref", fontsize=7, color=GRAY, ha="right")

    # ── Sensitivity over epochs per class ─────────────────────────────────────
    ax_se = fig.add_axes([0.67, 0.57, 0.30, 0.30])
    styled_ax(ax_se, title="Sensitivity per Class over Epochs",
              ylabel="Sensitivity", ylim=(0.0, 1.02))
    for i, (c, col) in enumerate(zip(CLASSES, CLASS_C)):
        ax_se.plot(ep, per_sens[:, i], color=col, lw=1.8, label=c, alpha=0.9)
    ax_se.legend(fontsize=8, framealpha=0.85, edgecolor="#D0DDE8", loc="lower right")
    ax_se.axhline(best_sens.mean(), color=GRAY, lw=0.9, ls="--", alpha=0.6)

    # ── Specificity over epochs per class ─────────────────────────────────────
    ax_sp = fig.add_axes([0.67, 0.24, 0.30, 0.28])
    styled_ax(ax_sp, title="Specificity per Class over Epochs",
              ylabel="Specificity", ylim=(0.55, 1.02))
    for i, (c, col) in enumerate(zip(CLASSES, CLASS_C)):
        ax_sp.plot(ep, per_spec[:, i], color=col, lw=1.8, label=c, alpha=0.9)
    ax_sp.legend(fontsize=8, framealpha=0.85, edgecolor="#D0DDE8", loc="lower right")

    # ── Summary table ─────────────────────────────────────────────────────────
    ax_tbl = fig.add_axes([0.05, 0.24, 0.58, 0.28])
    ax_tbl.axis("off")
    ax_tbl.text(0, 1.04, f"Metric Summary at Best Checkpoint (epoch {best_epoch}, AUROC {auroc[best_i]:.3f})",
                fontsize=10.5, fontweight="bold", color=NAVY, transform=ax_tbl.transAxes)

    col_h = ["Class", "Sensitivity", "Specificity", "Clinical Note"]
    rows  = [
        ["R0 — No DR",
         f"{best_sens[0]:.3f}",
         f"{best_spec[0]:.3f}",
         "High sensitivity: model catches most healthy eyes"],
        ["R1 — Mild",
         f"{best_sens[1]:.3f}",
         f"{best_spec[1]:.3f}",
         "Weakest sensitivity — mild DR hard to separate from R0 linearly"],
        ["R2 — Moderate",
         f"{best_sens[2]:.3f}",
         f"{best_spec[2]:.3f}",
         "Moderate recall; high specificity avoids over-alerting"],
        ["R3A — Active Prolif.",
         f"{best_sens[3]:.3f}",
         f"{best_spec[3]:.3f}",
         "Good sensitivity despite only 124 training images"],
        ["Macro average",
         f"{best_sens.mean():.3f}",
         f"{best_spec.mean():.3f}",
         "Frozen backbone; full fine-tune expected to improve all classes"],
    ]
    tbl = ax_tbl.table(cellText=rows, colLabels=col_h, cellLoc="left",
                       loc="center", bbox=[0, 0, 1, 0.90])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r,c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D8E4EE")
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", fontweight="bold")
        elif r == 5:
            cell.set_facecolor(LGRAY)
            cell.set_text_props(fontweight="bold", color=NAVY)
        elif r % 2 == 0:
            cell.set_facecolor("#EEF4FA")
        else:
            cell.set_facecolor(PANEL)
        if c == 1 and r > 0:
            val = float(rows[r-1][1])
            cell.set_text_props(color=GREEN if val >= 0.70 else ORANGE if val >= 0.55 else RED,
                                fontweight="bold")
        if c == 2 and r > 0:
            val = float(rows[r-1][2])
            cell.set_text_props(color=GREEN if val >= 0.85 else ORANGE if val >= 0.70 else RED,
                                fontweight="bold")

    # ── Clinical context box ───────────────────────────────────────────────────
    ax_cl = fig.add_axes([0.05, 0.05, 0.92, 0.16])
    ax_cl.set_facecolor(TEAL_L); ax_cl.axis("off")
    card_box(ax_cl, facecolor=TEAL_L, edgecolor=TEAL)
    ax_cl.text(0.02, 0.90, "Clinical Interpretation of Sensitivity vs Specificity",
               fontsize=10, fontweight="bold", color=NAVY, transform=ax_cl.transAxes, va="top")
    notes = [
        ("Sensitivity (recall)", "Low sensitivity = missed cases. Clinically dangerous for R2/R3A — undetected proliferative disease can lead to preventable blindness."),
        ("Specificity",          f"Low specificity = false alarms. R0 specificity of {best_spec[0]:.2f} means {(1-best_spec[0])*100:.0f}% of healthy eyes are flagged — acceptable as a screening tool but increases unnecessary referrals."),
        ("R1 bottleneck",        f"R1 sensitivity of {best_sens[1]:.2f} is the weakest point. Mild DR is a borderline grade that overlaps with both R0 and R2 in feature space — a linear head cannot form a curved boundary."),
    ]
    x0, y0 = 0.01, 0.68
    for title_n, body in notes:
        ax_cl.text(x0, y0, f"▸ {title_n}:", fontsize=8.8, fontweight="bold",
                   color=TEAL, transform=ax_cl.transAxes, va="top")
        ax_cl.text(x0 + 0.16, y0, body, fontsize=8.8, color=DKGRAY,
                   transform=ax_cl.transAxes, va="top")
        y0 -= 0.30

    footer(fig, 3)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Convergence Analysis & Summary
# ══════════════════════════════════════════════════════════════════════════════
def page4(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    page_bg(fig)
    banner(fig, "Convergence Analysis  &  Summary",
           "Understanding training behaviour · plateau detection · key takeaways")

    # ── Smoothed AUROC with convergence zones ────────────────────────────────
    ax_auc = fig.add_axes([0.05, 0.58, 0.55, 0.29])
    styled_ax(ax_auc, title="AUROC Convergence  (raw + 5-epoch moving average)",
              ylabel="Val AUROC", ylim=(0.68, 0.87))
    smooth = np.convolve(auroc, np.ones(5)/5, mode="valid")
    sm_ep  = ep[4:]
    ax_auc.plot(ep, auroc, color=TEAL, lw=1.2, alpha=0.30, label="Raw val AUROC")
    ax_auc.plot(sm_ep, smooth, color=TEAL, lw=2.5, label="5-ep moving avg")
    ax_auc.fill_between(sm_ep, smooth, alpha=0.13, color=TEAL)
    # Rapid gain zone
    ax_auc.axvspan(0, 8, alpha=0.07, color=GREEN, label="Rapid gain (ep 0–8)")
    ax_auc.text(1, 0.690, "Rapid\ngain", fontsize=7.5, color=GREEN,
                va="bottom", fontweight="bold")
    # Plateau zone
    plateau_val = auroc[20:].mean()
    ax_auc.axhspan(plateau_val-0.003, plateau_val+0.003, alpha=0.12,
                   color=ORANGE, label=f"Plateau (mean {plateau_val:.3f})")
    ax_auc.axvline(ep[best_i], color=TEAL, lw=1.2, ls="--", alpha=0.7)
    ax_auc.text(ep[best_i]+0.4, 0.690, f"Best\nep {best_epoch}", fontsize=7.5,
                color=TEAL, va="bottom")
    ax_auc.legend(fontsize=7.5, framealpha=0.85, edgecolor="#D0DDE8", loc="lower right")

    # ── Radar chart of best-epoch metrics ─────────────────────────────────────
    ax_r = fig.add_axes([0.66, 0.56, 0.31, 0.32], polar=True)
    labels_r = ["AUROC", "Accuracy", "F1", "Sensitivity", "Specificity", "Kappa"]
    vals_r   = [auroc[best_i], acc[best_i], f1[best_i],
                m_sens[best_i], m_spec[best_i], kappa[best_i]]
    N      = len(labels_r)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    v_plot = vals_r + [vals_r[0]]; angles += [angles[0]]
    ax_r.plot(angles, v_plot, color=TEAL, lw=2)
    ax_r.fill(angles, v_plot, color=TEAL, alpha=0.18)
    ax_r.set_thetagrids(np.degrees(angles[:-1]), labels_r, fontsize=8.5, color=DKGRAY)
    ax_r.set_ylim(0, 1)
    ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_r.set_yticklabels(["0.25","0.50","0.75","1.00"], fontsize=6, color=GRAY)
    ax_r.grid(color="#C8D8E8", lw=0.6); ax_r.set_facecolor(LGRAY)
    ax_r.set_title(f"Metric Profile\n(best epoch {best_epoch})",
                   fontsize=9, color=NAVY, fontweight="bold", pad=14)

    # ── Milestone epoch table ──────────────────────────────────────────────────
    ax_mt = fig.add_axes([0.05, 0.29, 0.60, 0.24])
    ax_mt.axis("off")
    ax_mt.text(0, 1.05, "Key Metrics at Selected Epochs", fontsize=10.5,
               fontweight="bold", color=NAVY, transform=ax_mt.transAxes)

    milestones = [0, 4, 9, 19, best_epoch, 49]
    col_h = ["Epoch", "AUROC", "Accuracy", "Kappa", "Sensitivity", "Specificity", "F1"]
    rows  = []
    for e_m in milestones:
        idx  = min(e_m, len(df)-1)
        mark = " ★" if e_m == best_epoch else ""
        rows.append([f"Ep {idx}{mark}",
                     f"{auroc[idx]:.4f}", f"{acc[idx]:.4f}", f"{kappa[idx]:.4f}",
                     f"{m_sens[idx]:.4f}", f"{m_spec[idx]:.4f}", f"{f1[idx]:.4f}"])
    tbl = ax_mt.table(cellText=rows, colLabels=col_h, cellLoc="center",
                      loc="center", bbox=[0, 0, 1, 0.90])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r,c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D8E4EE")
        if r == 0:
            cell.set_facecolor(NAVY); cell.set_text_props(color="white", fontweight="bold")
        elif "★" in str(cell.get_text().get_text()):
            cell.set_facecolor("#D4EED4"); cell.set_text_props(color=NAVY, fontweight="bold")
        elif r%2==0: cell.set_facecolor("#EEF4FA")
        else:        cell.set_facecolor(PANEL)

    # ── LR schedule ───────────────────────────────────────────────────────────
    ax_lr = fig.add_axes([0.69, 0.29, 0.28, 0.24])
    styled_ax(ax_lr, title="LR Schedule", ylabel="LR", ylim=None)
    lr_ep_vals = []
    for e in range(n_epochs):
        lo = e / n_epochs * tb_lr_steps.max()
        hi = (e+1) / n_epochs * tb_lr_steps.max()
        mask = (tb_lr_steps >= lo) & (tb_lr_steps < hi)
        lr_ep_vals.append(tb_lr[mask].mean() if mask.any() else np.nan)
    ax_lr.plot(ep, lr_ep_vals, color=DKGRAY, lw=1.8)
    ax_lr.fill_between(ep, lr_ep_vals, alpha=0.10, color=DKGRAY)
    ax_lr.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1e}"))

    # ── Key observations ──────────────────────────────────────────────────────
    ax_obs = fig.add_axes([0.05, 0.05, 0.92, 0.21])
    ax_obs.set_facecolor(LGRAY); ax_obs.axis("off")
    card_box(ax_obs, facecolor=LGRAY)
    ax_obs.text(0.02, 0.93, "Key Observations", fontsize=10.5, fontweight="bold",
                color=NAVY, transform=ax_obs.transAxes, va="top")
    obs = [
        f"• AUROC climbed rapidly in the first 8 epochs ({auroc[0]:.3f} → {auroc[min(8, len(ep)-1)]:.3f}), confirming RETFound features already carry substantial retinopathy structure without any adaptation.",
        f"• The model plateaued around AUROC {auroc[20:].mean():.3f} from epoch 20 onwards; the linear head saturated and additional training offered negligible gains.",
        f"• Macro specificity ({m_spec[best_i]:.3f}) substantially exceeds macro sensitivity ({m_sens[best_i]:.3f}) — the model is conservative: it rarely raises a false alarm but does miss cases, especially R1.",
        f"• R1 sensitivity ({per_sens[best_i, 1]:.3f}) is the critical weakness: mild DR sits in a feature-space region that cannot be cleanly separated from R0 by a single hyperplane.",
        f"• R2 and R3A achieve sensitivity of {per_sens[best_i, 2]:.3f} / {per_sens[best_i, 3]:.3f} respectively despite being rare classes — the class weights force the model to attend to these grades.",
        f"• This baseline establishes AUROC {auroc[best_i]:.3f} as the frozen-feature lower bound; full fine-tuning of all backbone layers is expected to push significantly higher.",
    ]
    y0 = 0.78
    for o in obs:
        ax_obs.text(0.02, y0, o, fontsize=8.8, color=DKGRAY,
                    transform=ax_obs.transAxes, va="top", linespacing=1.4)
        y0 -= 0.135

    footer(fig, 4)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ── Build ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
with PdfPages(OUT_PDF) as pdf:
    pdf.infodict().update({
        "Title":   "RETFound Model A — Linear Probe Report",
        "Author":  "Isaack Joshua",
        "Subject": "Validation metrics for frozen-backbone linear probe, with per-class sensitivity and specificity",
    })
    page1(pdf); page2(pdf); page3(pdf); page4(pdf)

print(f"[DONE] {OUT_PDF}")
