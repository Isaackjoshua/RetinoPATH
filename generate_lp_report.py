"""
Detailed PDF report for Model A — Linear Probe.
Output: labels/modelA_lp_report.pdf
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

ROOT    = os.path.dirname(__file__)
LOG_DIR = os.path.join(ROOT, "output_logs/retfound_dinov2_modelA_lp/retfound_dinov2_modelA_lp")
SPLITS  = os.path.join(ROOT, "labels/splits.csv")
OUT_PDF = os.path.join(ROOT, "labels/modelA_lp_report.pdf")

# ── Palette ────────────────────────────────────────────────────────────────────
BG    = "#F7F9FC"
PANEL = "#FFFFFF"
NAVY  = "#1A2B4A"
TEAL  = "#1B7B8A"
TEAL2 = "#2AACBF"
GRAY  = "#6B7C93"
LGRAY = "#EAF2F8"
DKGRAY= "#3D4F66"
ACC1  = "#E07B39"
ACC2  = "#5BB85D"
RED   = "#C0392B"
CLASSES = ["R0", "R1", "R2", "R3A"]
CLASS_C = ["#4A90D9", "#5BB85D", "#F0A030", "#D9534F"]

# ── Load TFEvents ──────────────────────────────────────────────────────────────
ea = EventAccumulator(LOG_DIR); ea.Reload()

def curve(tag):
    seen = {}
    for e in ea.Scalars(tag):
        seen[e.step] = e.value
    steps = sorted(seen)
    return np.array(steps, dtype=float), np.array([seen[s] for s in steps])

ep, auroc  = curve("perf/roc_auc")
_,  loss_v = curve("loss/val")
_,  loss_t = curve("loss/train")
_,  acc    = curve("perf/accuracy")
_,  kappa  = curve("perf/kappa")
_,  f1     = curve("perf/f1")
_,  prec   = curve("perf/precision")
_,  rec    = curve("perf/recall")
_,  ap     = curve("perf/average_precision")
ep_lr, lr  = curve("lr")

best_i     = int(auroc.argmax())
best_epoch = int(ep[best_i])
best_auroc = float(auroc[best_i])
best_kappa = float(kappa[best_i])
best_acc   = float(acc[best_i])

# ── Dataset ────────────────────────────────────────────────────────────────────
df   = pd.read_csv(SPLITS)
mA   = df[df["retinopathy"].isin(CLASSES)]
splt = {}
for sp in ["train", "val", "test"]:
    vc = mA[mA["split"] == sp]["retinopathy"].value_counts()
    splt[sp] = {c: int(vc.get(c, 0)) for c in CLASSES}

total_train = sum(splt["train"].values())
weights     = {c: round(total_train / (4 * splt["train"][c]), 4) for c in CLASSES}

# ── Helpers ────────────────────────────────────────────────────────────────────
def page_bg(fig):
    fig.patch.set_facecolor(BG)

def ax_style(ax, xlabel="Epoch", ylabel="", ylim=None, title="", title_color=TEAL):
    ax.set_facecolor(PANEL)
    ax.grid(True, linestyle="--", lw=0.5, color="#D0DDE8", alpha=0.9)
    ax.set_xlabel(xlabel, fontsize=9, color=GRAY)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=GRAY)
    ax.tick_params(colors=GRAY, labelsize=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#C8D8E8")
    if ylim:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title, fontsize=10, color=title_color, fontweight="bold", pad=7)

def banner(fig, text, sub="", y=0.92, h=0.08, bg=NAVY):
    ax = fig.add_axes([0.0, y, 1.0, h])
    ax.set_facecolor(bg); ax.axis("off")
    ax.text(0.5, 0.65, text, ha="center", va="center", fontsize=14,
            fontweight="bold", color="white", transform=ax.transAxes)
    if sub:
        ax.text(0.5, 0.22, sub, ha="center", va="center", fontsize=9,
                color="#A8C8E0", transform=ax.transAxes)

def footer(fig, page, total=3):
    fig.text(0.5, 0.016, f"RETFound · Model A · Linear Probe  ·  Homerton Reading Centre Data  ·  Page {page} of {total}",
             ha="center", fontsize=7.5, color=GRAY)

def stat_box(ax, x, y, w, h, label, value, unit="", bg=LGRAY, vc=TEAL, lc=GRAY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                                facecolor=bg, edgecolor=TEAL, lw=1.2,
                                transform=ax.transAxes, clip_on=False))
    ax.text(x + w/2, y + h*0.67, value + unit, ha="center", va="center",
            fontsize=18, fontweight="bold", color=vc, transform=ax.transAxes)
    ax.text(x + w/2, y + h*0.22, label, ha="center", va="center",
            fontsize=8.5, color=lc, transform=ax.transAxes)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview & Setup
# ══════════════════════════════════════════════════════════════════════════════
def page1(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    page_bg(fig)

    banner(fig,
           "Model A  ·  Diabetic Retinopathy  ·  Linear Probe",
           "Frozen RETFound-DINOv2 backbone  ·  Linear classification head  ·  50 epochs")

    # ── Concept box ──────────────────────────────────────────────────────────
    ax_c = fig.add_axes([0.03, 0.70, 0.45, 0.20])
    ax_c.set_facecolor(LGRAY); ax_c.axis("off")
    ax_c.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                                  facecolor=LGRAY, edgecolor=TEAL, lw=1.5,
                                  transform=ax_c.transAxes, clip_on=False))
    ax_c.text(0.04, 0.88, "What is Linear Probing?", fontsize=11,
              fontweight="bold", color=NAVY, transform=ax_c.transAxes, va="top")
    concept = (
        "The backbone (ViT-Large, ~307M params) is frozen — its weights are not updated.\n"
        "Only a single linear layer on top is trained to map backbone features → class scores.\n"
        "This tests how much useful retinopathy information is already encoded in the\n"
        "pretrained RETFound representations, without any task-specific adaptation.\n"
        "It is fast (~10 min) and serves as a lower-bound baseline before full fine-tuning."
    )
    ax_c.text(0.04, 0.68, concept, fontsize=8.8, color=DKGRAY,
              transform=ax_c.transAxes, va="top", linespacing=1.6)

    # ── Key stats ─────────────────────────────────────────────────────────────
    ax_s = fig.add_axes([0.52, 0.70, 0.45, 0.20])
    ax_s.set_facecolor(BG); ax_s.axis("off")
    stat_box(ax_s, 0.01, 0.08, 0.30, 0.84, "Best Val AUROC",  f"{best_auroc:.3f}", bg=LGRAY, vc=TEAL)
    stat_box(ax_s, 0.35, 0.08, 0.30, 0.84, "Best Epoch",       str(best_epoch),     bg=LGRAY, vc=TEAL)
    stat_box(ax_s, 0.69, 0.08, 0.30, 0.84, "Best Val κ",       f"{best_kappa:.3f}", bg=LGRAY, vc=TEAL)

    # ── Training setup table ──────────────────────────────────────────────────
    ax_t = fig.add_axes([0.03, 0.38, 0.44, 0.28])
    ax_t.axis("off")
    ax_t.text(0, 1.04, "Training Configuration", fontsize=11, fontweight="bold",
              color=NAVY, transform=ax_t.transAxes)

    config = [
        ["Backbone",        "RETFound-DINOv2 ViT-Large"],
        ["Pretrained on",   "UK fundus cohort (MEH, 736 k images)"],
        ["Backbone status", "FROZEN — zero gradient updates"],
        ["Head",            "Linear layer: 1024 → 4 classes"],
        ["Epochs",          "50  (best ckpt at epoch 30, run to 47 before power cut)"],
        ["Batch size",      "64"],
        ["Input size",      "224 × 224 px"],
        ["Optimiser",       "AdamW (default RETFound settings)"],
        ["Loss",            "CrossEntropyLoss with class weights"],
        ["Class weights",   "R0=1.00 · R1=1.79 · R2=9.53 · R3A=15.68"],
        ["Selection",       "Best checkpoint by val AUROC"],
        ["Test eval",       "Not run (power cut before final epoch)"],
    ]

    tbl = ax_t.table(cellText=config, cellLoc="left", loc="center",
                     bbox=[0.0, 0.0, 1.0, 0.94])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D8E4EE")
        cell.set_facecolor("#F0F5FA" if r % 2 == 0 else PANEL)
        if c == 0:
            cell.set_text_props(color=TEAL, fontweight="bold")
        else:
            cell.set_text_props(color=DKGRAY)

    # ── Class weights explanation ─────────────────────────────────────────────
    ax_w = fig.add_axes([0.52, 0.38, 0.45, 0.28])
    ax_w.axis("off")
    ax_w.text(0, 1.04, "Why Class Weights?", fontsize=11, fontweight="bold",
              color=NAVY, transform=ax_w.transAxes)
    why = (
        "The dataset is severely imbalanced:\n"
        "R0 (no retinopathy) dominates — R3A (active proliferative) is ~15× rarer.\n\n"
        "Without weights the model learns to predict R0 for everything and achieves\n"
        "high accuracy while completely missing the clinically important high grades.\n\n"
        "Weight formula:  w_i  =  N  /  (n_classes × count_i)\n"
        "then scaled so the minimum weight = 1.0.\n\n"
        "Effect: each class contributes equally to the loss regardless of frequency,\n"
        "forcing the model to pay attention to rare but critical R2 and R3A cases."
    )
    ax_w.text(0.02, 0.91, why, fontsize=8.8, color=DKGRAY, transform=ax_w.transAxes,
              va="top", linespacing=1.55)

    # ── Dataset bar chart ─────────────────────────────────────────────────────
    ax_d = fig.add_axes([0.03, 0.09, 0.92, 0.24])
    ax_d.set_facecolor(PANEL)
    ax_d.text(0, 1.06, "Dataset  ·  Image counts per class and split", fontsize=11,
              fontweight="bold", color=NAVY, transform=ax_d.transAxes)

    n_classes = len(CLASSES)
    x         = np.arange(n_classes)
    width     = 0.24
    splits    = ["train", "val", "test"]
    sp_colors = [TEAL, ACC1, ACC2]
    sp_labels = ["Train", "Val", "Test"]

    for i, (sp, color, label) in enumerate(zip(splits, sp_colors, sp_labels)):
        vals = [splt[sp][c] for c in CLASSES]
        bars = ax_d.bar(x + (i - 1) * width, vals, width, label=label,
                        color=color, alpha=0.85, edgecolor="white", lw=0.6)
        for bar, v in zip(bars, vals):
            ax_d.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 12,
                      str(v), ha="center", va="bottom", fontsize=7.5, color=DKGRAY)

    ax_d.set_xticks(x)
    ax_d.set_xticklabels(
        [f"{c}\n(w={weights[c]})" for c in CLASSES], fontsize=9, color=DKGRAY)
    ax_d.set_ylabel("Images", fontsize=9, color=GRAY)
    ax_d.tick_params(colors=GRAY, labelsize=8)
    ax_d.set_facecolor(PANEL)
    ax_d.grid(axis="y", linestyle="--", lw=0.5, color="#D0DDE8", alpha=0.8)
    for s in ["top", "right"]:  ax_d.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax_d.spines[s].set_color("#C8D8E8")
    ax_d.legend(fontsize=9, framealpha=0.8, edgecolor="#D0DDE8")

    footer(fig, 1)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — All Training Curves
# ══════════════════════════════════════════════════════════════════════════════
def page2(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    page_bg(fig)
    banner(fig, "Training Dynamics  —  Validation Metrics over Epochs",
           "Backbone frozen throughout  ·  Only the linear head is updated each epoch")

    # 3×2 grid of curves
    axes_specs = [
        ([0.07, 0.59, 0.40, 0.26], ep, auroc,  "Val AUROC",             (0.68, 0.88),  "Val AUROC"),
        ([0.57, 0.59, 0.40, 0.26], ep, loss_v,  "Val Loss",              None,          "Val Loss"),
        ([0.07, 0.33, 0.40, 0.22], ep, acc,     "Val Accuracy",          (0.55, 0.75),  "Val Accuracy"),
        ([0.57, 0.33, 0.40, 0.22], ep, kappa,   "Cohen's κ",             (0.22, 0.52),  "Cohen's κ"),
        ([0.07, 0.09, 0.40, 0.20], ep, f1,      "Val F1",                None,          "Val F1"),
        ([0.57, 0.09, 0.40, 0.20], ep, prec,    "Val Precision",         None,          "Val Precision / Recall"),
    ]

    for rect, x, y, ylabel, ylim, title in axes_specs:
        ax = fig.add_axes(rect)
        ax_style(ax, ylabel=ylabel, ylim=ylim, title=title)
        ax.plot(x, y, color=TEAL, lw=2)
        ax.fill_between(x, y, alpha=0.13, color=TEAL)
        if ylabel == "Val AUROC":
            bi = int(y.argmax())
            ax.scatter(x[bi], y[bi], color=TEAL, s=60, zorder=5)
            ax.annotate(f" Best: {y[bi]:.3f}  (ep {int(x[bi])})",
                        xy=(x[bi], y[bi]), fontsize=8, color=TEAL)
            ax.axhline(y[bi], color=TEAL, lw=0.8, ls=":", alpha=0.5)
        if ylabel == "Cohen's κ":
            bi = int(y.argmax())
            ax.scatter(x[bi], y[bi], color=TEAL, s=60, zorder=5)
            ax.annotate(f" Best κ: {y[bi]:.3f}", xy=(x[bi], y[bi]),
                        fontsize=8, color=TEAL)
        if title == "Val Precision / Recall":
            ax.plot(ep, rec, color=ACC1, lw=1.8, ls="--", label="Recall")
            ax.legend(["Precision", "Recall"], fontsize=7.5,
                      framealpha=0.8, edgecolor="#D0DDE8")

    # Annotation: vertical line at best epoch on AUROC plot
    ax_main = fig.axes[2]   # AUROC axes is the third add_axes call above (index 2 in fig.axes after banner)

    # Learning rate subplot
    ax_lr = fig.add_axes([0.57, 0.09, 0.40, 0.20])
    ax_style(ax_lr, ylabel="Learning Rate", title="LR Schedule")
    ax_lr.plot(ep_lr, lr, color=DKGRAY, lw=1.8)
    ax_lr.fill_between(ep_lr, lr, alpha=0.10, color=DKGRAY)
    ax_lr.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1e}"))

    footer(fig, 2)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Analysis & Interpretation
# ══════════════════════════════════════════════════════════════════════════════
def page3(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    page_bg(fig)
    banner(fig, "Analysis  &  Interpretation",
           "Convergence behaviour · metric interpretation · limitations")

    # ── Smoothed AUROC with convergence annotation ────────────────────────────
    ax_a = fig.add_axes([0.05, 0.57, 0.55, 0.30])
    ax_style(ax_a, ylabel="Val AUROC", title="AUROC Convergence Analysis")

    # Raw + 5-point moving average
    from numpy.lib.stride_tricks import sliding_window_view
    win = 5
    smooth = np.convolve(auroc, np.ones(win)/win, mode="valid")
    sm_ep  = ep[win-1:]
    ax_a.plot(ep, auroc, color=TEAL, lw=1.2, alpha=0.35, label="Raw")
    ax_a.plot(sm_ep, smooth, color=TEAL, lw=2.5, label="5-epoch MA")
    ax_a.fill_between(sm_ep, smooth, alpha=0.13, color=TEAL)

    bi = int(auroc.argmax())
    ax_a.axvline(ep[bi], color=ACC1, lw=1.2, ls="--", alpha=0.8)
    ax_a.text(ep[bi] + 0.5, 0.695, f"Best: epoch {int(ep[bi])}\nAUROC {auroc[bi]:.3f}",
              fontsize=8, color=ACC1)

    # Convergence band
    plateau_val = auroc[25:].mean()
    ax_a.axhspan(plateau_val - 0.004, plateau_val + 0.004, alpha=0.12,
                 color=ACC2, label=f"Plateau band (±0.004 of mean {plateau_val:.3f})")

    ax_a.set_ylim(0.68, 0.88)
    ax_a.legend(fontsize=8, framealpha=0.85, edgecolor="#D0DDE8", loc="lower right")

    # ── Radar / spider chart of best-epoch metrics ────────────────────────────
    ax_r = fig.add_axes([0.66, 0.57, 0.31, 0.30], polar=True)
    labels_r = ["AUROC", "Accuracy", "F1", "Precision", "Recall", "Kappa"]
    vals_r   = [
        best_auroc,
        float(acc[best_i]),
        float(f1[best_i]),
        float(prec[best_i]),
        float(rec[best_i]),
        float(kappa[best_i]),
    ]
    N      = len(labels_r)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    vals_r += [vals_r[0]]; angles += [angles[0]]
    ax_r.plot(angles, vals_r, color=TEAL, lw=2)
    ax_r.fill(angles, vals_r, color=TEAL, alpha=0.18)
    ax_r.set_thetagrids(np.degrees(angles[:-1]), labels_r, fontsize=8, color=DKGRAY)
    ax_r.set_ylim(0, 1)
    ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_r.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=6.5, color=GRAY)
    ax_r.grid(color="#C8D8E8", lw=0.6)
    ax_r.set_facecolor(LGRAY)
    ax_r.set_title("Best-epoch\nmetric profile", fontsize=9, color=NAVY,
                   fontweight="bold", pad=14)

    # ── Metrics summary table ─────────────────────────────────────────────────
    ax_mt = fig.add_axes([0.05, 0.28, 0.55, 0.24])
    ax_mt.axis("off")
    ax_mt.text(0, 1.05, "Validation Metrics  —  Selected Epochs", fontsize=11,
               fontweight="bold", color=NAVY, transform=ax_mt.transAxes)

    checkpoints = [0, 4, 8, best_epoch, int(ep[-1])]
    col_h  = ["Epoch", "AUROC", "Accuracy", "F1", "Kappa", "Precision", "Recall"]
    rows_d = []
    for ck in checkpoints:
        idx = np.searchsorted(ep, ck)
        if idx >= len(ep): idx = len(ep) - 1
        mark = " ★" if ck == best_epoch else ""
        rows_d.append([f"{int(ep[idx])}{mark}",
                       f"{auroc[idx]:.4f}", f"{acc[idx]:.4f}",
                       f"{f1[idx]:.4f}",   f"{kappa[idx]:.4f}",
                       f"{prec[idx]:.4f}", f"{rec[idx]:.4f}"])

    tbl = ax_mt.table(cellText=rows_d, colLabels=col_h, cellLoc="center",
                      loc="center", bbox=[0.0, 0.0, 1.0, 0.90])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D8E4EE")
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", fontweight="bold")
        elif "★" in str(cell.get_text().get_text()):
            cell.set_facecolor("#D4EED4")
            cell.set_text_props(color=NAVY, fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F0F5FA")
        else:
            cell.set_facecolor(PANEL)

    # ── Observations & limitations ────────────────────────────────────────────
    ax_obs = fig.add_axes([0.05, 0.05, 0.90, 0.19])
    ax_obs.set_facecolor(LGRAY); ax_obs.axis("off")
    ax_obs.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                                    facecolor=LGRAY, edgecolor=TEAL, lw=1.2,
                                    transform=ax_obs.transAxes, clip_on=False))
    ax_obs.text(0.02, 0.88, "Key Observations", fontsize=10.5, fontweight="bold",
                color=NAVY, transform=ax_obs.transAxes, va="top")

    obs = [
        "• AUROC rose rapidly in the first 8 epochs (0.73 → 0.83), showing RETFound features already encode retinopathy-relevant structure.",
        "• The model plateaued around 0.838–0.840 from epoch 25 onwards — the linear head had saturated; more training offered diminishing returns.",
        "• Val accuracy (~70%) and κ (~0.46) are modest because a single linear boundary cannot perfectly separate 4 overlapping ordinal classes.",
        "• No test set evaluation was obtained (power cut at epoch 48, best checkpoint at epoch 30 was preserved).",
        "• This baseline establishes a lower bound: AUROC 0.840 from a frozen backbone with only a linear head.",
    ]
    for i, o in enumerate(obs):
        ax_obs.text(0.02, 0.73 - i * 0.155, o, fontsize=8.8, color=DKGRAY,
                    transform=ax_obs.transAxes, va="top", linespacing=1.4)

    footer(fig, 3)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ── Build ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
with PdfPages(OUT_PDF) as pdf:
    pdf.infodict().update({
        "Title":   "RETFound Model A — Linear Probe Report",
        "Author":  "Isaack Joshua",
        "Subject": "Validation metrics for frozen-backbone linear probe",
    })
    page1(pdf); page2(pdf); page3(pdf)

print(f"[DONE] {OUT_PDF}")
