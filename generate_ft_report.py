"""
Detailed PDF report — Model A, Full Fine-tune.
Output: labels/modelA_ft_report.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT     = os.path.dirname(__file__)
LOG_DIR  = os.path.join(ROOT, "output_logs/retfound_dinov2_modelA_finetune/retfound_dinov2_modelA_finetune")
VAL_CSV  = os.path.join(ROOT, "output_dir/retfound_dinov2_modelA_finetune/retfound_dinov2_modelA_finetune/metrics_val.csv")
TEST_CSV = os.path.join(ROOT, "output_dir/retfound_dinov2_modelA_finetune/retfound_dinov2_modelA_finetune/metrics_test.csv")
LP_TEST  = os.path.join(ROOT, "output_dir/retfound_dinov2_modelA_lp/retfound_dinov2_modelA_lp/metrics_test.csv")
SPLITS   = os.path.join(ROOT, "labels/splits.csv")
OUT_PDF  = os.path.join(ROOT, "labels/modelA_ft_report.pdf")

# ── Palette ─────────────────────────────────────────────────────────────────────
BG     = "#F7F9FC"
PANEL  = "#FFFFFF"
NAVY   = "#1A2B4A"
TEAL   = "#1B7B8A"
TEAL_L = "#EAF5F7"
GRAY   = "#6B7C93"
LGRAY  = "#EAF2F8"
DKGRAY = "#3D4F66"
GREEN  = "#2E8B57"
ORANGE = "#E07B39"
RED    = "#C0392B"
PURPLE = "#8B5CF6"

CLASSES    = ["R0", "R1", "R2", "R3A"]
CLASS_FULL = ["R0 — No DR", "R1 — Mild", "R2 — Moderate", "R3A — Active Proliferative"]
CLASS_C    = ["#4A90D9", "#5BB85D", "#F0A030", "#D9534F"]

# ── Load val data ───────────────────────────────────────────────────────────────
df  = pd.read_csv(VAL_CSV)
ep  = np.arange(len(df))

auroc    = df["roc_auc"].values
loss_v   = df["val_loss"].values
acc      = df["accuracy"].values
kappa    = df["kappa"].values
f1       = df["f1"].values
prec     = df["precision"].values
rec      = df["recall"].values
m_sens   = df["macro_sensitivity"].values
m_spec   = df["macro_specificity"].values
per_sens = np.column_stack([df[f"sensitivity_{i}"].values for i in range(4)])
per_spec = np.column_stack([df[f"specificity_{i}"].values for i in range(4)])

best_i     = int(auroc.argmax())
best_epoch = int(ep[best_i])

# ── Load test data ───────────────────────────────────────────────────────────────
tst     = pd.read_csv(TEST_CSV).iloc[0]
t_auroc = float(tst["roc_auc"])
t_acc   = float(tst["accuracy"])
t_f1    = float(tst["f1"])
t_kappa = float(tst["kappa"])
t_sens  = float(tst["macro_sensitivity"])
t_spec  = float(tst["macro_specificity"])
t_ps    = [float(tst[f"sensitivity_{i}"]) for i in range(4)]
t_pspec = [float(tst[f"specificity_{i}"]) for i in range(4)]

# ── Load LP test data for comparison ────────────────────────────────────────────
lp      = pd.read_csv(LP_TEST).iloc[0]
lp_auroc = float(lp["roc_auc"])
lp_acc   = float(lp["accuracy"])
lp_f1    = float(lp["f1"])
lp_kappa = float(lp["kappa"])
lp_sens  = float(lp["macro_sensitivity"])
lp_spec  = float(lp["macro_specificity"])
lp_ps    = [float(lp[f"sensitivity_{i}"]) for i in range(4)]
lp_pspec = [float(lp[f"specificity_{i}"]) for i in range(4)]

# ── TFEvents — train loss per epoch ─────────────────────────────────────────────
ea = EventAccumulator(LOG_DIR); ea.Reload()
tb_steps, tb_loss = zip(*[(e.step, e.value) for e in ea.Scalars("loss/train")])
tb_steps = np.array(tb_steps); tb_loss = np.array(tb_loss)
tb_lr_steps, tb_lr = zip(*[(e.step, e.value) for e in ea.Scalars("lr")])
tb_lr_steps = np.array(tb_lr_steps); tb_lr = np.array(tb_lr)

n_epochs = len(ep)
max_step = tb_steps.max()
train_loss_ep = []
for e in range(n_epochs):
    lo   = e / n_epochs * max_step
    hi   = (e + 1) / n_epochs * max_step
    mask = (tb_steps >= lo) & (tb_steps < hi)
    train_loss_ep.append(tb_loss[mask].mean() if mask.any() else np.nan)
train_loss_ep = np.array(train_loss_ep)

# ── Dataset split counts ─────────────────────────────────────────────────────────
sp_df = pd.read_csv(SPLITS)
mA    = sp_df[sp_df["retinopathy"].isin(CLASSES)]
splt  = {sp: {c: int(mA[mA["split"] == sp]["retinopathy"].value_counts().get(c, 0))
              for c in CLASSES} for sp in ["train", "val", "test"]}
total_train = sum(splt["train"].values())
weights = {c: round(total_train / (4 * splt["train"][c]), 4) for c in CLASSES}

# ── Helpers ──────────────────────────────────────────────────────────────────────
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
             f"RETFound · Model A · Full Fine-tune  ·  Homerton Reading Centre Data  ·  Page {page} of {total}",
             ha="center", fontsize=7.5, color=GRAY)

def styled_ax(ax, title="", ylabel="", ylim=None, grid=True):
    ax.set_facecolor(PANEL)
    if grid:
        ax.grid(True, linestyle="--", lw=0.5, color="#D0DDE8", alpha=0.85)
    ax.set_xlabel("Epoch", fontsize=8.5, color=GRAY)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8.5, color=GRAY)
    ax.tick_params(colors=GRAY, labelsize=8)
    for s in ["top", "right"]:   ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_color("#C8D8E8")
    if ylim: ax.set_ylim(ylim)
    if title: ax.set_title(title, fontsize=10, color=TEAL, fontweight="bold", pad=7)

def annotate_best(ax, x, y, color=TEAL, label_prefix="Best"):
    bi = int(y.argmax())
    ax.scatter(x[bi], y[bi], color=color, s=55, zorder=6)
    ax.axvline(x[bi], color=color, lw=0.9, ls=":", alpha=0.55)
    ax.annotate(f" {label_prefix}: {y[bi]:.3f}  (ep {int(x[bi])})",
                xy=(x[bi], y[bi]), fontsize=7.5, color=color,
                xytext=(x[bi] + 0.8, y[bi]))

def card_box(ax, facecolor=LGRAY, edgecolor=TEAL, lw=1.4):
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                                facecolor=facecolor, edgecolor=edgecolor, lw=lw,
                                transform=ax.transAxes, clip_on=False))

def kpi(ax, x, y, w, h, label, value, vc=TEAL, lc=GRAY, fontsize=18):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                                facecolor=LGRAY, edgecolor=TEAL, lw=1.3,
                                transform=ax.transAxes, clip_on=False))
    ax.text(x + w / 2, y + h * 0.65, value, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=vc, transform=ax.transAxes)
    ax.text(x + w / 2, y + h * 0.20, label, ha="center", va="center",
            fontsize=8, color=lc, transform=ax.transAxes)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — What Changed from LP, FT-specific Config & LP vs FT Comparison
# ══════════════════════════════════════════════════════════════════════════════
def page1(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    page_bg(fig)
    banner(fig,
           "Model A  ·  Diabetic Retinopathy Grading  ·  Full Fine-tune",
           "All RETFound-DINOv2 layers updated  ·  Layer-wise LR decay  ·  50 epochs  ·  4-class (R0/R1/R2/R3A)")

    # ── "What changed from LP" explanation box ────────────────────────────────
    ax_ft = fig.add_axes([0.03, 0.70, 0.44, 0.19])
    ax_ft.axis("off"); card_box(ax_ft)
    ax_ft.text(0.04, 0.90, "Upgrade from Linear Probe  →  Full Fine-tune",
               fontsize=10.5, fontweight="bold", color=NAVY, transform=ax_ft.transAxes, va="top")
    txt = ("LP kept the ~307 M backbone FROZEN — only a single linear head\n"
           "was trained, measuring how much structure was already encoded.\n\n"
           "FT UNFREEZES all layers with layer-wise LR decay (factor 0.65):\n"
           "deepest blocks get full LR; shallowest get ~0.0002× LR.\n"
           "This allows non-linear class boundaries that a linear head cannot form.")
    ax_ft.text(0.04, 0.68, txt, fontsize=8.8, color=DKGRAY,
               transform=ax_ft.transAxes, va="top", linespacing=1.55)

    # ── KPI strip — test metrics with LP delta ────────────────────────────────
    ax_k = fig.add_axes([0.52, 0.70, 0.45, 0.19])
    ax_k.axis("off")
    kpi(ax_k, 0.01, 0.05, 0.31, 0.90, f"Test AUROC  (LP {lp_auroc:.3f})",
        f"{t_auroc:.3f}", vc=GREEN)
    kpi(ax_k, 0.35, 0.05, 0.31, 0.90, f"Macro Sensitivity  (LP {lp_sens:.3f})",
        f"{t_sens:.3f}")
    kpi(ax_k, 0.69, 0.05, 0.31, 0.90, f"Macro Specificity  (LP {lp_spec:.3f})",
        f"{t_spec:.3f}")

    # ── FT-only config table (only what differs from LP) ─────────────────────
    ax_cfg = fig.add_axes([0.03, 0.38, 0.44, 0.28])
    ax_cfg.axis("off")
    ax_cfg.text(0, 1.04, "FT Settings  (differences from Linear Probe)",
                fontsize=10.5, fontweight="bold", color=NAVY, transform=ax_cfg.transAxes)
    config = [
        ["Backbone",         "FULLY UNFROZEN  (LP: frozen — no gradient flow)"],
        ["LR scheme",        "Layer-wise decay ×0.65 per depth level"],
        ["Batch size",       "24  (LP: 64 — reduced to fit full backprop GPU memory)"],
        ["Best epoch",       f"{best_epoch}  (LP: 38 — early convergence with unfrozen layers)"],
        ["Epochs trained",   "50  ·  Best ckpt saved by val AUROC"],
        ["Loss / weights",   "CrossEntropyLoss · R0=1.00 R1=1.79 R2=9.53 R3A=15.68"],
        ["Test eval",        "Best ckpt reloaded → single pass on held-out test set"],
    ]
    tbl = ax_cfg.table(cellText=config, cellLoc="left", loc="center",
                       bbox=[0, 0, 1, 0.93])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D8E4EE")
        cell.set_facecolor("#EEF4FA" if r % 2 == 0 else PANEL)
        if c == 0: cell.set_text_props(color=TEAL, fontweight="bold")
        else:      cell.set_text_props(color=DKGRAY)

    # ── LP vs FT test-set comparison table ───────────────────────────────────
    ax_cmp = fig.add_axes([0.52, 0.38, 0.45, 0.28])
    ax_cmp.axis("off")
    ax_cmp.text(0, 1.04, "LP vs FT  ·  Test Set Metrics",
                fontsize=10.5, fontweight="bold", color=NAVY, transform=ax_cmp.transAxes)
    cmp_h = ["Metric", "Linear Probe", "Fine-tune", "Δ"]
    cmp_rows = [
        ["AUROC",           f"{lp_auroc:.3f}", f"{t_auroc:.3f}", f"{t_auroc - lp_auroc:+.3f}"],
        ["Accuracy",        f"{lp_acc:.1%}",   f"{t_acc:.1%}",   f"{t_acc - lp_acc:+.1%}"],
        ["F1  (macro)",     f"{lp_f1:.3f}",    f"{t_f1:.3f}",    f"{t_f1 - lp_f1:+.3f}"],
        ["Kappa",           f"{lp_kappa:.3f}", f"{t_kappa:.3f}", f"{t_kappa - lp_kappa:+.3f}"],
        ["Macro Sensitivity", f"{lp_sens:.3f}", f"{t_sens:.3f}", f"{t_sens - lp_sens:+.3f}"],
        ["Macro Specificity", f"{lp_spec:.3f}", f"{t_spec:.3f}", f"{t_spec - lp_spec:+.3f}"],
    ]
    ctbl = ax_cmp.table(cellText=cmp_rows, colLabels=cmp_h, cellLoc="center",
                        loc="center", bbox=[0, 0, 1, 0.93])
    ctbl.auto_set_font_size(False); ctbl.set_fontsize(9)
    for (r, c), cell in ctbl.get_celld().items():
        cell.set_edgecolor("#D8E4EE")
        if r == 0:
            cell.set_facecolor(NAVY); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0: cell.set_facecolor("#EEF4FA")
        else:            cell.set_facecolor(PANEL)
        if c == 3 and r > 0:
            sign = cmp_rows[r - 1][3][0]
            cell.set_text_props(color=GREEN if sign == "+" else RED, fontweight="bold")

    # ── Per-class sensitivity LP vs FT bar chart ──────────────────────────────
    ax_d = fig.add_axes([0.03, 0.07, 0.92, 0.27])
    ax_d.set_facecolor(PANEL)
    ax_d.text(-0.01, 1.05, "Per-class Sensitivity: Linear Probe vs Fine-tune  (test set)",
              fontsize=10.5, fontweight="bold", color=NAVY, transform=ax_d.transAxes)
    x = np.arange(4); w = 0.33
    bl = ax_d.bar(x - w / 2, lp_ps, w, color=ORANGE, alpha=0.80,
                  edgecolor="white", lw=0.7, label="Linear Probe")
    bf = ax_d.bar(x + w / 2, t_ps,  w, color=TEAL,   alpha=0.85,
                  edgecolor="white", lw=0.7, label="Fine-tune")
    for bar, v in list(zip(bl, lp_ps)) + list(zip(bf, t_ps)):
        ax_d.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                  f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=DKGRAY)
    for i in range(4):
        delta = t_ps[i] - lp_ps[i]
        col   = GREEN if delta > 0 else RED
        ax_d.annotate(f"Δ{delta:+.2f}", xy=(x[i], max(t_ps[i], lp_ps[i]) + 0.09),
                      ha="center", fontsize=9, color=col, fontweight="bold")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(CLASS_FULL, fontsize=9, color=DKGRAY)
    ax_d.set_ylabel("Sensitivity", fontsize=9, color=GRAY)
    ax_d.set_ylim(0, 1.18)
    ax_d.tick_params(colors=GRAY, labelsize=8)
    ax_d.grid(axis="y", linestyle="--", lw=0.5, color="#D0DDE8", alpha=0.8)
    for s in ["top", "right"]:   ax_d.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax_d.spines[s].set_color("#C8D8E8")
    ax_d.axhline(0.70, color="#AABBCC", lw=0.9, ls=":", alpha=0.8)
    ax_d.text(3.5, 0.71, "0.70 ref", fontsize=7, color=GRAY, ha="right")
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
           f"Best checkpoint at epoch {best_epoch} (val AUROC {auroc[best_i]:.4f})  ·  All backbone layers updated each step")

    # Row 1 — AUROC and Loss
    ax_auc = fig.add_axes([0.06, 0.61, 0.40, 0.25])
    styled_ax(ax_auc, title="Val AUROC", ylabel="AUROC", ylim=(0.75, 0.95))
    ax_auc.plot(ep, auroc, color=TEAL, lw=2)
    ax_auc.fill_between(ep, auroc, alpha=0.12, color=TEAL)
    annotate_best(ax_auc, ep, auroc, color=TEAL)

    ax_loss = fig.add_axes([0.57, 0.61, 0.40, 0.25])
    styled_ax(ax_loss, title="Train Loss vs Val Loss", ylabel="Loss")
    ax_loss.plot(ep, train_loss_ep, color=DKGRAY, lw=1.8, alpha=0.75, label="Train (epoch mean)")
    ax_loss.plot(ep, loss_v, color=TEAL, lw=2, label="Val")
    ax_loss.fill_between(ep, loss_v, alpha=0.10, color=TEAL)
    ax_loss.axvline(best_epoch, color=TEAL, lw=0.9, ls=":", alpha=0.55)
    ax_loss.text(best_epoch + 0.5, loss_v.max() * 0.97,
                 f"best ep {best_epoch}", fontsize=7, color=TEAL)
    ax_loss.legend(fontsize=7.5, framealpha=0.85, edgecolor="#D0DDE8")

    # Row 2 — Sensitivity and Specificity
    ax_sens = fig.add_axes([0.06, 0.36, 0.40, 0.21])
    styled_ax(ax_sens, title="Macro Sensitivity over Epochs",
              ylabel="Sensitivity", ylim=(0.50, 0.77))
    ax_sens.plot(ep, m_sens, color=GREEN, lw=2)
    ax_sens.fill_between(ep, m_sens, alpha=0.12, color=GREEN)
    annotate_best(ax_sens, ep, m_sens, color=GREEN)
    ax_sens.axhline(0.5, color="#AABBCC", lw=0.9, ls=":", alpha=0.7)
    ax_sens.text(ep[-1] * 0.55, 0.503, "0.5 reference", fontsize=6.5, color=GRAY)

    ax_spec = fig.add_axes([0.57, 0.36, 0.40, 0.21])
    styled_ax(ax_spec, title="Macro Specificity over Epochs",
              ylabel="Specificity", ylim=(0.85, 0.93))
    ax_spec.plot(ep, m_spec, color=PURPLE, lw=2)
    ax_spec.fill_between(ep, m_spec, alpha=0.12, color=PURPLE)
    bi_sp = int(m_spec.argmax())
    ax_spec.scatter(ep[bi_sp], m_spec[bi_sp], color=PURPLE, s=55, zorder=6)
    ax_spec.annotate(f" Best: {m_spec[bi_sp]:.3f}  (ep {int(ep[bi_sp])})",
                     xy=(ep[bi_sp], m_spec[bi_sp]), fontsize=7.5, color=PURPLE)

    # Row 3 — Accuracy / Kappa and Precision / Recall / F1
    ax_ak = fig.add_axes([0.06, 0.09, 0.40, 0.22])
    styled_ax(ax_ak, title="Accuracy & Cohen's κ", ylabel="Score", ylim=(0.55, 0.85))
    ax_ak.plot(ep, acc,   color=TEAL,   lw=2,   label="Accuracy")
    ax_ak.plot(ep, kappa, color=ORANGE, lw=1.8, ls="--", label="Cohen's κ")
    ax_ak.fill_between(ep, acc,   alpha=0.10, color=TEAL)
    ax_ak.fill_between(ep, kappa, alpha=0.08, color=ORANGE)
    ax_ak.legend(fontsize=7.5, framealpha=0.85, edgecolor="#D0DDE8")
    bi_a = int(acc.argmax())
    ax_ak.scatter(ep[bi_a], acc[bi_a], color=TEAL, s=50, zorder=6)

    ax_prf = fig.add_axes([0.57, 0.09, 0.40, 0.22])
    styled_ax(ax_prf, title="Precision / Recall / F1", ylabel="Score", ylim=(0.45, 0.82))
    ax_prf.plot(ep, prec, color="#4A90D9", lw=1.8, label="Precision")
    ax_prf.plot(ep, rec,  color="#D9534F", lw=1.8, ls="--", label="Recall")
    ax_prf.plot(ep, f1,   color=TEAL,     lw=2.2, label="F1")
    ax_prf.fill_between(ep, f1, alpha=0.10, color=TEAL)
    ax_prf.legend(fontsize=7.5, framealpha=0.85, edgecolor="#D0DDE8", loc="lower right")

    footer(fig, 2)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Per-class Test Sensitivity & Specificity
# ══════════════════════════════════════════════════════════════════════════════
def page3(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    page_bg(fig)
    banner(fig, "Per-class Sensitivity & Specificity  —  Full Fine-tune  (TEST SET)",
           "Sensitivity = TP/(TP+FN)  ·  Specificity = TN/(TN+FP)  ·  OvR decomposition at best-AUROC checkpoint")

    # ── Per-class bar chart (test) ────────────────────────────────────────────
    ax_bar = fig.add_axes([0.05, 0.57, 0.56, 0.30])
    ax_bar.set_facecolor(PANEL)
    x = np.arange(4); w = 0.38
    b1 = ax_bar.bar(x - w / 2, t_ps,    w, color=CLASS_C, alpha=0.85,
                    edgecolor="white", lw=0.8, label="Sensitivity")
    b2 = ax_bar.bar(x + w / 2, t_pspec, w, color=CLASS_C, alpha=0.40,
                    edgecolor=CLASS_C, lw=1.2, label="Specificity", hatch="///")
    for bar, v in list(zip(b1, t_ps)) + list(zip(b2, t_pspec)):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color=DKGRAY)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(CLASS_FULL, fontsize=9, color=DKGRAY)
    ax_bar.set_ylabel("Score", fontsize=9, color=GRAY)
    ax_bar.set_ylim(0, 1.10)
    ax_bar.set_title("Test Set  ·  Sensitivity & Specificity per Class  (held-out, best checkpoint)",
                     fontsize=10, color=NAVY, fontweight="bold", pad=8)
    ax_bar.tick_params(colors=GRAY, labelsize=8)
    ax_bar.grid(axis="y", linestyle="--", lw=0.5, color="#D0DDE8", alpha=0.8)
    for s in ["top", "right"]:   ax_bar.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax_bar.spines[s].set_color("#C8D8E8")
    ax_bar.legend(fontsize=9, framealpha=0.85, edgecolor="#D0DDE8", loc="upper left")
    ax_bar.axhline(0.80, color="#AABBCC", lw=0.9, ls=":", alpha=0.8)
    ax_bar.text(3.5, 0.81, "0.80 ref", fontsize=7, color=GRAY, ha="right")

    # ── Validation per-class sensitivity over epochs ──────────────────────────
    ax_se = fig.add_axes([0.67, 0.57, 0.30, 0.30])
    styled_ax(ax_se, title="Val Sensitivity per Class (epochs)",
              ylabel="Sensitivity", ylim=(0.0, 1.05))
    for i, (c, col) in enumerate(zip(CLASSES, CLASS_C)):
        ax_se.plot(ep, per_sens[:, i], color=col, lw=1.8, label=c, alpha=0.9)
    ax_se.axvline(best_epoch, color=GRAY, lw=0.8, ls=":", alpha=0.6)
    ax_se.legend(fontsize=8, framealpha=0.85, edgecolor="#D0DDE8", loc="lower right")

    # ── Validation per-class specificity over epochs ──────────────────────────
    ax_sp = fig.add_axes([0.67, 0.24, 0.30, 0.28])
    styled_ax(ax_sp, title="Val Specificity per Class (epochs)",
              ylabel="Specificity", ylim=(0.60, 1.05))
    for i, (c, col) in enumerate(zip(CLASSES, CLASS_C)):
        ax_sp.plot(ep, per_spec[:, i], color=col, lw=1.8, label=c, alpha=0.9)
    ax_sp.axvline(best_epoch, color=GRAY, lw=0.8, ls=":", alpha=0.6)
    ax_sp.legend(fontsize=8, framealpha=0.85, edgecolor="#D0DDE8", loc="lower right")

    # ── Summary table ─────────────────────────────────────────────────────────
    ax_tbl = fig.add_axes([0.05, 0.24, 0.58, 0.28])
    ax_tbl.axis("off")
    ax_tbl.text(0, 1.04,
                f"Test Set Metrics  (best checkpoint epoch {best_epoch}, val AUROC {auroc[best_i]:.3f})",
                fontsize=10.5, fontweight="bold", color=NAVY, transform=ax_tbl.transAxes)

    col_h    = ["Class", "Sensitivity", "Specificity", "Clinical Note"]
    tbl_rows = [
        ["R0 — No DR",
         f"{t_ps[0]:.3f}", f"{t_pspec[0]:.3f}",
         "Excellent recall; ~15% false alarm rate acceptable for screening"],
        ["R1 — Mild",
         f"{t_ps[1]:.3f}", f"{t_pspec[1]:.3f}",
         "Major FT gain (+0.309 vs LP); mild DR now well-separated"],
        ["R2 — Moderate",
         f"{t_ps[2]:.3f}", f"{t_pspec[2]:.3f}",
         "Below LP (0.706); small test n=51 inflates per-prediction variance"],
        ["R3A — Active Prolif.",
         f"{t_ps[3]:.3f}", f"{t_pspec[3]:.3f}",
         "Critical gap: 75% of prolif. cases missed on test set (n=48 images)"],
        ["Macro average",
         f"{np.mean(t_ps):.3f}", f"{np.mean(t_pspec):.3f}",
         f"Test AUROC {t_auroc:.3f} · Accuracy {t_acc:.1%} · Kappa {t_kappa:.3f}"],
    ]
    tbl = ax_tbl.table(cellText=tbl_rows, colLabels=col_h, cellLoc="left",
                       loc="center", bbox=[0, 0, 1, 0.90])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
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
            val = float(tbl_rows[r - 1][1])
            cell.set_text_props(color=GREEN if val >= 0.70 else ORANGE if val >= 0.55 else RED,
                                fontweight="bold")
        if c == 2 and r > 0:
            val = float(tbl_rows[r - 1][2])
            cell.set_text_props(color=GREEN if val >= 0.85 else ORANGE if val >= 0.70 else RED,
                                fontweight="bold")

    # ── Clinical context box ───────────────────────────────────────────────────
    ax_cl = fig.add_axes([0.05, 0.05, 0.92, 0.16])
    ax_cl.set_facecolor(TEAL_L); ax_cl.axis("off")
    card_box(ax_cl, facecolor=TEAL_L, edgecolor=TEAL)
    ax_cl.text(0.02, 0.90, "Clinical Interpretation of Fine-tuned Per-class Performance",
               fontsize=10, fontweight="bold", color=NAVY, transform=ax_cl.transAxes, va="top")
    notes = [
        ("R3A sensitivity (0.250)",
         "75% of proliferative cases go undetected — the most dangerous failure. Low test n (48) means each missed prediction moves sensitivity by 0.021; interpret with caution."),
        ("R1 recovery (+0.309 vs LP)",
         "Biggest win of fine-tuning: mild DR jumped from 0.411 to 0.720. Backbone adaptation enables non-linear class boundaries the frozen linear head could not form."),
        ("R2 regression (−0.157 vs LP)",
         "Moderate DR sensitivity dropped from 0.706 to 0.549. Likely the model re-allocates R2 feature space to better separate R1, trading moderate recall for mild recall."),
    ]
    x0, y0 = 0.01, 0.68
    for title_n, body in notes:
        ax_cl.text(x0, y0, f"▸ {title_n}:", fontsize=8.8, fontweight="bold",
                   color=TEAL, transform=ax_cl.transAxes, va="top")
        ax_cl.text(x0 + 0.18, y0, body, fontsize=8.8, color=DKGRAY,
                   transform=ax_cl.transAxes, va="top")
        y0 -= 0.30

    footer(fig, 3)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Convergence Analysis, LR Schedule & Milestone Epochs
# ══════════════════════════════════════════════════════════════════════════════
def page4(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    page_bg(fig)
    plateau_val = auroc[10:].mean()
    banner(fig, "Convergence Analysis  ·  LR Schedule  ·  Milestone Epochs",
           f"Best ckpt at epoch {best_epoch}  ·  Plateau ~{plateau_val:.3f} from epoch 10 onwards  ·  Val loss divergence = overfitting signal")

    # ── Smoothed AUROC convergence ─────────────────────────────────────────────
    ax_auc = fig.add_axes([0.05, 0.58, 0.55, 0.29])
    styled_ax(ax_auc, title="AUROC Convergence  (raw + 5-epoch moving average)",
              ylabel="Val AUROC", ylim=(0.76, 0.94))
    smooth = np.convolve(auroc, np.ones(5) / 5, mode="valid")
    sm_ep  = ep[4:]
    ax_auc.plot(ep, auroc, color=TEAL, lw=1.2, alpha=0.30, label="Raw val AUROC")
    ax_auc.plot(sm_ep, smooth, color=TEAL, lw=2.5, label="5-ep moving avg")
    ax_auc.fill_between(sm_ep, smooth, alpha=0.13, color=TEAL)
    ax_auc.axvspan(0, best_epoch + 1, alpha=0.07, color=GREEN,
                   label=f"Rapid gain (ep 0–{best_epoch})")
    ax_auc.text(1, 0.775, "Rapid\ngain", fontsize=7.5, color=GREEN,
                va="bottom", fontweight="bold")
    ax_auc.axhspan(plateau_val - 0.003, plateau_val + 0.003, alpha=0.12,
                   color=ORANGE, label=f"Plateau band ({plateau_val:.3f})")
    ax_auc.axvline(best_epoch, color=TEAL, lw=1.2, ls="--", alpha=0.7)
    ax_auc.text(best_epoch + 0.4, 0.777, f"Best\nep {best_epoch}", fontsize=7.5,
                color=TEAL, va="bottom")
    ax_auc.legend(fontsize=7.5, framealpha=0.85, edgecolor="#D0DDE8", loc="lower right")

    # ── LR schedule ───────────────────────────────────────────────────────────
    ax_lr = fig.add_axes([0.67, 0.58, 0.30, 0.29])
    styled_ax(ax_lr, title="Learning Rate Schedule", ylabel="LR")
    lr_ep_vals = []
    for e in range(n_epochs):
        lo   = e / n_epochs * tb_lr_steps.max()
        hi   = (e + 1) / n_epochs * tb_lr_steps.max()
        mask = (tb_lr_steps >= lo) & (tb_lr_steps < hi)
        lr_ep_vals.append(tb_lr[mask].mean() if mask.any() else np.nan)
    ax_lr.plot(ep, lr_ep_vals, color=DKGRAY, lw=1.8)
    ax_lr.fill_between(ep, lr_ep_vals, alpha=0.10, color=DKGRAY)
    ax_lr.axvline(best_epoch, color=TEAL, lw=0.9, ls=":", alpha=0.55)
    ax_lr.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1e}"))
    peak_lr = max(v for v in lr_ep_vals if v is not np.nan)
    ax_lr.text(0.5, 0.90, f"Warm-up → cosine decay\nPeak LR: {peak_lr:.2e}",
               ha="center", va="top", fontsize=8, color=DKGRAY,
               transform=ax_lr.transAxes)

    # ── Milestone epoch table ──────────────────────────────────────────────────
    ax_mt = fig.add_axes([0.05, 0.29, 0.92, 0.25])
    ax_mt.axis("off")
    ax_mt.text(0, 1.05, "Key Metrics at Selected Epochs  (validation set)",
               fontsize=10.5, fontweight="bold", color=NAVY, transform=ax_mt.transAxes)
    milestones = [0, 3, best_epoch, 12, 24, 49]
    col_h = ["Epoch", "AUROC", "Accuracy", "Kappa", "Macro Sensitivity", "Macro Specificity",
             "Train Loss", "Val Loss"]
    import json as _json
    log_rows = {}
    with open(os.path.join(ROOT,
              "output_dir/retfound_dinov2_modelA_finetune/retfound_dinov2_modelA_finetune/log.txt")) as f:
        for line in f:
            d = _json.loads(line.strip())
            log_rows[int(d["epoch"])] = float(d["train_loss"])
    m_rows = []
    for e_m in milestones:
        idx  = min(e_m, len(df) - 1)
        mark = " ★" if idx == best_epoch else ""
        m_rows.append([f"Ep {idx}{mark}",
                       f"{auroc[idx]:.4f}", f"{acc[idx]:.4f}", f"{kappa[idx]:.4f}",
                       f"{m_sens[idx]:.4f}", f"{m_spec[idx]:.4f}",
                       f"{log_rows.get(idx, float('nan')):.4f}",
                       f"{loss_v[idx]:.4f}"])
    tbl = ax_mt.table(cellText=m_rows, colLabels=col_h, cellLoc="center",
                      loc="center", bbox=[0, 0, 1, 0.90])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D8E4EE")
        if r == 0:
            cell.set_facecolor(NAVY); cell.set_text_props(color="white", fontweight="bold")
        elif "★" in str(cell.get_text().get_text()):
            cell.set_facecolor("#D4EED4"); cell.set_text_props(color=NAVY, fontweight="bold")
        elif r % 2 == 0: cell.set_facecolor("#EEF4FA")
        else:            cell.set_facecolor(PANEL)

    # ── Key observations ───────────────────────────────────────────────────────
    ax_obs = fig.add_axes([0.05, 0.05, 0.92, 0.21])
    ax_obs.set_facecolor(LGRAY); ax_obs.axis("off")
    card_box(ax_obs, facecolor=LGRAY)
    ax_obs.text(0.02, 0.93, "Key Observations", fontsize=10.5, fontweight="bold",
                color=NAVY, transform=ax_obs.transAxes, va="top")
    obs = [
        f"• AUROC jumped 0.782 → {auroc[best_i]:.3f} in just {best_epoch} epochs — pretrained backbone features activate rapidly when all layers receive gradients.",
        f"• Val AUROC peaked at epoch {best_epoch} then oscillated ~{plateau_val:.3f} while val loss climbed 0.56 → 0.87: a clear overfitting signal. Early stopping (patience ≈ 10 on val AUROC) would save ~42 epochs of compute.",
        f"• Train loss continued declining to 0.337 by epoch 44, confirming the model kept learning on training data after val AUROC plateaued — the gap between train and val loss widens steadily past epoch 10.",
        f"• LR warm-up ramps over the first 10 epochs, peaking at {peak_lr:.2e}, then cosine decays to near-zero — this schedule matches the AUROC curve's rapid early rise followed by diminishing returns.",
        "• Sensitivity and specificity at the best checkpoint (epoch 7) reflect the sharpest moment of generalisation; later epochs improve train metrics but not val/test performance.",
        "• Recommended next step: add early stopping to the fine-tune script to avoid unnecessary epochs; then proceed to Model B (maculopathy binary task).",
    ]
    y0 = 0.80
    for o in obs:
        ax_obs.text(0.02, y0, o, fontsize=8.8, color=DKGRAY,
                    transform=ax_obs.transAxes, va="top", linespacing=1.4)
        y0 -= 0.135

    footer(fig, 4)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ── Build ────────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
with PdfPages(OUT_PDF) as pdf:
    pdf.infodict().update({
        "Title":   "RETFound Model A — Full Fine-tune Report",
        "Author":  "Isaack Joshua",
        "Subject": "Test-set performance after full backbone fine-tuning, with LP vs FT comparison and per-class sensitivity/specificity",
    })
    page1(pdf); page2(pdf); page3(pdf); page4(pdf)

print(f"[DONE] {OUT_PDF}")
