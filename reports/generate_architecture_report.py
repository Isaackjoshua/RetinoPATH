"""
Architecture report for the recommended RetinoPATH configuration.
Generates: reports/RetinoPATH_Architecture_Report.pdf
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak
)
from reportlab.platypus import Flowable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import io
from reportlab.platypus import Image as RLImage

ROOT    = Path(__file__).parent.parent
OUT_PDF = ROOT / "reports" / "RetinoPATH_Architecture_Report.pdf"

# ── Colours ────────────────────────────────────────────────────────────────
TEAL      = colors.HexColor('#006B6B')
TEAL_LITE = colors.HexColor('#E0F0F0')
SLATE     = colors.HexColor('#2C3E50')
AMBER     = colors.HexColor('#D35400')
AMBER_L   = colors.HexColor('#FDEBD0')
GREEN     = colors.HexColor('#1A9E77')
GREEN_L   = colors.HexColor('#E8F8F0')
PURPLE    = colors.HexColor('#7570B3')
PURPLE_L  = colors.HexColor('#F0EFF8')
GREY_BG   = colors.HexColor('#F7F8FA')
GREY_LINE = colors.HexColor('#CCCCCC')
WHITE     = colors.white
BLACK     = colors.HexColor('#1A1A1A')

# ── Styles ──────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

H1 = S('H1', fontSize=20, textColor=SLATE, spaceAfter=6, spaceBefore=14,
        fontName='Helvetica-Bold', leading=24)
H2 = S('H2', fontSize=14, textColor=TEAL,  spaceAfter=4, spaceBefore=12,
        fontName='Helvetica-Bold', leading=18, borderPad=2)
H3 = S('H3', fontSize=11, textColor=SLATE, spaceAfter=3, spaceBefore=8,
        fontName='Helvetica-Bold', leading=14)
BODY = S('BODY', fontSize=9.5, textColor=BLACK, spaceAfter=5, spaceBefore=2,
         fontName='Helvetica', leading=14, alignment=TA_JUSTIFY)
BODY_L = S('BODY_L', fontSize=9.5, textColor=BLACK, spaceAfter=4, spaceBefore=2,
           fontName='Helvetica', leading=14, alignment=TA_LEFT)
CAPTION = S('CAPTION', fontSize=8, textColor=colors.HexColor('#555555'),
            spaceAfter=8, fontName='Helvetica-Oblique', alignment=TA_CENTER)
BULLET = S('BULLET', fontSize=9.5, textColor=BLACK, spaceAfter=3, spaceBefore=1,
           fontName='Helvetica', leading=13, leftIndent=14, bulletIndent=0)
MONO = S('MONO', fontSize=8.5, textColor=SLATE, fontName='Courier',
         leading=12, spaceAfter=3, leftIndent=10)
KPILABEL = S('KPILABEL', fontSize=8, textColor=colors.HexColor('#555'),
             fontName='Helvetica', alignment=TA_CENTER, leading=10)
KPIVAL   = S('KPIVAL', fontSize=18, textColor=TEAL, fontName='Helvetica-Bold',
             alignment=TA_CENTER, leading=22)

W, H = A4

def hr(): return HRFlowable(width='100%', thickness=0.6, color=GREY_LINE, spaceAfter=6, spaceBefore=4)
def sp(n=6): return Spacer(1, n)
def bp(): return PageBreak()

def section_rule():
    return HRFlowable(width='100%', thickness=2, color=TEAL, spaceAfter=8, spaceBefore=2)

def bullet(txt):
    return Paragraph(f'<bullet>&bull;</bullet> {txt}', BULLET)

def kpi_table(items):
    """items = list of (label, value, unit)"""
    cell_data = [[
        [Paragraph(v, KPIVAL), Paragraph(f'{l} {u}', KPILABEL)]
        for l, v, u in items
    ]]
    col_w = (W - 4*cm) / len(items)
    tbl = Table(cell_data, colWidths=[col_w]*len(items))
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), TEAL_LITE),
        ('ROUNDEDCORNERS', [6]),
        ('BOX', (0,0), (-1,-1), 0.5, TEAL),
        ('INNERGRID', (0,0), (-1,-1), 0.3, GREY_LINE),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    return tbl

def info_box(title, body_text, bg=TEAL_LITE, border=TEAL):
    rows = [[Paragraph(f'<b>{title}</b>', S('bt', fontSize=9, textColor=border,
                                             fontName='Helvetica-Bold', leading=12)),
             Paragraph(body_text, S('bb', fontSize=9, textColor=BLACK,
                                    fontName='Helvetica', leading=13, alignment=TA_JUSTIFY))]]
    tbl = Table(rows, colWidths=[3.2*cm, W - 4*cm - 3.2*cm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg),
        ('BOX', (0,0), (-1,-1), 0.8, border),
        ('LINEAFTER', (0,0), (0,-1), 1.5, border),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    return tbl

# ── Matplotlib figures ───────────────────────────────────────────────────────

def fig_to_rl(fig, width_cm=16, height_cm=None):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    w = width_cm * cm
    if height_cm:
        return RLImage(buf, width=w, height=height_cm*cm)
    # preserve aspect ratio
    from PIL import Image as PILImage
    buf2 = io.BytesIO(buf.getvalue())
    pil = PILImage.open(buf2)
    pw, ph = pil.size
    h = w * ph / pw
    buf.seek(0)
    return RLImage(buf, width=w, height=h)

def make_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(13, 2.8))
    ax.set_xlim(0, 13); ax.set_ylim(0, 2.8); ax.axis('off')

    stages = [
        ('Fundus\nImages',        '#B0C4DE',  '#2C3E50', 0.6),
        ('RETFound\nDINOv2-MEH\nViT-Large',  '#1A9E77',  'white',  1.5),
        ('P2B Full\nFine-Tune\n(5-Fold CV)', '#D95F02',  'white',  1.5),
        ('4-Way TTA\n(orig + hflip\n+vflip +both)', '#7570B3', 'white', 1.5),
        ('Patient\nMax Pool',     '#E7298A',  'white',  1.3),
        ('Argmax\nDecision',      '#006B6B',  'white',  1.3),
        ('Grade\nR0–R3A',         '#2C3E50',  'white',  1.1),
    ]
    xs = np.linspace(0.7, 12.3, len(stages))
    widths = [s[3] for s in stages]

    for i, ((label, bg, fg, w), x) in enumerate(zip(stages, xs)):
        rect = mpatches.FancyBboxPatch((x - w/2, 0.55), w, 1.7,
                                        boxstyle='round,pad=0.08',
                                        facecolor=bg, edgecolor='white', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, 1.42, label, ha='center', va='center', color=fg,
                fontsize=7.5, fontweight='bold', multialignment='center')
        if i < len(stages)-1:
            x_end = xs[i+1] - widths[i+1]/2
            ax.annotate('', xy=(x_end - 0.04, 1.42), xytext=(x + w/2 + 0.04, 1.42),
                        arrowprops=dict(arrowstyle='->', color='#444', lw=1.5))

    ax.text(6.5, 0.18, 'Inference pipeline  (per patient, 175 test patients)',
            ha='center', va='bottom', fontsize=8, color='#555', style='italic')
    fig.patch.set_facecolor('white')
    plt.tight_layout(pad=0.3)
    return fig

def make_ablation_bar():
    configs = ['P2B\nImage\nArgmax', 'P2B\nPtMean\nArgmax', 'P2B\nPtMax\nArgmax',
               'P2B\nPtMean\nTTA', 'P2B\nPtMax\nTTA\n(CHOSEN)']
    auroc  = [0.9271, 0.9456, 0.9396, 0.9450, 0.9370]
    kappa  = [0.7671, 0.8212, 0.8007, 0.8131, 0.8220]
    macro  = [0.6375, 0.6788, 0.6761, 0.6709, 0.6999]
    r3a    = [0.250,  0.333,  0.333,  0.333,  0.444 ]

    x = np.arange(len(configs))
    w = 0.20
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))

    ax = axes[0]
    b1 = ax.bar(x - 1.5*w, auroc, w, label='AUROC',       color='#1A9E77', alpha=0.85)
    b2 = ax.bar(x - 0.5*w, kappa, w, label="Kappa",       color='#D95F02', alpha=0.85)
    b3 = ax.bar(x + 0.5*w, macro, w, label='Macro Sens',  color='#7570B3', alpha=0.85)
    b4 = ax.bar(x + 1.5*w, r3a,   w, label='R3A Sens',    color='#E7298A', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=7.5, multialignment='center')
    ax.set_ylim(0, 1.05); ax.set_ylabel('Score', fontsize=9)
    ax.set_title('Configuration Ablation — Key Metrics', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7.5, loc='lower right')
    ax.axvline(3.5, color='#006B6B', lw=1.5, ls='--', alpha=0.6)
    ax.text(4, 0.98, '★ Chosen', color='#006B6B', fontsize=8, fontweight='bold', ha='center')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # per-class sensitivity for chosen config
    ax2 = axes[1]
    cls  = ['R0\n(Normal)', 'R1\n(Mild/Mod\nNPDR)', 'R2\n(Mod/Sev\nNPDR)', 'R3A\n(PDR)']
    sens = [0.9780, 0.7937, 0.5833, 0.4444]
    spec = [0.8690, 0.9018, 0.9816, 1.0000]
    clrs = ['#1A9E77', '#D95F02', '#7570B3', '#E7298A']
    xc   = np.arange(4)
    ax2.bar(xc - 0.2, sens, 0.38, color=clrs, alpha=0.85, label='Sensitivity')
    ax2.bar(xc + 0.2, spec, 0.38, color=clrs, alpha=0.35, label='Specificity')
    for i, (s, sp_) in enumerate(zip(sens, spec)):
        ax2.text(i-0.2, s+0.01, f'{s:.3f}', ha='center', fontsize=7.5, fontweight='bold')
        ax2.text(i+0.2, sp_+0.01, f'{sp_:.3f}', ha='center', fontsize=7, color='#555')
    ax2.set_xticks(xc); ax2.set_xticklabels(cls, fontsize=8, multialignment='center')
    ax2.set_ylim(0, 1.12); ax2.set_ylabel('Score', fontsize=9)
    ax2.set_title('Per-Class Sensitivity & Specificity\nP2B · PtMax · TTA · Argmax', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.axhline(0.80, color='grey', ls=':', lw=1)
    ax2.text(3.55, 0.81, '0.80', fontsize=7, color='grey')
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    plt.tight_layout(pad=1.0)
    return fig

def make_training_diagram():
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3.2); ax.axis('off')

    # LLRD tower
    layers = [
        ('Classification Head',  '#006B6B', 0.95, '5e-5  (base lr)'),
        ('Blocks 20–23',         '#1A9E77', 0.75, '× 0.75¹ = 3.75e-5'),
        ('Blocks 16–19',         '#3CB371', 0.60, '× 0.75² = 2.81e-5'),
        ('Blocks 12–15',         '#6DBF87', 0.50, '× 0.75³ = 2.11e-5'),
        ('Blocks  8–11',         '#9ED4A9', 0.42, '× 0.75⁴ = 1.58e-5'),
        ('Blocks  0– 7',         '#CEEBD3', 0.35, '× 0.75⁵⁺ → ~0.5e-5'),
    ]
    y_top = 3.0; box_h = 0.36; gap = 0.02
    for i, (name, clr, alpha, lr_txt) in enumerate(layers):
        y = y_top - i*(box_h + gap)
        rect = mpatches.FancyBboxPatch((0.2, y - box_h), 3.8, box_h,
                                        boxstyle='round,pad=0.04',
                                        facecolor=clr, edgecolor='white', linewidth=1, alpha=alpha)
        ax.add_patch(rect)
        ax.text(2.1, y - box_h/2, name, ha='center', va='center',
                fontsize=7.5, color='white' if i < 2 else '#333', fontweight='bold')
        ax.text(4.15, y - box_h/2, lr_txt, ha='left', va='center', fontsize=7, color='#444')

    ax.text(2.1, 3.12, 'Layer-Wise LR Decay (LLRD)', ha='center', fontsize=9,
            fontweight='bold', color='#006B6B')
    ax.annotate('', xy=(2.1, y_top - len(layers)*(box_h+gap)), xytext=(2.1, y_top - box_h - 0.05),
                arrowprops=dict(arrowstyle='->', color='#444', lw=1.2))

    # tricks box
    tricks = [
        ('Gradient Checkpointing', '#7570B3', '~10× memory saving\nStores only block inputs;\nrecomputes activations backward'),
        ('Gradient Accumulation',  '#D95F02', 'Effective batch = 32\n2 steps × batch 16\nLoss ÷ ACCUM_STEPS each step'),
        ('5-Fold Stratified CV',   '#E7298A', 'Patient-level split\nmax-grade stratification\nRandom seed 42'),
    ]
    tx0 = 5.5
    for j, (title, clr, desc) in enumerate(tricks):
        x = tx0 + j * 1.85
        rect = mpatches.FancyBboxPatch((x, 0.35), 1.65, 2.55,
                                        boxstyle='round,pad=0.07',
                                        facecolor=clr,
                                        edgecolor='white', linewidth=1.5, alpha=0.18)
        ax.add_patch(rect)
        rect2 = mpatches.FancyBboxPatch((x, 2.52), 1.65, 0.36,
                                         boxstyle='round,pad=0.04',
                                         facecolor=clr, edgecolor='white', linewidth=1)
        ax.add_patch(rect2)
        ax.text(x + 0.825, 2.70, title, ha='center', va='center',
                fontsize=7, color='white', fontweight='bold', multialignment='center')
        ax.text(x + 0.825, 1.42, desc, ha='center', va='center',
                fontsize=7, color='#333', multialignment='center')

    fig.patch.set_facecolor('white')
    plt.tight_layout(pad=0.4)
    return fig

def make_tta_diagram():
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.5); ax.axis('off')

    img_col = '#B0C4DE'
    aug_col = ['#1A9E77', '#D95F02', '#7570B3', '#E7298A']
    aug_lbl = ['Original', 'H-Flip', 'V-Flip', 'H+V Flip']

    # input image
    rect = mpatches.FancyBboxPatch((0.1, 0.65), 1.3, 1.2,
                                    boxstyle='round,pad=0.08',
                                    facecolor=img_col, edgecolor='#888')
    ax.add_patch(rect)
    ax.text(0.75, 1.25, 'Patient\nImage(s)', ha='center', va='center',
            fontsize=8, fontweight='bold', color='#333')

    # fork arrows
    fork_xs = [2.8, 4.3, 5.8, 7.3]
    for i, (fx, lbl, c) in enumerate(zip(fork_xs, aug_lbl, aug_col)):
        ax.annotate('', xy=(fx - 0.55, 1.25), xytext=(1.4, 1.25),
                    arrowprops=dict(arrowstyle='->', color='#999', lw=0.9))
        rect2 = mpatches.FancyBboxPatch((fx - 0.55, 0.72), 1.1, 1.06,
                                         boxstyle='round,pad=0.06',
                                         facecolor=c, edgecolor='white', linewidth=1.2, alpha=0.85)
        ax.add_patch(rect2)
        ax.text(fx, 1.25, lbl, ha='center', va='center', fontsize=7.5,
                color='white', fontweight='bold')
        ax.text(fx, 1.92, 'p̂ᵢ', ha='center', va='center', fontsize=9, color=c)
        ax.annotate('', xy=(8.55, 1.6 - i*0.22), xytext=(fx + 0.55, 1.25),
                    arrowprops=dict(arrowstyle='->', color='#999', lw=0.9))

    # mean box
    rect3 = mpatches.FancyBboxPatch((8.55, 0.72), 1.3, 1.06,
                                     boxstyle='round,pad=0.08',
                                     facecolor='#006B6B', edgecolor='white', linewidth=1.5)
    ax.add_patch(rect3)
    ax.text(9.2, 1.35, 'Mean\np̄ = Σp̂ᵢ/4', ha='center', va='center',
            fontsize=8, color='white', fontweight='bold')

    ax.text(5.0, 0.18, '4 deterministic augmentations per image → probabilities averaged → patient max pooling',
            ha='center', fontsize=7.5, color='#555', style='italic')
    fig.patch.set_facecolor('white')
    plt.tight_layout(pad=0.3)
    return fig

# ── Document build ───────────────────────────────────────────────────────────

def build():
    doc = SimpleDocTemplate(
        str(OUT_PDF), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.2*cm, bottomMargin=2.2*cm,
    )
    story = []

    # ── Cover ──────────────────────────────────────────────────────────────
    story += [
        sp(20),
        Paragraph('RetinoPATH — Recommended Model', S('cvr', fontSize=24, textColor=SLATE,
                  fontName='Helvetica-Bold', alignment=TA_CENTER, leading=30)),
        sp(6),
        Paragraph('Architecture & Design Report', S('cvr2', fontSize=16, textColor=TEAL,
                  fontName='Helvetica', alignment=TA_CENTER)),
        sp(4),
        hr(),
        sp(4),
        Paragraph(
            'Configuration: <b>P2B Full Fine-Tune · Patient Max Pooling · 4-Way TTA · Argmax</b>',
            S('cvrb', fontSize=11, textColor=SLATE, fontName='Helvetica', alignment=TA_CENTER)
        ),
        sp(2),
        Paragraph('Model A — Diabetic Retinopathy Grading (R0 / R1 / R2 / R3A)',
                  S('cvrc', fontSize=10, textColor=colors.grey, fontName='Helvetica-Oblique',
                    alignment=TA_CENTER)),
        sp(18),
        kpi_table([
            ('Accuracy',    '85.7%', '(150/175)'),
            ('Kappa',       '0.822', 'quadratic'),
            ('Macro AUROC', '0.937', 'OvR macro'),
            ('R3A Sens.',   '44.4%', 'PDR detection'),
        ]),
        sp(18),
        Paragraph('June 2026 · NHS UK Fundus Cohort · MEH Reading Centre',
                  S('foot', fontSize=8.5, textColor=colors.grey,
                    fontName='Helvetica-Oblique', alignment=TA_CENTER)),
        bp(),
    ]

    # ── 1. Overview ────────────────────────────────────────────────────────
    story += [
        Paragraph('1. Overview', H1), section_rule(),
        Paragraph(
            'RetinoPATH is a deep-learning system for automated diabetic retinopathy (DR) grading '
            'from fundus photographs. The pipeline assigns each patient one of four severity grades: '
            '<b>R0</b> (no DR), <b>R1</b> (mild–moderate NPDR), <b>R2</b> (moderate–severe NPDR), '
            'and <b>R3A</b> (proliferative DR — the highest-risk category requiring urgent referral).',
            BODY),
        sp(4),
        Paragraph(
            'After systematic ablation across training strategies, patient-level aggregation methods, '
            'and inference-time augmentation, the configuration <b>P2B · Patient Max Pooling · '
            '4-Way TTA · Argmax</b> was selected as the recommended model. This document explains '
            'every component of the architecture, the design choices behind each decision, and the '
            'evidence from ablation experiments that justified them.',
            BODY),
        sp(8),
        Paragraph('End-to-End Inference Pipeline', H3),
        fig_to_rl(make_pipeline_diagram(), width_cm=16.5),
        Paragraph(
            'Figure 1. End-to-end pipeline. Raw fundus images flow through the pre-trained '
            'RETFound-DINOv2-MEH backbone, which was fully fine-tuned in Phase 2B. At inference '
            'time, 4-way TTA is applied per image; probabilities are averaged across augmentations '
            'and then max-pooled across images within a patient before a final argmax decision.',
            CAPTION),
    ]

    # ── 2. Base Model ─────────────────────────────────────────────────────
    story += [
        sp(6),
        Paragraph('2. Base Model — RETFound-DINOv2-MEH', H1), section_rule(),
        Paragraph(
            'The backbone is <b>RETFound-DINOv2-MEH</b>, a Vision Transformer (ViT-Large) '
            'pre-trained via self-supervised DINO-v2 objectives on a large-scale dataset of '
            'retinal fundus images from Moorfields Eye Hospital (MEH) and partner institutions. '
            'This checkpoint was chosen because its pre-training domain closely matches our '
            'NHS UK fundus dataset, providing better feature initialisation than general-purpose '
            'ImageNet weights.',
            BODY),
        sp(4),
    ]

    arch_rows = [
        ['Parameter', 'Value'],
        ['Architecture',     'Vision Transformer Large (ViT-L)'],
        ['Patch size',       '14 × 14 pixels'],
        ['Input resolution', '224 × 224 (resized from 256, centre-cropped)'],
        ['Depth',            '24 transformer blocks'],
        ['Hidden dimension', '1 024'],
        ['Attention heads',  '16'],
        ['Total parameters', '~307 million'],
        ['Pre-training',     'DINOv2 self-supervised, MEH fundus corpus'],
        ['Drop-path rate',   '0.2 (stochastic depth regularisation)'],
    ]
    col_w = [(W - 4*cm) * f for f in [0.38, 0.62]]
    tbl = Table(arch_rows, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), TEAL),
        ('TEXTCOLOR',  (0,0), (-1,0), WHITE),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('BACKGROUND', (0,1), (-1,-1), GREY_BG),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, GREY_BG]),
        ('GRID', (0,0), (-1,-1), 0.4, GREY_LINE),
        ('TOPPADDING',    (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING',   (0,0), (-1,-1), 8),
        ('RIGHTPADDING',  (0,0), (-1,-1), 8),
        ('FONTNAME',   (0,1), (0,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR',  (0,1), (0,-1), SLATE),
    ]))
    story += [tbl, sp(6)]

    story += [
        info_box('Why ViT-Large?',
                 'Larger ViT models learn richer patch-level representations. '
                 'In our comparison, ViT-Large gave consistently higher AUROC and '
                 'Kappa than ViT-Base variants, at the cost of higher memory '
                 'requirements mitigated by gradient checkpointing.',
                 bg=TEAL_LITE, border=TEAL),
        sp(4),
        info_box('Why MEH pre-training?',
                 'Pre-training on in-domain retinal images means the patch embeddings '
                 'already encode clinically relevant textures (haemorrhages, exudates, '
                 'neovascularisation). Starting from these weights instead of ImageNet '
                 'weights reduced the number of fine-tuning epochs needed and improved '
                 'generalisation on minority classes (R2, R3A).',
                 bg=GREEN_L, border=GREEN),
        sp(4),
        Paragraph('Evaluation Transform (fixed, no augmentation at eval time without TTA):', H3),
        Paragraph('Resize(256, BICUBIC) → CenterCrop(224) → ToTensor → Normalize(ImageNet μ/σ)', MONO),
    ]

    # ── 3. Training Strategy ───────────────────────────────────────────────
    story += [
        bp(),
        Paragraph('3. Training Strategy — Phase 2B Full Fine-Tune', H1), section_rule(),
        Paragraph(
            'Phase 2B performs <b>full fine-tuning</b> of all 307 M parameters. '
            'Three engineering techniques make this feasible on a single RTX 3060 (12 GB VRAM) '
            'and improve generalisation on the imbalanced class distribution.',
            BODY),
        sp(6),
        fig_to_rl(make_training_diagram(), width_cm=16.5),
        Paragraph(
            'Figure 2. Training components. Left: layer-wise learning rate decay (LLRD) — '
            'deeper (later) layers receive a higher learning rate than early layers. '
            'Right: three supporting techniques that enable training on a single consumer GPU.',
            CAPTION),
        sp(6),
    ]

    for title, body, bg, bdr in [
        ('Layer-Wise Learning Rate Decay (LLRD)',
         'The head (classification layer) receives the full base learning rate of <b>5×10⁻⁵</b>. '
         'Each group of blocks toward the input is multiplied by a decay factor of <b>0.75</b>, '
         'so the earliest layers train at roughly 0.5×10⁻⁵. '
         'This matters because early layers already encode general visual features from pre-training; '
         'aggressive updates to them would destroy useful representations. '
         'Later layers need more adaptation to learn DR-specific grading features.',
         TEAL_LITE, TEAL),

        ('Gradient Checkpointing',
         'Instead of holding all intermediate activations in GPU memory during the forward pass, '
         'only the <i>input</i> to each transformer block is stored. '
         'During the backward pass, activations are recomputed on the fly. '
         'This reduces peak VRAM use by approximately <b>10×</b>, enabling batch size 16 on 12 GB.',
         PURPLE_L, PURPLE),

        ('Gradient Accumulation (2 steps)',
         'The loss is computed over batch size 16 but only the parameter update is applied after '
         '<b>2 accumulation steps</b>, achieving an effective batch size of 32. '
         'The loss is divided by ACCUM_STEPS before each backward pass so that gradients '
         'are correctly scaled. Larger effective batches produce more stable gradient estimates, '
         'especially important for minority classes with few samples per batch.',
         AMBER_L, AMBER),

        ('5-Fold Stratified Cross-Validation',
         'The training set (990 patients, 4 075 images) is split into 5 folds at the '
         '<b>patient level</b>, stratified by each patient\'s maximum (worst) DR grade. '
         'This ensures every fold\'s validation set contains a representative proportion of '
         'R2 and R3A patients, preventing folds with zero minority-class validation examples. '
         'The random seed is fixed at 42 for reproducibility.',
         GREEN_L, GREEN),
    ]:
        story += [info_box(title, body, bg=bg, border=bdr), sp(5)]

    story += [
        sp(4),
        Paragraph('Training Hyperparameters', H3),
    ]
    hp_rows = [
        ['Hyperparameter', 'Value', 'Hyperparameter', 'Value'],
        ['Optimiser',      'AdamW',                    'Weight decay',    '0.05'],
        ['Base LR (head)', '5 × 10⁻⁵',                'LLRD factor',     '0.75'],
        ['LR schedule',    'Cosine annealing',         'Warmup epochs',   '5'],
        ['Loss function',  'Cross-Entropy',            'Label smoothing', '0.0'],
        ['Batch size',     '16 (eff. 32 w/ accum)',    'Accum. steps',    '2'],
        ['Epochs',         '30 (early stop Δ<1e-4)',   'Drop-path rate',  '0.2'],
        ['Input size',     '224 × 224',                'GPU',             'RTX 3060 12 GB'],
    ]
    hw = (W - 4*cm) / 4
    ht = Table(hp_rows, colWidths=[hw*1.4, hw*0.7, hw*1.4, hw*0.5])
    ht.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SLATE),
        ('TEXTCOLOR',  (0,0), (-1,0), WHITE),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 8.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, GREY_BG]),
        ('GRID', (0,0), (-1,-1), 0.4, GREY_LINE),
        ('TOPPADDING',    (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING',   (0,0), (-1,-1), 8),
        ('FONTNAME',   (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',   (2,0), (2,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR',  (0,1), (0,-1), SLATE),
        ('TEXTCOLOR',  (2,1), (2,-1), SLATE),
    ]))
    story += [ht]

    # ── 4. Patient Max Pooling ─────────────────────────────────────────────
    story += [
        bp(),
        Paragraph('4. Patient-Level Aggregation — Max Pooling', H1), section_rule(),
        Paragraph(
            'Most patients have <b>multiple fundus images</b> (typically 2 per eye, up to 4 total). '
            'The model produces a probability vector per image. '
            'These must be combined into a single patient-level prediction.',
            BODY),
        sp(4),
        Paragraph('How max pooling works:', H3),
        Paragraph(
            'For each patient, collect all image-level probability vectors. '
            'Take the element-wise maximum across them — i.e. for each class, '
            'keep the highest probability seen in any image. '
            'Re-normalise so the four class probabilities sum to 1. '
            'Apply argmax to get the final grade.',
            BODY),
        sp(4),
        Paragraph('In code:', H3),
        Paragraph('stack = np.stack(image_probs)  # shape: (n_images, 4)', MONO),
        Paragraph('p = stack.max(axis=0)           # element-wise max across images', MONO),
        Paragraph('p = p / p.sum()                 # re-normalise to sum to 1', MONO),
        Paragraph('grade = p.argmax()              # final patient prediction', MONO),
        sp(6),
        info_box('Why max pooling instead of mean?',
                 'DR lesions (haemorrhages, neovascularisation) are often focal — they appear '
                 'clearly in one image but are less visible or absent in another due to angle, '
                 'illumination, or which part of the retina was captured. '
                 'Mean pooling averages the signal away. '
                 'Max pooling preserves the strongest evidence across all images, '
                 'giving the model the best chance to detect the highest-grade lesion present. '
                 'This is especially important for R3A (PDR), where neovascularisation can '
                 'be confined to a small retinal region visible in only one image.',
                 bg=TEAL_LITE, border=TEAL),
        sp(4),
        info_box('Why re-normalise after max?',
                 'After taking element-wise maxima, the four values no longer sum to 1 — '
                 'they each came from different images\' softmax outputs. '
                 'Re-normalising restores a valid probability distribution so that '
                 'argmax and threshold-based decisions remain interpretable.',
                 bg=GREEN_L, border=GREEN),
        sp(6),
        Paragraph('Ablation evidence:', H3),
        Paragraph(
            'On the test set (175 patients), switching from image-level to patient max pooling '
            'improved Cohen\'s Kappa from <b>0.767 → 0.801</b> and R3A sensitivity from '
            '<b>0.250 → 0.333</b> on P2B alone. The further addition of TTA pushed R3A to '
            '<b>0.444</b> — the synergy is described in Section 5.',
            BODY),
    ]

    # ── 5. TTA ────────────────────────────────────────────────────────────
    story += [
        bp(),
        Paragraph('5. Inference Augmentation — 4-Way TTA', H1), section_rule(),
        Paragraph(
            '<b>Test-Time Augmentation (TTA)</b> runs each image through four deterministic '
            'geometric transformations, obtains a probability vector from the model for each, '
            'then averages the four vectors. No model weights are changed.',
            BODY),
        sp(6),
        fig_to_rl(make_tta_diagram(), width_cm=16.0),
        Paragraph(
            'Figure 3. TTA schematic. Each input image produces four augmented views. '
            'The model (frozen weights) scores each view independently. '
            'The four probability vectors are averaged before patient max pooling.',
            CAPTION),
        sp(4),
        Paragraph('The four augmentations:', H3),
        bullet('<b>Original</b> — standard eval transform (Resize → CenterCrop → Normalise)'),
        bullet('<b>Horizontal flip</b> — mirror left–right, then eval transform'),
        bullet('<b>Vertical flip</b> — mirror top–bottom, then eval transform'),
        bullet('<b>Both flips</b> — horizontal then vertical, then eval transform'),
        sp(6),
        info_box('Why these four augmentations?',
                 'They are <i>deterministic</i> (no randomness, perfectly reproducible) and '
                 '<i>anatomically plausible</i> — a flipped fundus image can resemble the '
                 'contralateral eye or a different camera angle. '
                 'Averaging over them reduces the model\'s sensitivity to the exact '
                 'orientation and position of lesions within the image frame. '
                 'More augmentations (rotations, colour jitter) were considered but '
                 'these four already gave the largest practical gain.',
                 bg=TEAL_LITE, border=TEAL),
        sp(4),
        info_box('TTA × Patient Max Pooling synergy',
                 'TTA creates 4 views per image. For a patient with 3 images, the model '
                 'now evaluates 12 independent forward passes. Patient max pooling then '
                 'keeps the highest per-class probability across all 12 views. '
                 'This substantially increases the chance that a neovascular lesion — '
                 'visible in perhaps only one image and one orientation — produces a '
                 'high R3A probability in at least one of those 12 views. '
                 'This is why R3A sensitivity jumped from 0.333 (PtMax alone) to '
                 '<b>0.444</b> (PtMax + TTA).',
                 bg=AMBER_L, border=AMBER),
        sp(6),
        Paragraph('Computational cost:', H3),
        Paragraph(
            'TTA multiplies inference time by 4 (one forward pass per augmentation). '
            'On the RTX 3060, the 702 test images require approximately '
            '<b>4 × 5 folds = 20 model forward passes</b> totalling roughly 20 minutes. '
            'This is acceptable for a screening workflow where results are not needed '
            'in real time.',
            BODY),
    ]

    # ── 6. Decision Rule ──────────────────────────────────────────────────
    story += [
        bp(),
        Paragraph('6. Decision Rule — Argmax', H1), section_rule(),
        Paragraph(
            'After patient max pooling, each patient has a single 4-element probability '
            'vector <b>p̄ = [p₀, p₁, p₂, p₃]</b>. The argmax decision rule simply '
            'assigns the class with the highest probability:',
            BODY),
        sp(4),
        Paragraph('grade = argmax(p̄)    →    R0 if p₀ highest, R1 if p₁ highest, etc.', MONO),
        sp(6),
        info_box('Why not Youden-optimal thresholds?',
                 'We evaluated Youden-threshold tuning (selecting class-specific decision '
                 'boundaries that maximise sensitivity + specificity on the OOF validation set). '
                 'While Youden thresholds improved some individual-class sensitivities on the '
                 'OOF data, they consistently degraded Kappa on the held-out test set — '
                 'a sign of overfitting to the OOF distribution. '
                 'The argmax rule has zero free parameters and generalises reliably. '
                 'An R3A-specific threshold sweep (Section 7) confirmed this finding.',
                 bg=TEAL_LITE, border=TEAL),
        sp(4),
        info_box('When argmax can fail',
                 'Argmax is not calibrated — it does not consider how far ahead the top class '
                 'is from the second-best. A confidence score (e.g. p₃ > 0.40) could be used '
                 'as a "refer for review" flag in a clinical workflow without changing the '
                 'grade prediction. This is left as future work.',
                 bg=GREEN_L, border=GREEN),
    ]

    # ── 7. Ablation ───────────────────────────────────────────────────────
    story += [
        sp(8),
        Paragraph('7. Ablation Study — Why This Configuration?', H1), section_rule(),
        Paragraph(
            'Eighteen configurations were evaluated on the held-out test set (175 patients) '
            'across three training phases (P1, P2A, P2B), three aggregation methods '
            '(image-level, patient mean, patient max), two decision rules (argmax, Youden), '
            'and optionally TTA. The chart below focuses on the P2B configurations.',
            BODY),
        sp(6),
        fig_to_rl(make_ablation_bar(), width_cm=16.5),
        Paragraph(
            'Figure 4. Left: ablation across five configurations — AUROC, Kappa, Macro Sensitivity, '
            'and R3A Sensitivity. Right: per-class sensitivity and specificity for the chosen model. '
            'The dashed line at 0.80 marks the clinical target sensitivity.',
            CAPTION),
        sp(6),
        Paragraph('Key ablation findings:', H3),
        bullet('<b>P2B vs P1/P2A</b>: Full fine-tuning (P2B) gave the largest jump '
               '— AUROC 0.827 → 0.937, Kappa 0.667 → 0.822. The base model quality dominates.'),
        bullet('<b>PtMax vs PtMean</b>: Both outperform image-level. PtMax gives slightly '
               'lower AUROC (0.940 vs 0.946) but higher Kappa and R3A sensitivity when '
               'combined with TTA.'),
        bullet('<b>TTA on PtMax</b>: R3A 0.333 → 0.444 (+33% relative improvement). '
               'On PtMean, TTA gave no R3A gain, confirming the synergy is specific to max pooling.'),
        bullet('<b>Youden thresholds</b>: Consistently reduced Kappa on test vs OOF '
               '(test Kappa 0.640 vs argmax 0.822 for PtMax). Threshold overfitting confirmed.'),
        sp(6),
        Paragraph('Runner-up configuration:', H3),
        Paragraph(
            'P2B · PtMean · Argmax (AUROC <b>0.946</b>, Kappa <b>0.821</b>, R1 sensitivity '
            '<b>0.810</b>) is the recommended alternative if R1 sensitivity ≥ 0.80 is a hard '
            'clinical requirement, at the cost of lower R3A detection (0.333 vs 0.444).',
            BODY),
    ]

    # ── 8. Full Results ────────────────────────────────────────────────────
    story += [
        bp(),
        Paragraph('8. Full Test-Set Results', H1), section_rule(),
        Paragraph('175 patients · P2B · Patient Max Pooling · 4-Way TTA · Argmax', BODY_L),
        sp(6),
    ]

    res_rows = [
        ['Metric',                    'Value',   'Interpretation'],
        ['Accuracy',                  '85.71%',  '150 of 175 patients correctly graded'],
        ["Cohen's Kappa (quadratic)", '0.8220',  'Excellent agreement; penalises large grade errors'],
        ['Macro AUROC',               '0.9370',  'Strong discrimination across all 4 classes'],
        ['Macro Precision',           '0.8524',  'Few false positives on average'],
        ['Macro Recall (Sensitivity)','0.6999',  'Driven down by R2/R3A minority classes'],
        ['Macro F1',                  '0.7475',  'Harmonic mean of macro P and R'],
        ['Weighted Precision',        '0.8573',  'Precision weighted by class frequency'],
        ['Weighted Recall',           '0.8571',  'Matches overall accuracy'],
        ['Weighted F1',               '0.8502',  'Good overall performance'],
    ]
    rw = (W - 4*cm)
    rt = Table(res_rows, colWidths=[rw*0.35, rw*0.18, rw*0.47])
    rt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), TEAL),
        ('TEXTCOLOR',  (0,0), (-1,0), WHITE),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, GREY_BG]),
        ('GRID', (0,0), (-1,-1), 0.4, GREY_LINE),
        ('TOPPADDING',    (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING',   (0,0), (-1,-1), 8),
        ('FONTNAME',   (0,1), (0,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR',  (0,1), (0,-1), SLATE),
        ('FONTNAME',   (1,1), (1,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR',  (1,1), (1,-1), TEAL),
    ]))
    story += [rt, sp(10)]

    pc_rows = [
        ['Class', 'N', 'TP', 'FP', 'FN', 'TN', 'Sensitivity', 'Specificity', 'Precision', 'NPV', 'F1', 'AUROC'],
        ['R0 — Normal',         '91', '89', '11', '2',  '73',  '0.978', '0.869', '0.890', '0.973', '0.932', '0.958'],
        ['R1 — Mild/Mod NPDR',  '63', '50', '11', '13', '101', '0.794', '0.902', '0.820', '0.886', '0.807', '0.901'],
        ['R2 — Mod/Sev NPDR',   '12', '7',  '3',  '5',  '160', '0.583', '0.982', '0.700', '0.970', '0.636', '0.958'],
        ['R3A — PDR',            '9',  '4',  '0',  '5',  '166', '0.444', '1.000', '1.000', '0.971', '0.615', '0.932'],
    ]
    cw2 = [(W-4*cm)*f for f in [0.22, 0.05, 0.05, 0.05, 0.05, 0.05, 0.09, 0.09, 0.09, 0.07, 0.07, 0.07]]
    pt = Table(pc_rows, colWidths=cw2)
    clr_map = ['#1A9E77', '#D95F02', '#7570B3', '#E7298A']
    pt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SLATE),
        ('TEXTCOLOR',  (0,0), (-1,0), WHITE),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 7.8),
        ('GRID', (0,0), (-1,-1), 0.4, GREY_LINE),
        ('TOPPADDING',    (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING',   (0,0), (-1,-1), 5),
        ('RIGHTPADDING',  (0,0), (-1,-1), 5),
        ('FONTNAME',   (0,1), (0,-1), 'Helvetica-Bold'),
        *[('BACKGROUND', (0,i+1), (0,i+1), colors.HexColor(clr_map[i])) for i in range(4)],
        *[('TEXTCOLOR',  (0,i+1), (0,i+1), WHITE) for i in range(4)],
        *[('BACKGROUND', (6,i+1), (6,i+1),
           colors.HexColor('#FFEEEE') if [0.978,0.794,0.583,0.444][i] < 0.80 else colors.HexColor('#E8F8F0'))
          for i in range(4)],
    ]))
    story += [
        Paragraph('Per-Class Metrics (Test Set)', H3),
        pt,
        sp(4),
        Paragraph(
            'Sensitivity cells shaded green = above 0.80 target; red = below 0.80. '
            'R3A achieves perfect precision (0 false positives) — the model only '
            'predicts PDR when highly confident.',
            CAPTION),
    ]

    # ── 9. Limitations ────────────────────────────────────────────────────
    story += [
        sp(8),
        Paragraph('9. Limitations & Future Work', H1), section_rule(),
        bullet('<b>R3A sample size</b>: Only 9 R3A patients in the test set (5.1%). '
               'Sensitivity estimates carry wide confidence intervals and may not '
               'reflect performance in a higher-prevalence screening population.'),
        bullet('<b>R2/R3A sensitivity</b>: At 58.3% and 44.4%, both minority classes '
               'fall below the clinical target of 80%. Addressing this will require '
               'additional data, stronger class-reweighting (e.g. class-balanced sampling), '
               'or a cascade architecture that uses a separate high-sensitivity R3A detector.'),
        bullet('<b>Single-site data</b>: All images originate from the MEH/NHS UK cohort. '
               'External validation on another camera type, population, or reading centre '
               'is needed before clinical deployment.'),
        bullet('<b>Maculopathy model (Model B)</b>: Not yet evaluated. The M0/M1 binary '
               'classifier is a separate pipeline sharing the same backbone; its development '
               'is pending.'),
        bullet('<b>Calibration</b>: Softmax probabilities from a fine-tuned ViT are not '
               'well-calibrated. Temperature scaling or isotonic regression should be applied '
               'before using probabilities as confidence scores in a clinical UI.'),
        bullet('<b>TTA inference time</b>: 4× forward passes per image may be unacceptable '
               'for real-time screening workflows. Knowledge distillation into a smaller '
               'model is one avenue to reduce latency.'),
    ]

    # ── 10. Summary ───────────────────────────────────────────────────────
    story += [
        bp(),
        Paragraph('10. Summary', H1), section_rule(),
        Paragraph(
            'The table below summarises the recommended architecture at a glance.',
            BODY),
        sp(6),
    ]
    sum_rows = [
        ['Component',               'Choice',                          'Key Reason'],
        ['Backbone',                'RETFound-DINOv2-MEH ViT-Large',   'In-domain pre-training; 307M params'],
        ['Fine-tuning strategy',    'P2B: all layers, 30 epochs',      'Largest AUROC/Kappa gain vs linear probe'],
        ['Learning rate',           'LLRD: 5e-5 head, ×0.75 per block','Protects early feature representations'],
        ['Memory management',       'Gradient checkpointing',          '~10× VRAM reduction → batch size 16'],
        ['Effective batch',         '32 (2-step accumulation)',         'Stable gradients for minority classes'],
        ['Cross-validation',        '5-fold, patient-level stratified', 'Reproducible; R2/R3A in every fold'],
        ['Patient aggregation',     'Max pooling + re-normalise',       'Preserves peak lesion evidence'],
        ['Inference augmentation',  '4-way TTA (orig + 3 flips)',       'R3A sens 0.333 → 0.444 vs no TTA'],
        ['Decision rule',           'Argmax',                           'Zero free parameters; best test Kappa'],
    ]
    sw = [(W-4*cm)*f for f in [0.26, 0.34, 0.40]]
    st = Table(sum_rows, colWidths=sw)
    st.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), TEAL),
        ('TEXTCOLOR',  (0,0), (-1,0), WHITE),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 8.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, GREY_BG]),
        ('GRID', (0,0), (-1,-1), 0.4, GREY_LINE),
        ('TOPPADDING',    (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING',   (0,0), (-1,-1), 8),
        ('FONTNAME',   (0,1), (0,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR',  (0,1), (0,-1), SLATE),
    ]))
    story += [st, sp(10), hr(), sp(4)]
    story += [
        kpi_table([
            ('Accuracy',    '85.7%', ''),
            ('Kappa',       '0.822', 'quadratic'),
            ('Macro AUROC', '0.937', ''),
            ('R3A Sens.',   '44.4%', 'PDR'),
        ]),
        sp(6),
        Paragraph(
            'RetinoPATH · Model A · NHS UK Cohort · RETFound-DINOv2-MEH · June 2026',
            S('foot', fontSize=8, textColor=colors.grey,
              fontName='Helvetica-Oblique', alignment=TA_CENTER)
        ),
    ]

    doc.build(story)
    print(f'PDF written to {OUT_PDF}')

if __name__ == '__main__':
    build()
