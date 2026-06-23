"""Generate phase5_mae_pilot.ipynb by cloning phase2b_full_finetune.ipynb and
swapping only: output dir, MAE checkpoint, backbone-loader cell, optimizer call,
and the fold range (pilot folds 0-1). Single-variable vs P2B."""
import copy, json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, 'phase2b_full_finetune.ipynb')
OUT = os.path.join(ROOT, 'phase5_mae_pilot.ipynb')

MAE_LOADER_CELL = '''# ── MAE backbone loader (Phase 5) — imports tested p5_mae module ──
import torch
import numpy as np
from p5_mae import build_classifier, load_pretrained_mae_, build_mae_llrd_optimizer

def load_backbone_fft(device, num_classes=NUM_CLASSES, seed=None):
    """Build RETFound-MAE (ViT-L/16, global-pooled), load MEH weights, full FT.
    Same name as the DINOv2 loader so the CV loop is unchanged. Grad-ckpt is a
    no-op for this backbone; the A6000's 48 GB covers ViT-L FT at batch 16."""
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed)
    model = build_classifier(num_classes=num_classes, img_size=INPUT_SIZE, drop_path_rate=0.2)
    load_pretrained_mae_(model.backbone, repo_id=HF_REPO, filename=HF_FILE, device='cpu')
    for p in model.parameters():
        p.requires_grad = True
    model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'MAE classifier — Trainable {trainable:,}/{total:,}; grad-ckpt OFF')
    return model

print('Verifying MAE backbone load...')
_m = load_backbone_fft(DEVICE)
_out = _m(torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE).to(DEVICE))
assert _out.shape == (2, NUM_CLASSES), _out.shape
del _m, _out
torch.cuda.empty_cache()
print('OK — forward returns logits.')
'''


def sub(cell, a, b):
    cell['source'] = [ln.replace(a, b) for ln in cell['source']]


def main():
    nb = copy.deepcopy(json.load(open(SRC)))
    cells = nb['cells']

    # cell 1: config — output dir + MAE checkpoint
    sub(cells[1], 'output_dir/phase2b_cv', 'output_dir/phase5_mae_cv')
    sub(cells[1], 'YukunZhou/RETFound_dinov2_meh', 'YukunZhou/RETFound_mae_meh')
    sub(cells[1], 'RETFound_dinov2_meh.pth', 'RETFound_mae_meh.pth')

    # cell 6: replace the DINOv2 loader with the MAE loader
    cells[6]['source'] = MAE_LOADER_CELL.splitlines(keepends=True)

    # cell 11: pilot fold range + MAE optimizer + pilot results filename
    sub(cells[11], 'for fold in range(N_FOLDS):', 'for fold in range(2):  # PILOT folds 0-1')
    sub(cells[11], 'build_llrd_optimizer(model, BASE_LR, WEIGHT_DECAY, decay=LLRD_DECAY)',
        'build_mae_llrd_optimizer(model, BASE_LR, WEIGHT_DECAY, decay=LLRD_DECAY)')
    sub(cells[11], "fold_results.json", "fold_results_pilot.json")

    # clear all stale outputs/exec counts
    for c in cells:
        if c['cell_type'] == 'code':
            c['outputs'] = []
            c['execution_count'] = None

    json.dump(nb, open(OUT, 'w'), indent=1)
    print(f'Wrote {OUT}')


if __name__ == '__main__':
    main()
