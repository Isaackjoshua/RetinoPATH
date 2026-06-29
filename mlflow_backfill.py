"""Backfill all completed RetinoPATH experiments into a local MLflow file store.

These runs already finished; their final metrics live in output_dir/**/*.json (and,
for P2B/P2E, in CLAUDE.md since no summary JSON was kept). This script *replays* them
into MLflow — one run per experiment — so they're queryable/comparable in `mlflow ui`.
No training, no GPU. Per-epoch curves are not recoverable (only final metrics survived).

Run:   python mlflow_backfill.py            # fresh backfill (clears mlruns/ first)
       python mlflow_backfill.py --append   # keep existing mlruns/, add/refresh runs
View:  mlflow ui --backend-store-uri ./mlruns   then open http://127.0.0.1:5000
"""
import argparse, json, os, shutil, statistics
from pathlib import Path
# MLflow 3.14 put the file store in maintenance mode; opt back in (fine for a static
# backfill logbook). Switch to sqlite:///mlflow.db later if you want newer features.
os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE', 'true')
import mlflow

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
MLRUNS = ROOT / 'mlruns'

CLASSES4 = ['R0', 'R1', 'R2', 'R3A']
CLASSES3 = ['R0', 'R1', 'R2']
CLASSESB = ['M0', 'M1']

COMMON = dict(backbone='RETFound-DINOv2-MEH', backbone_arch='vit_large_patch14_dinov2',
              img_size=224, n_folds=5, seed=42, cv_split='StratifiedKFold (patient-worst grade)',
              cohort='new-data 2026-06-20 (2147 patients / 8844 images)')
FT = dict(method='full fine-tune', batch_size=16, accum_steps=2, base_lr=5e-5,
          llrd_decay=0.75, weight_decay=0.05, warmup_epochs=5, max_epochs=50, patience=10,
          grad_checkpointing=True)

_ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-./:")
def _san(k):
    return ''.join(c if c in _ALLOWED else '_' for c in str(k))

_PERCLASS = {'sensitivity', 'specificity', 'sens', 'spec'}
def flatten(obj, classes, prefix=''):
    """Recursively turn a metrics JSON into flat {metric_name: float}.
    Per-class sens/spec lists are expanded by class name; CIs become _lo/_hi;
    confusion matrices and other lists are skipped (they ride along as JSON artifacts)."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = _san(k) if not prefix else f"{prefix}/{_san(k)}"
            out.update(flatten(v, classes, p))
    elif isinstance(obj, list):
        last = prefix.split('/')[-1].lower()
        if last in _PERCLASS and len(obj) == len(classes) and all(isinstance(x, (int, float)) for x in obj):
            for cn, val in zip(classes, obj):
                out[f"{prefix}_{cn}"] = float(val)
        elif 'ci' in last and len(obj) == 2 and all(isinstance(x, (int, float)) for x in obj):
            out[f"{prefix}_lo"], out[f"{prefix}_hi"] = float(obj[0]), float(obj[1])
        # else: confusion_matrix / class-name lists / etc -> skip
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)
    return out


def fold_stats(paths):
    """mean/min/max OOF AUROC across whatever per-fold JSONs exist."""
    vals = []
    for p in paths:
        if Path(p).exists():
            for f in json.load(open(p)):
                if isinstance(f, dict) and 'oof_auroc' in f:
                    vals.append(f['oof_auroc'])
    if not vals:
        return {}
    return {'cv/mean_oof_auroc': statistics.mean(vals), 'cv/min_oof_auroc': min(vals),
            'cv/max_oof_auroc': max(vals), 'cv/n_folds_run': len(vals)}


# ── Experiment registry ───────────────────────────────────────────────────────
FIG = 'figures'
def figs(*names): return [f'{FIG}/{n}' for n in names]

REGISTRY = [
 dict(exp='ModelA-DR-grading', name='P1-linear-probe-CE', classes=CLASSES4,
      result='baseline', notebook='phase1_cv.ipynb',
      params={**COMMON, 'method': 'linear probe (frozen backbone)', 'loss': 'CE', 'class_weights': 'none'},
      recipe='5-fold CV linear probe, cross-entropy',
      jsons=[('', 'output_dir/phase1_cv/cv_summary.json')],
      folds=['output_dir/phase1_cv/fold_results.json'],
      hl_config='CV test ensemble',
      hl={'test_auroc': 'cv_test_ens/auroc', 'test_kappa': 'cv_test_ens/kappa',
          'test_macro_sens': 'cv_test_ens/macro_sensitivity'}),

 dict(exp='ModelA-DR-grading', name='P2A-focal-linear-probe', classes=CLASSES4,
      result='superseded', notebook='phase2a_focal_loss.ipynb',
      params={**COMMON, 'method': 'linear probe', 'loss': 'focal', 'focal_gamma': 2.0, 'class_weights': 'none'},
      recipe='linear probe + focal loss gamma=2',
      jsons=[('', 'output_dir/phase2a_cv/phase2a_summary.json')],
      folds=['output_dir/phase2a_cv/fold_results.json'],
      hl_config='Phase2A + Argmax (test)',
      hl={'test_auroc': 'test/Phase2A_Argmax/auroc', 'test_macro_sens': 'test/Phase2A_Argmax/macro_sensitivity'}),

 dict(exp='ModelA-DR-grading', name='P2B-full-finetune-RECOMMENDED', classes=CLASSES4,
      result='recommended', notebook='phase2b_full_finetune.ipynb',
      params={**COMMON, **FT, 'loss': 'focal', 'focal_gamma': 2.0,
              'class_weights': '[1.0, 1.796, 10.847, 17.502] inverse-freq', 'aggregation': 'patient-mean + 4-way TTA'},
      recipe='full fine-tune, LLRD 0.75 + grad-ckpt + accum; focal + inverse-freq weights; PtMean+TTA',
      note='Model A production config (new cohort). No summary JSON kept on disk; metrics from CLAUDE.md.',
      manual_metrics={  # documented PtMean + 4-way TTA on test (CLAUDE.md Recommended Configuration)
          'oof/auroc': 0.911, 'oof/kappa': 0.766,
          'test/accuracy': 0.8483, 'test/kappa_quadratic': 0.8501, 'test/auroc': 0.9475,
          'test/macro_sensitivity': 0.7513,
          'test/sensitivity_R0': 0.9769, 'test/sensitivity_R1': 0.7069,
          'test/sensitivity_R2': 0.7500, 'test/sensitivity_R3A': 0.5714,
          'test/auroc_R0': 0.931, 'test/auroc_R1': 0.900, 'test/auroc_R2': 0.980, 'test/auroc_R3A': 0.979},
      hl_config='PtMean + 4-way TTA (production)',
      hl={'test_auroc': 'test/auroc', 'test_kappa': 'test/kappa_quadratic', 'test_macro_sens': 'test/macro_sensitivity'},
      artifacts=figs('confusion_matrix.png', 'roc_curves.png', 'performance_dashboard.png',
                     'precision_recall_curves.png', 'probability_distributions.png',
                     'sensitivity_specificity.png', 'sanity_check_modelA.png')),

 dict(exp='ModelA-DR-grading', name='P2E-weighted-sampler-NEGATIVE', classes=CLASSES4,
      result='negative', notebook='phase2e_balanced_sampling.ipynb',
      params={**COMMON, **FT, 'loss': 'CE', 'sampler': 'WeightedRandomSampler', 'class_weights': 'none'},
      recipe='full fine-tune + WeightedRandomSampler + plain CE',
      note='R1 sensitivity collapsed to 0.000 — model over-fired on R2. Sampler alone not viable.',
      manual_metrics={'test/sensitivity_R1': 0.0},
      artifacts=figs('p2e_confusion_matrix.png', 'p2e_roc_curves.png', 'p2e_sensitivity_comparison.png',
                     'p2e_probability_distributions.png', 'p2e_summary_dashboard.png')),

 dict(exp='ModelA-DR-grading', name='Exp01-normal-finetuning', classes=CLASSES4,
      result='baseline', params={**COMMON, **FT, 'loss': 'CE', 'sampling': 'none', 'class_weights': 'none'},
      recipe='vanilla 4-class full fine-tune (plain CE, uniform LR, no class weights)',
      jsons=[('', 'output_dir/exp01_normal_finetuning_cv/exp01_summary.json')],
      folds=['output_dir/exp01_normal_finetuning_cv/fold_results.json'],
      hl_config='Argmax (test)',
      hl={'test_auroc': 'test/Argmax/auroc', 'test_kappa': 'test/Argmax/kappa', 'test_macro_sens': 'test/Argmax/macro_sensitivity'},
      artifacts=figs('Exp01_Normal_finetuning_cm_test.png')),

 dict(exp='ModelA-DR-grading', name='Exp02-balanced-class-finetuning', classes=CLASSES4,
      result='comparison', params={**COMMON, **FT, 'loss': 'CE', 'class_weights': 'none',
                                    'sampling': 'hybrid under+over to ~1000/class'},
      recipe='balanced 4-class full fine-tune (plain CE, uniform LR, hybrid under+over to ~1000/class)',
      jsons=[('', 'output_dir/exp02_balanced_class_finetuning_cv/exp02_summary.json')],
      folds=['output_dir/exp02_balanced_class_finetuning_cv/fold_results.json'],
      hl_config='Argmax (test)',
      hl={'test_auroc': 'test/Argmax/auroc', 'test_kappa': 'test/Argmax/kappa', 'test_macro_sens': 'test/Argmax/macro_sensitivity'},
      artifacts=figs('Exp02_Balanced_class_finetuning_cm_oof.png', 'Exp02_Balanced_class_finetuning_cm_test.png')),

 dict(exp='ModelA-DR-grading', name='Exp03-bestmodel-hybrid-balancing', classes=CLASSES4,
      result='comparison', params={**COMMON, **FT, 'loss': 'focal', 'focal_gamma': 2.0, 'llrd_decay': 0.75,
                                    'class_weights': 'none', 'sampling': 'hybrid under+over to ~1000/class'},
      recipe='best-model (focal g=2, LLRD 0.75) + hybrid under+over sampling to ~1000/class, NO class weights',
      jsons=[('', 'output_dir/exp03_bestmodel_hybrid_class_balancing_cv/exp03_summary.json'),
             ('tta/', 'output_dir/exp03_bestmodel_hybrid_class_balancing_cv/exp03_tta_results.json')],
      hl_config='PtMax + 4-way TTA (best)',
      hl={'test_auroc': 'tta/PtMax/auroc', 'test_kappa': 'tta/PtMax/kappa', 'test_macro_sens': 'tta/PtMax/macro_sensitivity'},
      artifacts=figs('Exp03_bestmodel_hybrid_class_balancing_cm_oof.png',
                     'Exp03_bestmodel_hybrid_class_balancing_cm_test.png')),

 dict(exp='ModelA-DR-grading', name='P3-res518-NEGATIVE', classes=CLASSES4,
      result='negative', notebook='phase3_res518_*.ipynb',
      params={**COMMON, **FT, 'img_size': 518, 'loss': 'focal', 'focal_gamma': 2.0,
              'class_weights': '[1.0, 1.796, 10.847, 17.502] inverse-freq'},
      recipe='native 518px full fine-tune (single-var: INPUT_SIZE 224->518)',
      note='No gain — 518 ~= 224 on OOF, slightly worse on test (macro-sens 0.726 vs 0.751). Do not repeat.',
      folds=['output_dir/phase3_res518_cv/fold_results.json'],
      hl_config='OOF only (no test JSON kept)', hl={'oof_auroc': 'cv/mean_oof_auroc'}),

 dict(exp='ModelA-DR-grading', name='P4-lesion-multitask-NEGATIVE', classes=CLASSES4,
      result='negative', notebook='phase4_mt_pilot.ipynb / phase4_mt_folds2to4.ipynb',
      params={**COMMON, **FT, 'loss': 'focal + 0.5*BCE(4 lesion features)', 'focal_gamma': 2.0,
              'aux_head': 'binary haem/exud/cws/nvd', 'class_weights': '[1.0, 1.796, 10.847, 17.502] inverse-freq'},
      recipe='shared-backbone aux head (4 binary lesion features), loss = focal(grade) + 0.5*BCE(features)',
      note='At-or-below P2B everywhere (test PtMean+TTA macro-sens 0.719 vs 0.751, kappa 0.833 vs 0.850). Do not repeat.',
      manual_metrics={'test/accuracy': 0.817, 'test/kappa': 0.833, 'test/auroc': 0.942,
                      'test/macro_sensitivity': 0.719, 'test/sensitivity_R0': 0.977,
                      'test/sensitivity_R1': 0.629, 'test/sensitivity_R2': 0.700, 'test/sensitivity_R3A': 0.571},
      folds=['output_dir/phase4_mt_cv/fold_results_pilot.json',
             'output_dir/phase4_mt_cv/fold_results_folds2to4.json'],
      hl_config='PtMean + 4-way TTA',
      hl={'test_auroc': 'test/auroc', 'test_kappa': 'test/kappa', 'test_macro_sens': 'test/macro_sensitivity'}),

 dict(exp='ModelA-DR-grading', name='P5-MAE-backbone-NEGATIVE', classes=CLASSES4,
      result='negative', notebook='phase5_mae_pilot.ipynb',
      params={**COMMON, **FT, 'backbone': 'RETFound-MAE-MEH', 'backbone_arch': 'vit_large_patch16 (global-pool)',
              'loss': 'focal', 'focal_gamma': 2.0},
      recipe='swap backbone DINOv2-MEH -> MAE-MEH (single-var), same P2B pipeline',
      note='MAE ~0.07 AUROC worse than DINOv2 across both pilot folds (0.843 vs 0.911); stopped at 2-fold gate.',
      folds=['output_dir/phase5_mae_cv/fold_results_pilot.json'],
      hl_config='OOF 2-fold pilot only (gated)', hl={'oof_auroc': 'cv/mean_oof_auroc'}),

 dict(exp='ModelA-DR-grading-3class', name='P6-3class-R0R1R2', classes=CLASSES3,
      result='variant', notebook='phase6 3-class pilot',
      params={**COMMON, **FT, 'num_classes': 3, 'classes': 'R0/R1/R2 (R3A patients dropped)',
              'loss': 'focal', 'focal_gamma': 2.0, 'class_weights': 'inverse-freq (3-class)',
              'aggregation': 'patient-mean + 4-way TTA'},
      recipe='3-class (R0/R1/R2) variant of P2B; drop R3A patients, inverse-freq weights',
      jsons=[('tta/', 'output_dir/phase6_3class_cv/phase6_tta_results.json')],
      folds=['output_dir/phase6_3class_cv/fold_results_pilot.json'],
      hl_config='PtMean + 4-way TTA (3-class)',
      hl={'test_auroc': 'tta/PtMean/auroc', 'test_kappa': 'tta/PtMean/kappa', 'test_macro_sens': 'tta/PtMean/macro_sensitivity'}),

 dict(exp='ModelB-maculopathy', name='ModelB-maculopathy-DONE', classes=CLASSESB,
      result='recommended', script='modelB_maculopathy.py',
      params={**COMMON, **FT, 'num_classes': 2, 'task': 'maculopathy M0/M1 (referable)',
              'loss': 'focal', 'focal_gamma': 2.0, 'class_weights': 'none (M1 ~26%, near-balanced)',
              'aggregation': 'patient-mean + 4-way TTA'},
      recipe='binary maculopathy: P2B recipe, 2-class head, focal g=2 no weights; OOF-chosen operating point',
      note='Model B production config. Much cleaner than 4-class DR grading.',
      jsons=[('tta/', 'output_dir/modelB_maculopathy_cv/modelB_tta_results.json')],
      hl_config='PtMean + 4-way TTA (production)',
      hl={'test_auroc': 'tta/PtMean/auroc_test'},
      artifacts=figs('sanity_check_modelB.png')),
]


def log_entry(e):
    mlflow.set_experiment(e['exp'])
    with mlflow.start_run(run_name=e['name']):
        # params
        params = dict(e.get('params', {}))
        params['classes'] = '/'.join(e['classes'])
        mlflow.log_params(params)
        # tags
        tags = {'result': e['result'], 'recipe': e.get('recipe', '')}
        for k in ('note', 'notebook', 'script'):
            if e.get(k):
                tags[k] = e[k]
        mlflow.set_tags(tags)
        # metrics from JSONs
        metrics = {}
        artifacts = list(e.get('artifacts', []))
        for prefix, path in e.get('jsons', []):
            if Path(path).exists():
                metrics.update(flatten(json.load(open(path)), e['classes'], prefix.rstrip('/')))
                artifacts.append(path)            # the source JSON rides along
        metrics.update(fold_stats(e.get('folds', [])))
        metrics.update(e.get('manual_metrics', {}))
        # headline aliases -> common hl/ namespace so all runs sort on one set of columns
        for canon, src in e.get('hl', {}).items():
            if src in metrics:
                metrics[f'hl/{canon}'] = metrics[src]
            else:
                print(f"    WARN {e['name']}: hl source '{src}' not found")
        if e.get('hl_config'):
            mlflow.set_tag('hl_config', e['hl_config'])
        if metrics:
            mlflow.log_metrics(metrics)
        # artifacts (small only; never the 1.2 GB checkpoints)
        for f in e.get('folds', []):
            if Path(f).exists():
                artifacts.append(f)
        logged = 0
        for a in artifacts:
            ap = Path(a)
            if ap.exists() and ap.stat().st_size < 50 * 1024 * 1024:
                mlflow.log_artifact(str(ap))
                logged += 1
        print(f"  [{e['exp']}] {e['name']:38s} params={len(params):2d} metrics={len(metrics):2d} artifacts={logged}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--append', action='store_true', help='keep existing mlruns/ (default wipes for a clean backfill)')
    a = ap.parse_args()
    if not a.append and MLRUNS.exists():
        shutil.rmtree(MLRUNS)
        print(f'cleared {MLRUNS}')
    mlflow.set_tracking_uri(f"file://{MLRUNS}")
    print(f'tracking_uri = {mlflow.get_tracking_uri()}\n')
    for e in REGISTRY:
        log_entry(e)
    print(f'\nDone: {len(REGISTRY)} runs across 3 experiments.\n'
          f'View with:  mlflow ui --backend-store-uri {MLRUNS}\n'
          f'then open http://127.0.0.1:5000')


if __name__ == '__main__':
    main()
