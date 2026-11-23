#!/usr/bin/env python3
"""Fixed-sample multi-seed internal validation. No changes to trained CNN checkpoint.

Run from NIDS root:
  python scripts/sprint2_final_validation.py --dataset both --per-class 350 --seeds 42 43 44
For fast smoke test:
  python scripts/sprint2_final_validation.py --dataset ctu --models rf --per-class 250 --seeds 42 43
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nids.sprint2_data import canonical_cic_label, ctu_label, ctu_numeric_features
from nids.sprint2_experiments import bounded_reservoir, run_experiment
from nids.sprint2_validation import leakage_audit, stability_summary, rare_class_warnings


def main(argv=None):
    p=argparse.ArgumentParser(description='Internal fixed-sample repeated-seed validation')
    p.add_argument('--dataset',choices=['cic','ctu','both'],default='both')
    p.add_argument('--per-class',type=int,default=350)
    p.add_argument('--chunksize',type=int,default=40000)
    p.add_argument('--sample-seed',type=int,default=42)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--models',nargs='+',choices=['rf','xgb','lgbm'],default=['rf','xgb','lgbm'])
    p.add_argument('--output',default='reports/sprint2_final_validation.json')
    args=p.parse_args(argv)
    if args.per_class<20 or args.chunksize<1 or len(set(args.seeds))!=len(args.seeds):
        p.error('per-class >=20, chunksize >=1 and distinct seeds required')
    manifest=json.loads((ROOT/'models/final/deployment_manifest.json').read_text(encoding='utf-8'))
    output={'generated_utc':datetime.now(timezone.utc).isoformat(),
            'methodology':'fixed_sample_multi_seed_internal_duplicate_aware_random_splits',
            'parameters':vars(args), 'datasets':{},
            'nonclaims':['Not an external or temporal holdout','Not a model-family fair comparison with the pre-existing CNN checkpoint',
                         'Seed-to-seed variation is conditional on one fixed class-balanced sample, not sampling uncertainty over full datasets.']}
    kinds=['cic','ctu'] if args.dataset=='both' else [args.dataset]
    for kind in kinds:
        folder=ROOT/'data/raw'/('CIC-IDS2017' if kind=='cic' else 'CTU-13')
        files=sorted(folder.rglob('*.csv'))
        required_count=8 if kind=='cic' else 2
        if len(files)!=required_count:raise ValueError(f'Expected {required_count} {kind} CSV files, found {len(files)}')
        if kind=='cic':features=manifest['feature_order'];label_fn=canonical_cic_label
        else:
            features=ctu_numeric_features(pd.read_csv(files[0],nrows=0).columns.str.strip().tolist())
            label_fn=ctu_label
        sampled,original_counts=bounded_reservoir(files,features,label_fn,cap=args.per_class,
                                                  chunksize=args.chunksize,seed=args.sample_seed)
        per_seed=[]
        for seed in args.seeds:
            print(f'Validating dataset={kind} seed={seed}',flush=True)
            leakage=leakage_audit(sampled,features,seed)
            result=run_experiment(sampled,features,seed=seed,enabled=args.models,gaussian=True)
            per_seed.append({'seed':seed,'leakage':leakage,'metrics':result['metrics']})
        scored=[{'metrics':x['metrics']} for x in per_seed]
        dataset={'source_files':[str(f.relative_to(ROOT)).replace('\\','/') for f in files],
                 'n_source_rows':int(sum(original_counts.values())),
                 'sample_rows':len(sampled),'features':len(features),
                 'source_class_counts':original_counts,
                 'sampled_class_counts':{str(k):int(v) for k,v in sampled['target'].value_counts().items()},
                 'rare_class_warnings':rare_class_warnings(sampled['target'].value_counts().to_dict()),
                 'per_seed':per_seed,'stability':stability_summary(scored),
                 'limitations':['Class-balanced sampling changes natural prevalence and class metrics',
                                'Cannot rule out near-duplicates or latent attack-family/host leakage',
                                'CTU CSVs do not support scenario-level or temporal holdout' if kind=='ctu' else
                                'CIC CSVs lack verified timestamps/flow groups; random split is not cross-day holdout']}
        output['datasets'][kind]=dataset
        print(f'Completed {kind}: sampled {len(sampled)} rows',flush=True)
    target=Path(args.output); target=target if target.is_absolute() else ROOT/target
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Saved:',target)
    for kind,section in output['datasets'].items():
        for name,v in section['stability'].items():
            print(f'{kind} {name}: macro-F1 {v["macro_f1_mean"]:.4f} +/- {v["macro_f1_std"]:.4f} across {v["n_seeds"]} seeds')
    return output

if __name__=='__main__':main()
