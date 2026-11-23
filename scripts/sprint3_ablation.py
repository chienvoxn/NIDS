#!/usr/bin/env python3
"""CSV-only CIC ablation. Fresh train; does not modify V10 checkpoint.
Example: python scripts/sprint3_ablation.py --per-class 200 --epochs 5 --seeds 42
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from nids.sprint2_data import canonical_cic_label
from nids.sprint2_experiments import bounded_reservoir
from nids.sprint3_residual import ABLATIONS
from nids.sprint3_training import prepare_split, train_variant


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--per-class', type=int, default=200)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--sample-seed', type=int, default=42)
    p.add_argument('--variants', nargs='+', choices=list(ABLATIONS), default=list(ABLATIONS))
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--class-weights', action='store_true')
    p.add_argument('--output', default='reports/sprint3_ablation.json')
    args = p.parse_args(argv)
    if args.per_class < 20 or args.epochs < 1 or args.width % 4 or args.batch_size < 1:
        p.error('per-class >=20; epochs >=1; width divisible by 4; batch-size >=1 required')
    if len(set(args.seeds)) != len(args.seeds):
        p.error('seeds must be distinct')
    manifest = json.loads((ROOT/'models/final/deployment_manifest.json').read_text(encoding='utf-8'))
    features = manifest['feature_order']
    files = sorted((ROOT/'data/raw/CIC-IDS2017').rglob('*.csv'))
    if len(files) != 8:
        p.error(f'Expected 8 CIC-IDS2017 CSV files, found {len(files)}')
    sample, counts = bounded_reservoir(files, features, canonical_cic_label,
                                       cap=args.per_class, seed=args.sample_seed)
    result = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'protocol': 'sampled_internal_duplicate_aware_random_split_CIC_only',
              'sample_rows': len(sample), 'source_class_counts': counts,
              'feature_order': features, 'arguments': vars(args), 'experiments': [],
              'warnings': [
                  'CNN variants use same split and preprocessing within each seed; not independent holdout.',
                  'This is NOT directly comparable to saved V10 CNN or previously reported baseline scores.',
                  'A tabular feature order has no inherent physical neighborhood; convolutional inductive bias requires empirical validation.',
                  'Rare classes may have only a few test rows; per-class metrics are unstable.'
              ]}
    for seed in args.seeds:
        print(f'Seed {seed}: preparing same split for all variants', flush=True)
        parts, classes, audit = prepare_split(sample, features, seed)
        result.setdefault('splits', {})[str(seed)] = audit
        result['label_order'] = classes
        result['class_counts_sampled'] = {k: int(v) for k,v in sample.target.value_counts().items()}
        for variant in args.variants:
            print(f'Seed {seed}: training {variant}', flush=True)
            record = train_variant(parts, len(classes), variant, seed, epochs=args.epochs,
                                   width=args.width, batch_size=args.batch_size,
                                   class_weights=args.class_weights)
            result['experiments'].append(record)
            print(f'  best_val_macro_f1={record["best_val_macro_f1"]:.4f}; test_macro_f1={record["test"]["macro_f1"]:.4f}', flush=True)
    summaries = {}
    for variant in args.variants:
        values = [r['test']['macro_f1'] for r in result['experiments'] if r['variant'] == variant]
        summaries[variant] = {'mean_macro_f1': float(np.mean(values)),
                              'std_macro_f1': float(np.std(values, ddof=1)) if len(values)>1 else None,
                              'n_seeds': len(values)}
    result['summary'] = summaries
    path = Path(args.output)
    if not path.is_absolute(): path = ROOT/path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('Saved:',path)
    return result

if __name__ == '__main__':
    main()
