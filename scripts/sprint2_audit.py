#!/usr/bin/env python3
"""Stream CSV quality audit: no training, no silent relabeling, no dataset rewrite.

Usage from repo root (with PYTHONPATH=src):
  python scripts/sprint2_audit.py
  python scripts/sprint2_audit.py --chunksize 20000
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from nids.sprint2_data import canonical_cic_label, check_cic_header, ctu_label, ctu_numeric_features


def audit_one(path, kind, manifest, chunksize):
    hdr = pd.read_csv(path, nrows=0, encoding='utf-8-sig').columns.str.strip().tolist()
    feature_cols = manifest['feature_order'] if kind == 'cic' else ctu_numeric_features(hdr)
    schema = check_cic_header(hdr, manifest) if kind == 'cic' else {
        'n_features': len(feature_cols), 'label_present': 'Label' in hdr,
        'has_host_time': bool(set(hdr) & {'Src IP', 'Dst IP', 'Timestamp', 'SrcAddr', 'DstAddr', 'StartTime'})}
    info = {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
            'columns': len(hdr), 'schema': schema, 'rows': 0, 'missing_cells_selected_features': 0,
            'infinite_cells_selected_features': 0, 'non_numeric_cells_selected_features': 0,
            'label_counts': {}, 'unknown_labels': {}, 'notes': []}
    counts, bad = Counter(), Counter()
    if not schema['label_present']:
        info['notes'].append('Missing Label column')
        return info
    if kind == 'cic' and schema['missing']:
        info['notes'].append('Missing deployment CNN features: data cannot be passed to checkpoint')
        return info
    for chunk in pd.read_csv(path, chunksize=chunksize, encoding='utf-8-sig', low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        info['rows'] += len(chunk)
        for value, n in chunk['Label'].value_counts(dropna=False).items():
            try:
                norm = canonical_cic_label(value) if kind == 'cic' else ctu_label(value)
                if norm is None:
                    raise ValueError('missing label')
                counts[str(norm)] += int(n)
            except ValueError:
                bad[str(value)] += int(n)
        if not feature_cols:
            continue
        raw = chunk[feature_cols]
        numeric = raw.apply(pd.to_numeric, errors='coerce')
        values = numeric.to_numpy(dtype='float64', na_value=np.nan)
        inf = int(np.isinf(values).sum())
        null = int(np.isnan(values).sum())
        original_null = int(raw.isna().sum().sum())
        info['infinite_cells_selected_features'] += inf
        info['missing_cells_selected_features'] += original_null
        info['non_numeric_cells_selected_features'] += max(0, null-original_null)
    info['label_counts'] = dict(sorted(counts.items()))
    info['unknown_labels'] = dict(sorted(bad.items()))
    if kind == 'ctu':
        info['notes'].append('CTU-13 is binary only; no claim of scenario split, C2 topology or beaconing labels')
    else:
        info['notes'].append('CIC CSV lacks timestamp/IP; no temporal/group split possible with this version')
    info['notes'].append('Duplicate flow audit intentionally not performed: needs a defined key and substantial resources')
    return info


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--chunksize', type=int, default=20000)
    p.add_argument('--output', default='reports/sprint2_data_quality.json')
    args = p.parse_args(argv)
    if args.chunksize < 1:
        p.error('chunksize must be positive')
    root = args.root.resolve()
    manifest = json.loads((root / 'models/final/deployment_manifest.json').read_text(encoding='utf-8'))
    files = sorted((root / 'data/raw').rglob('*.csv'))
    if not files:
        p.error('No CSV files under data/raw')
    results = []
    for f in files:
        rel = f.relative_to(root).as_posix()
        if '/CIC-IDS2017/' in '/'+rel:
            kind = 'cic'
        elif '/CTU-13/' in '/'+rel:
            kind = 'ctu'
        else:
            print('SKIP unknown dataset path:', rel)
            continue
        result = audit_one(f, kind, manifest, args.chunksize)
        results.append({'dataset': 'CIC-IDS2017' if kind=='cic' else 'CTU-13', **result})
        print(f"{f.name}: {result['rows']} rows; {result['columns']} cols; "
              f"missing={result['missing_cells_selected_features']} inf={result['infinite_cells_selected_features']} "
              f"non_numeric={result['non_numeric_cells_selected_features']}", flush=True)
    summary = {'generated_utc': datetime.now(timezone.utc).isoformat(),
               'status': 'audit_only_no_training', 'total_rows': sum(r['rows'] for r in results),
               'files': results,
               'warnings': ['No model fit or evaluation performed',
                            'This does not prove independent holdout evaluation',
                            'Missing/Inf calculated on selected features only, not all CSV columns']}
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print('Saved:', output)
    return summary

if __name__ == '__main__':
    main()
