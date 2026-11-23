"""Evidence summaries for internal duplicate-aware sampled validation."""
from __future__ import annotations
from collections import Counter
import numpy as np
import pandas as pd
from .sprint2_experiments import grouped_stratified_indices


def leakage_audit(sample, features, seed):
    """Verify exact duplicate groups are disjoint and document proximity caveat."""
    values, indices, original = grouped_stratified_indices(sample, features, seed=seed)
    fingerprints = pd.util.hash_pandas_object(values, index=False).to_numpy(dtype=np.uint64)
    sets = [set(fingerprints[ix].tolist()) for ix in indices]
    overlaps = {f'{a}_{b}': len(sets[i] & sets[j]) for a,b,i,j in [
        ('train','validation',0,1), ('train','test',0,2), ('validation','test',1,2)]}
    assert all(n == 0 for n in overlaps.values()), 'Exact duplicate leakage detected'
    classes = sample['target'].astype(str).to_numpy()
    by_fold = {name: dict(sorted(Counter(classes[ix]).items())) for name,ix in
               zip(('train','validation','test'),indices)}
    return {'split': original, 'exact_duplicate_hash_overlap': overlaps,
            'label_counts_by_fold': by_fold,
            'warnings': ['Exact-row fingerprint disjointness is not evidence of independent collection or no near-duplicate/host/campaign leakage.',
                         'Source CSV has no reliable flow ID, timestamp or host grouping for external or temporal holdout.']}


def stability_summary(results):
    """Summarize only scores from the same pre-selected sampled population."""
    if not results:
        raise ValueError('No experiments')
    models = sorted(set.intersection(*(set(r['metrics']) for r in results)))
    summary = {}
    for model in models:
        values = [float(r['metrics'][model]['test']['macro_f1']) for r in results]
        summary[model] = {'macro_f1_mean': float(np.mean(values)),
                          'macro_f1_std': float(np.std(values,ddof=1)) if len(values)>1 else 0.0,
                          'macro_f1_min': min(values),'macro_f1_max': max(values),
                          'per_seed':values,
                          'n_seeds':len(values)}
    return summary


def rare_class_warnings(class_counts, threshold=50):
    return [{'label':str(k),'sampled_rows':int(v),'issue':f'Fewer than {threshold} sampled rows; per-class metrics are statistically fragile.'}
            for k,v in sorted(class_counts.items()) if int(v)<threshold]
