"""Compare independently matched flow reference rows; no automatic CNN approval."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def compare(candidate,reference,features,rtol=1e-6,atol=1e-6):
    if len(features)!=60 or len(set(features))!=60: raise ValueError('Expected 60 distinct features')
    for frame in (candidate,reference):
        if frame.columns.duplicated().any() or frame.empty or frame['match_id'].isna().any() or frame['match_id'].duplicated().any():
            raise ValueError('Nonempty frames with unique nonmissing match_id required')
    a=candidate.set_index('match_id'); b=reference.set_index('match_id')
    if set(a.index)!=set(b.index): raise ValueError('Matched ID sets differ')
    b=b.loc[a.index]; rows=[]
    for name in features:
        x=pd.to_numeric(a[name],errors='raise').to_numpy(dtype=float)
        y=pd.to_numeric(b[name],errors='raise').to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)

        equal = finite & np.isclose(
            x, y, rtol=rtol, atol=atol
        )

        both_nan = np.isnan(x) & np.isnan(y)
        both_posinf = np.isposinf(x) & np.isposinf(y)
        both_neginf = np.isneginf(x) & np.isneginf(y)

        matched = (
            equal
            | both_nan
            | both_posinf
            | both_neginf
        )

        rows.append(dict(
            feature=name,
            matched_rows=len(x),
            finite_pairs=int(finite.sum()),
            within_tolerance=int(equal.sum()),
            finite_mismatches=int((finite & ~equal).sum()),
            matching_nan=int(both_nan.sum()),
            matching_posinf=int(both_posinf.sum()),
            matching_neginf=int(both_neginf.sum()),
            nonfinite_mismatches=int((~finite & ~matched).sum()),
            total_matches=int(matched.sum()),
            total_mismatches=int((~matched).sum()),
            match_rate=float(matched.mean()),
            all_values_match=bool(matched.all()),
            max_abs_error=float(
                np.max(np.abs(x[finite] - y[finite]))
            ) if finite.any() else None
        ))
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',required=True); p.add_argument('--reference',required=True)
    p.add_argument('--metrics',default='outputs/final/cic_final_runs/seed_42/cnn_metrics.json')
    p.add_argument('--out',default='outputs/live_feature_audit.csv'); a=p.parse_args()
    report=json.loads(Path(a.metrics).read_text(encoding='utf-8'))
    if report['dataset']!='cic': raise ValueError('CIC report required')
    result=compare(pd.read_csv(a.candidate),pd.read_csv(a.reference),report['features'])
    path=Path(a.out); path.parent.mkdir(parents=True,exist_ok=True); result.to_csv(path,index=False)
    print('Saved:',path,'; comparison only, not proof of semantic parity or permission for CNN inference')
if __name__=='__main__': main()
