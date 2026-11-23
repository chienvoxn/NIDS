#!/usr/bin/env python3
"""Evaluation of SAVED CNN checkpoint on a CSV sample.

WARNING: model was previously selected using a different split. Without proof
of disjoint training data, this is DIAGNOSTIC, NOT independent test evidence.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from nids.sprint2_data import canonical_cic_label, check_cic_header, map_to_checkpoint_index


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv', default='data/demo/cic_pilot_sample.csv')
    ap.add_argument('--max-rows', type=int, default=5000)
    ap.add_argument('--output', default='reports/sprint2_cnn_diagnostic.json')
    args=ap.parse_args(argv)
    if args.max_rows < 1: ap.error('--max-rows must be positive')
    import torch, joblib
    from nids.deployment_model import TokenizedCNNSEAttention
    manifest=json.loads((ROOT/'models/final/deployment_manifest.json').read_text(encoding='utf-8'))
    path=Path(args.csv)
    if not path.is_absolute(): path=ROOT/path
    frame=pd.read_csv(path,nrows=args.max_rows,low_memory=False)
    frame.columns=frame.columns.str.strip()
    schema=check_cic_header(frame.columns,manifest)
    if schema['missing'] or not schema['label_present']:
        raise ValueError('Input schema incompatibility: '+str(schema))
    modelcfg=manifest['model_config']
    model=TokenizedCNNSEAttention(n_features=manifest['n_features'],n_classes=manifest['n_classes'],
        d_model=modelcfg['d_model'],heads=modelcfg['heads'],dropout=modelcfg['dropout'])
    ckpt=torch.load(ROOT/'models/final/model.pt',map_location='cpu',weights_only=True)
    if not isinstance(ckpt,dict) or 'state_dict' not in ckpt:
        raise ValueError('Expected checkpoint dictionary containing state_dict')
    if ckpt.get('split_sha256') != manifest.get('split_sha256'):
        raise ValueError('Checkpoint/manifest split SHA mismatch')
    if list(ckpt.get('features',[])) != manifest['feature_order']:
        raise ValueError('Checkpoint/manifest features mismatch')
    if list(ckpt.get('classes',[])) != manifest['label_order']:
        raise ValueError('Checkpoint/manifest labels mismatch')
    model.load_state_dict(ckpt['state_dict'],strict=True)
    model.eval()
    prep=joblib.load(ROOT/'models/final/preprocessing.joblib')
    print('Loaded preprocessing object:',type(prep).__name__)
    features=frame[manifest['feature_order']].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    # pipeline contract is verified at runtime: do not assume dictionary layout
    if isinstance(prep,dict):
        print('Preprocessing keys:',sorted(prep))
        if list(prep.get('features',[])) != manifest['feature_order']:
            raise ValueError('Preprocessing/manifest features mismatch')
        if list(prep['label_encoder'].classes_) != manifest['label_order']:
            raise ValueError('Preprocessing/manifest labels mismatch')
        if 'imputer' not in prep or 'scaler' not in prep:
            raise ValueError('Unsupported preprocessing dict: missing imputer/scaler')
        values=prep['scaler'].transform(prep['imputer'].transform(features))
    elif hasattr(prep,'transform'):
        values=prep.transform(features)
    else: raise ValueError('Unsupported preprocessing artifact type')
    if values.shape[1]!=manifest['n_features'] or not np.isfinite(values).all():
        raise ValueError('Preprocessing output shape/nonfinite violation')
    preds=[]
    with torch.inference_mode():
        for start in range(0,len(values),256):
            logits=model(torch.as_tensor(values[start:start+256],dtype=torch.float32))
            if manifest.get('benign_logit_bias'):
                logits[:,manifest['label_order'].index('BENIGN')]+=float(manifest['benign_logit_bias'])
            preds.extend(logits.argmax(1).tolist())
    truth=[]
    for label in frame['Label']:
        truth.append(map_to_checkpoint_index(label,manifest))
    labels=list(range(manifest['n_classes']))
    pretty=[canonical_cic_label(s) for s in manifest['label_order']]
    result={'status':'diagnostic_only_not_independent_holdout',
            'source_csv':str(path),'n_rows':len(truth), 'manifest_split_sha256':manifest.get('split_sha256'),
            'accuracy':float(accuracy_score(truth,preds)),
            'macro_f1':float(f1_score(truth,preds,labels=labels,average='macro',zero_division=0)),
            'classification_report':classification_report(truth,preds,labels=labels,target_names=pretty,output_dict=True,zero_division=0),
            'labels':pretty, 'confusion_matrix':confusion_matrix(truth,preds,labels=labels).tolist(),
            'warning':'Sample metrics must not be described as independent/generalization performance; checkpoint may have seen these records'}
    out=Path(args.output)
    if not out.is_absolute():out=ROOT/out
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Rows:',len(truth),'accuracy:',result['accuracy'],'macro_f1:',result['macro_f1'])
    print('Saved:',out)
    return result
if __name__=='__main__':main()
