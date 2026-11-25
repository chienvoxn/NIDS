"""Inference for trusted local Final/Tuning artifacts; no training or refitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from nids.sprint3_residual import MultiScaleResidualCNN

VARIANT='cnn_se_mhsa'

class DemoEngine:
    def __init__(self, directory, dataset, profile):
        self.directory=Path(directory)
        self.dataset=dataset
        self.profile=profile
        if dataset not in ('cic','ctu') or profile not in ('full','tuning'):
            raise ValueError('Unsupported dataset/profile')
        report=json.loads((self.directory/('cnn_metrics.json' if profile=='full' else 'tuning_metrics.json')).read_text(encoding='utf-8'))
        if report['dataset']!=dataset or report['profile']!=profile or VARIANT not in report['models']:
            raise ValueError('Report dataset/profile/variant mismatch')
        self.features=list(report['features']); self.labels=list(report['classes'])
        dims=(60,15) if dataset=='cic' else (57,2)
        if (len(self.features),len(self.labels))!=dims or len(set(self.features))!=dims[0]:
            raise ValueError('Unexpected feature/label schema')
        self.normal='BENIGN' if dataset=='cic' else '0'
        if self.normal not in self.labels: raise ValueError('Missing normal label')
        self.prep=joblib.load(self.directory/'cnn_preprocessing.joblib') # trusted local artifacts only
        if list(self.prep['features'])!=self.features or list(self.prep['label_encoder'].classes_)!=self.labels:
            raise ValueError('Preprocessing schema mismatch')
        imp=self.prep['imputer']; scaler=self.prep['scaler']
        if not isinstance(imp,SimpleImputer) or imp.strategy!='median' or imp.add_indicator or not isinstance(scaler,StandardScaler) or not scaler.with_mean or not scaler.with_std:
            raise ValueError('Unsupported preprocessing configuration')
        if not np.isnan(imp.missing_values): raise ValueError('Expected NaN missing values')
        self.median=np.asarray(imp.statistics_,dtype=np.float64)
        self.mean=np.asarray(scaler.mean_,dtype=np.float64)
        self.scale=np.asarray(scaler.scale_,dtype=np.float64)
        if any(a.shape!=(dims[0],) or not np.isfinite(a).all() for a in (self.median,self.mean,self.scale)) or (self.scale<=0).any():
            raise ValueError('Invalid preprocessing statistics')
        if hasattr(imp,'feature_names_in_') and list(imp.feature_names_in_)!=self.features:
            raise ValueError('Imputer feature order mismatch')
        args=report['arguments']; row=report['models'][VARIANT]
        cfg=dict(n_features=dims[0],n_classes=dims[1],width=args['width'],heads=4,blocks=2,dropout=args['dropout'],residual=True,se=True,mhsa=True)
        if row['configuration']!={'se':True,'mhsa':True}: raise ValueError('Variant configuration mismatch')
        if profile=='full':
            ckpt=torch.load(self.directory/f'{VARIANT}.pt',map_location='cpu',weights_only=True)
            for k,v in {**cfg,'dataset':dataset,'variant':VARIANT,'classes':self.labels,'seed':args['seed'],'best_epoch':row['best_epoch']}.items():
                if ckpt[k]!=v: raise ValueError(f'Checkpoint mismatch: {k}')
            state=ckpt['state_dict']
        else:
            if not args['tune_only']: raise ValueError('Expected tuning-only report')
            state=torch.load(self.directory/f'{VARIANT}_best_state.pt',map_location='cpu',weights_only=True)
        self.model=MultiScaleResidualCNN(**cfg)
        self.model.load_state_dict(state,strict=True); self.model.eval()
        self.metadata={'dataset':dataset,'profile':profile,'seed':args['seed'],'best_epoch':row['best_epoch'],'validation_f1':row['validation']['f1_score'],'split_sha256':report['split_sha256']}
    def predict(self, frame, batch_size=128):
        frame=frame.copy(); frame.columns=frame.columns.astype(str).str.strip()
        if frame.columns.duplicated().any() or frame.empty: raise ValueError('Empty data or duplicate columns')
        missing=[x for x in self.features if x not in frame]
        if missing: raise ValueError('Missing features: '+', '.join(missing))
        numeric=frame[self.features].apply(pd.to_numeric,errors='raise')
        numeric=numeric.replace([np.inf,-np.inf],np.nan)
        # Median imputation + standard scaling using fitted statistics only.
        # Avoid version-dependent private sklearn attributes in saved estimators.
        raw=numeric.to_numpy(dtype=np.float64)
        raw=np.where(np.isnan(raw),self.median,raw)
        x=((raw-self.mean)/self.scale).astype(np.float32)
        if x.shape!=(len(frame),len(self.features)) or not np.isfinite(x).all(): raise ValueError('Invalid preprocessed matrix')
        probs=[]
        with torch.inference_mode():
            for i in range(0,len(x),batch_size):
                probs.append(torch.softmax(self.model(torch.from_numpy(x[i:i+batch_size])),1).numpy())
        p=np.concatenate(probs); labels=np.asarray(self.labels)[p.argmax(1)]
        out=pd.DataFrame({'flow_row':np.arange(1,len(frame)+1),'predicted':labels,'confidence':p.max(1),'is_attack':labels!=self.normal})
        for key in ('source_ip','destination_ip','timestamp','start_time','end_time','Source IP','Destination IP','Timestamp','Label','target'):
            if key in frame: out[key]=frame[key].to_numpy()
        return out
