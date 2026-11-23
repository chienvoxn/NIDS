"""Leakage-aware *internal* CSV pilot experiments; not a temporal/external holdout."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.covariance import EmpiricalCovariance


def bounded_reservoir(files, features, label_fn, cap=400, chunksize=40000, seed=42):
    """Bounded random-priority sample per class over all CSV chunks; reproducible."""
    rng=np.random.default_rng(seed)
    buckets: dict[str,pd.DataFrame] = {}
    counts=defaultdict(int)
    for file in files:
        for chunk in pd.read_csv(file, chunksize=chunksize, low_memory=False, encoding='utf-8-sig'):
            chunk.columns=chunk.columns.str.strip()
            absent=sorted(set(features+['Label'])-set(chunk.columns))
            if absent: raise ValueError(f'{file}: missing {absent}')
            label=chunk['Label'].map(label_fn)
            good=label.notna()
            selected=chunk.loc[good,features].copy()
            selected['target']=label[good].astype(str).to_numpy()
            selected['_priority']=rng.random(len(selected))
            for key, part in selected.groupby('target',sort=False):
                counts[key]+=len(part)
                joined=pd.concat([buckets[key],part],ignore_index=True) if key in buckets else part
                buckets[key]=joined.nsmallest(cap,'_priority')
        print(f'Sampled {file.name}',flush=True)
    if not buckets:raise ValueError('No labeled records')
    result=pd.concat(buckets.values(),ignore_index=True).drop(columns='_priority')
    return result,dict(sorted(counts.items()))


def grouped_stratified_indices(frame, features, target='target', seed=42):
    """Assign exact-duplicate feature rows to one fold; reject conflicting labels."""
    if len(frame)<10:raise ValueError('Dataset too small')
    values=frame[features].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    # stable canonical data value hashing, normalizes NaNs consistently
    fingerprints=pd.util.hash_pandas_object(values,index=False).to_numpy(dtype=np.uint64)
    group_labels=pd.DataFrame({'g':fingerprints,'y':frame[target].astype(str).to_numpy()})
    n_label=group_labels.groupby('g')['y'].nunique()
    conflicting=n_label[n_label>1].index
    if len(conflicting):
        raise ValueError(f'Found {len(conflicting)} identical-feature groups with conflicting labels; audit before training')
    group_frame=group_labels.drop_duplicates('g')
    train_groups=[];val_groups=[];test_groups=[]
    for label,sub in group_frame.groupby('y',sort=True):
        groups=sub['g'].to_numpy()
        if len(groups)<4:
            raise ValueError(f'Class {label!r} has only {len(groups)} unique feature rows; need >=4 for three splits')
        rng=np.random.default_rng(seed + int(hashlib.sha256(label.encode()).hexdigest()[:8],16))
        groups=rng.permutation(groups)
        nt=max(1,round(len(groups)*.15));nv=max(1,round(len(groups)*.15))
        test_groups.extend(groups[:nt]);val_groups.extend(groups[nt:nt+nv]);train_groups.extend(groups[nt+nv:])
    group_sets=[set(g) for g in (train_groups,val_groups,test_groups)]
    if any(group_sets[i]&group_sets[j] for i,j in [(0,1),(0,2),(1,2)]):raise AssertionError('Split overlap')
    masks=[np.flatnonzero(np.isin(fingerprints,list(groups))) for groups in group_sets]
    if any(not len(m) for m in masks):raise ValueError('Empty split')
    return values,tuple(masks),{'split':'stratified_by_label_exact_feature_duplicate_groups','conflicting_feature_groups':0,'unique_feature_groups':len(group_frame), 'train':len(masks[0]),'validation':len(masks[1]),'test':len(masks[2]), 'disclaimer':'Internal duplicate-aware random split, not cross-day, cross-scenario or independent external holdout. Near-duplicates may remain.'}


def scores(y,pred,labels):
    return {'accuracy':float(accuracy_score(y,pred)), 'macro_f1':float(f1_score(y,pred,labels=labels,average='macro',zero_division=0)), 'macro_precision':float(precision_score(y,pred,labels=labels,average='macro',zero_division=0)), 'macro_recall':float(recall_score(y,pred,labels=labels,average='macro',zero_division=0)), 'confusion_matrix':confusion_matrix(y,pred,labels=labels).tolist(), 'classification_report':classification_report(y,pred,labels=labels,output_dict=True,zero_division=0)}


def classifiers(seed=42, enabled=('rf','xgb','lgbm')):
    models={}
    if 'rf' in enabled:models['random_forest']=RandomForestClassifier(n_estimators=120,max_depth=20,min_samples_leaf=2,n_jobs=-1,class_weight='balanced_subsample',random_state=seed)
    if 'xgb' in enabled:
        from xgboost import XGBClassifier
        models['xgboost']=XGBClassifier(n_estimators=160,max_depth=5,learning_rate=.07,subsample=.85,colsample_bytree=.85,n_jobs=4,tree_method='hist',random_state=seed)
    if 'lgbm' in enabled:
        from lightgbm import LGBMClassifier
        models['lightgbm']=LGBMClassifier(n_estimators=160,num_leaves=31,learning_rate=.07,verbosity=-1,n_jobs=4,random_state=seed)
    return models


def run_experiment(sample,features,seed=42,enabled=('rf','xgb','lgbm'),gaussian=True):
    X, (itr,iva,ite), split=grouped_stratified_indices(sample,features,seed=seed)
    y=sample['target'].astype(str).to_numpy(); labels=sorted(set(y))
    # identical imputer/split for every classifier, fit ONLY on train
    from sklearn.preprocessing import LabelEncoder
    enc=LabelEncoder().fit(y[itr]);y_enc=enc.transform(y)
    imp=SimpleImputer(strategy='median',keep_empty_features=True)
    train=imp.fit_transform(X.iloc[itr]); val=imp.transform(X.iloc[iva]); test=imp.transform(X.iloc[ite]);
    output={'n_rows':len(sample),'n_features':len(features),'labels':labels,'class_counts':{v:int(n) for v,n in sample['target'].value_counts().items()},'split':split,'metrics':{},'warning':'Exploratory internal pilot. No claim of independent generalization or 15-class model superiority.'}
    for name,model in classifiers(seed,enabled).items():
        print('Training',name,flush=True)
        model.fit(train,y_enc[itr]); pv=enc.inverse_transform(model.predict(val)); pt=enc.inverse_transform(model.predict(test))
        output['metrics'][name]={'validation':scores(y[iva],pv,labels),'test':scores(y[ite],pt,labels)}
    if gaussian:
        # Diagonal Gaussian trained solely on normal train records; threshold picked from normal validation records
        normal='BENIGN' if 'BENIGN' in labels else '0'
        benign_train=train[y[itr]==normal]
        benign_val=val[y[iva]==normal]
        if len(benign_train)<3 or len(benign_val)<3:raise ValueError('Gaussian requires benign train and validation rows')
        scale=StandardScaler().fit(benign_train)
        xtr=scale.transform(benign_train); xv=scale.transform(benign_val);xt=scale.transform(test)
        center=xtr.mean(0);variance=np.maximum(xtr.var(0),1e-4)
        anomaly=lambda a: ((a-center)**2/variance).sum(axis=1)
        threshold=float(np.quantile(anomaly(xv),.99))
        target=np.where(y[ite]==normal,'NORMAL','ATTACK'); pred=np.where(anomaly(xt)>threshold,'ATTACK','NORMAL')
        output['metrics']['gaussian_anomaly']={'threshold_from_validation_benign_p99':threshold,'test':scores(target,pred,['NORMAL','ATTACK']),'note':'Diagonal Gaussian fitted on train normal, threshold determined only on validation normal; no test tuning.'}
    return output
