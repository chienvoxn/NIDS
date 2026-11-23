"""Offline utilities with explicit provenance and leakage warnings.

CIC provided ZIP lacks temporal identifiers; flow-only experiments are not
valid demonstrations of temporal attention. CTU uses scenario held-out tests.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             classification_report, confusion_matrix, average_precision_score)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from .botnet import resolve_columns, binary_labels


def metrics(y, pred, score=None, labels=None):
    o = {'accuracy': float(accuracy_score(y, pred)),
         'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
         'weighted_f1': float(f1_score(y, pred, average='weighted', zero_division=0)),
         'report': classification_report(y, pred, labels=labels, output_dict=True, zero_division=0),
         'confusion_matrix': confusion_matrix(y, pred, labels=labels).tolist()}
    if score is not None and len(np.unique(y)) == 2:
        o['pr_auc'] = float(average_precision_score(y, score))
    return o


def ctu_features(df):
    """Flow features from available columns, no IP/time/labels as predictors.

    Preserve endpoints and timestamp separately for graph experiments.
    """
    c=resolve_columns(df)
    required=('label','src','dst','time')
    absent=[k for k in required if c[k] is None]
    if absent: raise ValueError(f'CTU missing columns needed for labeled graph experiment: {absent}; found {list(df.columns)}')
    y=binary_labels(df[c['label']])
    kept=y.notna()
    cols=[k for k in ('duration','bytes','packets','sport','dport') if c.get(k) is not None]
    if not cols:raise ValueError('No numeric flow predictors found')
    X=pd.DataFrame(index=df.index)
    for k in cols:
        X[k]=pd.to_numeric(df[c[k]],errors='coerce')
        if k not in ('sport','dport'):
            X[k]=np.log1p(X[k].clip(lower=0))
    X=X.replace([np.inf,-np.inf],np.nan)
    meta=pd.DataFrame({'src':df[c['src']].astype(str), 'dst':df[c['dst']].astype(str),
                       'time':pd.to_datetime(df[c['time']],errors='coerce',utc=True),
                       'scenario_id':df['scenario_id']})
    return X.loc[kept].reset_index(drop=True), y.loc[kept].astype(int).reset_index(drop=True), meta.loc[kept].reset_index(drop=True)


def causal_graph_features(meta, window_seconds=300):
    """Causal history-only features; graph state is reset for each scenario.

    Feature at time t contains events strictly earlier than t. Repeated rows at
    same timestamp are processed only after feature lookup, avoiding lookahead.
    """
    from collections import deque, Counter
    result=np.zeros((len(meta),4),dtype=np.float32)
    t=pd.to_datetime(meta['time'],utc=True)
    valid=t.notna()
    for scenario, part in meta.loc[valid].groupby('scenario_id',sort=False):
        indices=sorted(part.index, key=lambda i:t.iloc[i])
        queue=deque(); outgoing=Counter(); incoming=Counter(); peers={}
        i=0
        while i<len(indices):
            stamp=t.iloc[indices[i]]
            batch=[]
            while i<len(indices) and t.iloc[indices[i]]==stamp:
                batch.append(indices[i]);i+=1
            cutoff=stamp-pd.Timedelta(seconds=window_seconds)
            while queue and queue[0][0]<cutoff:
                _,a,b=queue.popleft();outgoing[a]-=1;incoming[b]-=1
                peers[a][b]-=1
                if not peers[a][b]:del peers[a][b]
            for idx in batch:
                a,b=str(meta.at[idx,'src']),str(meta.at[idx,'dst'])
                result[idx]=[outgoing[a],incoming[a],len(peers.get(a,{})),outgoing[a]+incoming[a]]
            for idx in batch:
                a,b=str(meta.at[idx,'src']),str(meta.at[idx,'dst'])
                queue.append((stamp,a,b));outgoing[a]+=1;incoming[b]+=1
                if a not in peers:peers[a]=Counter()
                peers[a][b]+=1
    return pd.DataFrame(result,columns=['prior_out_flows','prior_in_flows','prior_unique_dst','prior_total_flows'])


def ctu_holdout_experiment(df, train_scenarios=(5,), val_scenarios=(11,), test_scenarios=(12,), graph=False):
    """Honest scenario holdout; caller must check class coverage in each fold."""
    X,y,meta=ctu_features(df)
    if graph:X=pd.concat([X,causal_graph_features(meta)],axis=1)
    groups=[meta.scenario_id.isin(s).to_numpy() for s in (train_scenarios,val_scenarios,test_scenarios)]
    if any(not mask.any() for mask in groups):raise ValueError('An empty CTU split; verify scenario IDs')
    if any(set(meta.scenario_id[groups[i]]) & set(meta.scenario_id[groups[j]]) for i,j in [(0,1),(0,2),(1,2)]):
        raise ValueError('Overlapping scenario splits')
    if len(np.unique(y[groups[0]]))<2:raise ValueError('Training scenario has <2 classes after excluding Background')
    model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),RandomForestClassifier(n_estimators=100,max_depth=16,min_samples_leaf=3,random_state=42,n_jobs=-1,class_weight='balanced_subsample'))
    model.fit(X[groups[0]],y[groups[0]])
    results={}
    for name,mask in zip(('validation','test'),groups[1:]):
        if len(np.unique(y[mask]))<2:
            results[name]={'warning':'Only one verified class present; PR-AUC not meaningful','class_counts':y[mask].value_counts().to_dict()}
        else:results[name]=metrics(y[mask],model.predict(X[mask]),model.predict_proba(X[mask])[:,1])
    return model,{'task':'ctu_botnet','feature_names':X.columns.tolist(),'splits':{'train':list(train_scenarios),'validation':list(val_scenarios),'test':list(test_scenarios)},'counts':{str(x):int(v) for x,v in df.groupby('scenario_id').size().items()},'results':results}


def time_ordered_windows(df, feature_cols, group_col, time_col, label_col, length=16):
    """Construct sequences only with actual, parseable timestamps and source group IDs.
    One sequence is emitted for each valid observed event once enough history exists.
    All rows within the window must come from the same externally supplied split.
    """
    if any(c not in df.columns for c in [group_col,time_col,label_col]+feature_cols):
        raise ValueError('Missing required sequence columns')
    work=df.copy()
    work[time_col]=pd.to_datetime(work[time_col],errors='coerce',utc=True)
    if work[time_col].isna().any():raise ValueError('Unparseable timestamps')
    xs=[];ys=[];endpoints=[]
    for group,part in work.groupby(group_col,sort=False):
        ordered=part.sort_values(time_col,kind='stable')
        vals=ordered[feature_cols].to_numpy(dtype=np.float32)
        if not np.isfinite(vals).all():raise ValueError('Preprocess nonfinite values before creating windows')
        lbl=ordered[label_col].to_numpy()
        for end in range(length-1,len(ordered)):
            xs.append(vals[end-length+1:end+1]);ys.append(lbl[end]);endpoints.append((str(group),str(ordered.iloc[end][time_col])))
    if not xs:raise ValueError('No full-length chronological windows exist')
    return np.stack(xs),np.asarray(ys),endpoints


def ctu_sequence_dataset(df,length=16,features=None):
    """Build chronological CTU sequences within scenario/source host boundaries.
    Labels use last observed flow; this is a simple pilot labeling policy.
    """
    X,y,meta=ctu_features(df)
    if features is None:features=X.columns.tolist()
    # Imputation is deliberately NOT fit here; train-only imputation done by caller.
    merged=meta.copy()
    merged['source_group']=merged['scenario_id'].astype(str)+'|'+merged['src'].astype(str)
    merged['target']=y.to_numpy()
    for col in features:merged[col]=X[col].to_numpy()
    return merged,features


def train_ctu_torch_temporal(df,train=(5,),validation=(11,),test=(12,),length=16,epochs=5,seed=42,out=None):
    """CTU temporal experiment with scenario holdout and train-only statistics.
    Requires scenarios with both label classes; a zero-class validation split is reported.
    """
    import torch
    from torch import nn
    from .models import SEATCNN
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    torch.manual_seed(seed)
    merged,feature_cols=ctu_sequence_dataset(df,length)
    train_rows=merged['scenario_id'].isin(train)
    imp=SimpleImputer(strategy='median')
    scaler=StandardScaler()
    scaler.fit(imp.fit_transform(merged.loc[train_rows,feature_cols]))
    merged.loc[:,feature_cols]=scaler.transform(imp.transform(merged[feature_cols]))
    datasets={}
    for key,scenarios in [('train',train),('validation',validation),('test',test)]:
        part=merged[merged.scenario_id.isin(scenarios)].copy()
        if part.empty:raise ValueError(f'Empty {key} split')
        xx,yy,_=time_ordered_windows(part,feature_cols,'source_group','time','target',length)
        datasets[key]=(torch.tensor(xx,dtype=torch.float32),torch.tensor(yy.astype(int),dtype=torch.long))
    model=SEATCNN(len(feature_cols),2,hidden=32)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    criterion=nn.CrossEntropyLoss()
    from torch.utils.data import DataLoader,TensorDataset
    loader=DataLoader(TensorDataset(*datasets['train']),batch_size=128,shuffle=True)
    history=[];best_f1=-1;best_state=None
    for epoch in range(epochs):
        model.train();loss_sum=0
        for xx,yy in loader:
            opt.zero_grad();logits=model(xx);loss=criterion(logits,yy);loss.backward();opt.step();loss_sum+=float(loss)*len(yy)
        model.eval()
        with torch.no_grad():
            xx,yy=datasets['validation'];pred=model(xx).argmax(1).numpy()
        f1=float(f1_score(yy.numpy(),pred,average='macro',zero_division=0))
        history.append({'epoch':epoch+1,'train_loss':loss_sum/len(datasets['train'][1]),'val_macro_f1':f1})
        if f1>best_f1:
            best_f1=f1;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state);model.eval()
    xx,yy=datasets['test']
    with torch.no_grad():
        logits=model(xx);prob=torch.softmax(logits,1)[:,1].numpy();pred=logits.argmax(1).numpy()
    report={'warning':'CTU subset only; reported metrics require checking class coverage and cross-family generalization',
            'history':history,'test':metrics(yy.numpy(),pred,prob if len(np.unique(yy))==2 else None),
            'samples':{k:len(v[1]) for k,v in datasets.items()},
            'features':feature_cols,'seq_length':length,'split':{'train':list(train),'val':list(validation),'test':list(test)}}
    if out:
        from pathlib import Path
        import joblib
        dest=Path(out);dest.mkdir(parents=True,exist_ok=True)
        torch.save({'state_dict':model.state_dict(),'features':feature_cols,'seq_length':length,'classes':2},dest/'ctu_se_at_cnn.pt')
        joblib.dump({'imputer':imp,'scaler':scaler,'features':feature_cols},dest/'ctu_sequence_preprocessing.joblib')
        (dest/'ctu_se_at_cnn_results.json').write_text(json.dumps(report,indent=2))
    return model,report
