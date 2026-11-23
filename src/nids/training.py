"""CIC pilot baselines and anomaly evaluation. Does not overclaim deployment readiness."""
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from .datasets import CIC_FEATURES,cic_reservoir
from .models import DiagonalGaussian

def run_pilot(zip_path,out,per_label=300,seed=42):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    df,counts=cic_reservoir(zip_path,per_label=per_label,seed=seed)
    if df.empty:raise RuntimeError('No records')
    X=df[CIC_FEATURES].replace([np.inf,-np.inf],np.nan)
    y=df['Label'].to_numpy()
    # Stratified within-source exploratory pilot; not final event/host-aware benchmark.
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=seed)
    models={'RandomForest':RandomForestClassifier(n_estimators=80,min_samples_leaf=2,n_jobs=2,
                                                  class_weight='balanced_subsample',random_state=seed)}
    try:
        from xgboost import XGBClassifier
        models['XGBoost']=XGBClassifier(n_estimators=80,max_depth=5,n_jobs=2,random_state=seed)
    except ImportError:pass
    try:
        from lightgbm import LGBMClassifier
        models['LightGBM']=LGBMClassifier(n_estimators=80,verbose=-1,n_jobs=2,random_state=seed)
    except ImportError:pass
    enc=LabelEncoder().fit(y)
    results=[]
    for name,est in models.items():
        pipe=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=False)),('model',est)])
        tic=time.perf_counter()
        pipe.fit(Xtr,enc.transform(ytr)); elapsed=time.perf_counter()-tic
        predictions=enc.inverse_transform(pipe.predict(Xte))
        item={'model':name,'accuracy':accuracy_score(yte,predictions),
              'macro_f1':f1_score(yte,predictions,average='macro',zero_division=0),'train_s':elapsed,
              'evaluation_protocol':'STRATIFIED_RANDOM_PILOT_NOT_LEAKAGE_SAFE'}
        (out/f'{name}_classification_report.json').write_text(json.dumps(classification_report(yte,predictions,output_dict=True,zero_division=0),indent=2))
        joblib.dump({'pipeline':pipe,'label_encoder':enc,'features':CIC_FEATURES},out/f'{name}.joblib')
        results.append(item)
    # Gaussian is fitted to benign TRAIN only; threshold chosen using TRAIN benign quantile here,
    # not a tuned validation threshold. This is a pilot indication, not a final benchmark.
    benign=Xtr.loc[np.array(ytr)=='BENIGN']
    prep=Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    Xbenign=prep.fit_transform(benign)
    anomaly=DiagonalGaussian().fit(Xbenign)
    threshold=float(np.quantile(anomaly.score_samples(Xbenign),.05))
    score=anomaly.score_samples(prep.transform(Xte))
    truth=(np.asarray(yte)!='BENIGN').astype(int)
    pred=(score<threshold).astype(int)
    from sklearn.metrics import precision_score,recall_score
    gaussian={'precision':precision_score(truth,pred,zero_division=0),
              'recall':recall_score(truth,pred,zero_division=0),
              'fpr':float(((pred==1)&(truth==0)).sum()/max((truth==0).sum(),1)),
              'threshold':threshold,'protocol':'PILOT_TRAIN_BENIGN_5_PERCENT_QUANTILE'}
    joblib.dump({'preprocessor':prep,'gaussian':anomaly,'threshold':threshold,'features':CIC_FEATURES},out/'gaussian.joblib')
    summary={'dataset':'CIC-IDS2017','sample_counts':dict(df.Label.value_counts()),
             'source_counts':counts,'split':'stratified random pilot, leakage risk',
             'note':'Data lacks timestamps/IDs; results are exploratory and must not be used as final generalization estimates.',
             'models':results,'gaussian':gaussian}
    (out/'pilot_results.json').write_text(json.dumps(summary,indent=2,default=float))
    return summary
