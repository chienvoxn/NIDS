import json
import os
from pathlib import Path
import joblib
import pandas as pd
import pytest
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,LabelEncoder
from nids.demo_engine import DemoEngine
from nids.sprint3_residual import MultiScaleResidualCNN

@pytest.fixture
def bundle(tmp_path):
    features=[f'f{i}' for i in range(57)]; frame=pd.DataFrame([[0]*57,[1]*57],columns=features)
    imp=SimpleImputer(strategy="median",keep_empty_features=True).fit(frame); scaler=StandardScaler().fit(imp.transform(frame)); enc=LabelEncoder().fit(['0','1'])
    joblib.dump(dict(features=features,imputer=imp,scaler=scaler,label_encoder=enc),tmp_path/'cnn_preprocessing.joblib')
    cfg=dict(n_features=57,n_classes=2,width=32,heads=4,blocks=2,dropout=.1,residual=True,se=True,mhsa=True)
    model=MultiScaleResidualCNN(**cfg)
    torch.save(dict(**cfg,state_dict=model.state_dict(),dataset='ctu',variant='cnn_se_mhsa',classes=['0','1'],seed=42,best_epoch=1),tmp_path/'cnn_se_mhsa.pt')
    report=dict(dataset='ctu',profile='full',features=features,classes=['0','1'],arguments=dict(width=32,dropout=.1,seed=42),split_sha256='synthetic',models={'cnn_se_mhsa':dict(configuration=dict(se=True,mhsa=True),best_epoch=1,validation=dict(f1_score=.5))})
    (tmp_path/'cnn_metrics.json').write_text(json.dumps(report))
    return tmp_path,frame

def test_predict(bundle):
    path,frame=bundle; engine=DemoEngine(path,'ctu','full'); result=engine.predict(frame)
    assert len(result)==2 and result.confidence.between(0,1).all()
    assert result.equals(engine.predict(frame[frame.columns[::-1]]))

def test_missing(bundle):
    path,frame=bundle; engine=DemoEngine(path,'ctu','full')
    with pytest.raises(ValueError,match='Missing features'): engine.predict(frame.iloc[:,:-1])

def test_wrong_dataset(bundle):
    with pytest.raises(ValueError,match='mismatch'): DemoEngine(bundle[0],'cic','full')

def test_real_ctu_if_available():
    path=Path(os.environ.get('NIDS_CTU_ARTIFACTS','outputs/final/ctu_final_runs/seed_42'))
    if not path.is_dir(): pytest.skip('Local CTU artifacts not available')
    engine=DemoEngine(path,'ctu','full')
    # Synthetic median vector tests loading and inference, not accuracy.
    row=pd.DataFrame([engine.prep['imputer'].statistics_],columns=engine.features)
    assert len(engine.predict(row))==1


def test_preprocessing_equivalence(bundle):
    import numpy as np
    path,frame=bundle; engine=DemoEngine(path,'ctu','full'); frame.iloc[0,0]=np.nan
    x=engine.prep['scaler'].transform(engine.prep['imputer'].transform(frame)).astype(np.float32)
    with torch.inference_mode():
        expected=torch.softmax(engine.model(torch.from_numpy(x)),1).numpy().max(1)
    assert np.allclose(engine.predict(frame).confidence,expected,atol=1e-7)
