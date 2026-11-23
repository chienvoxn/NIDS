from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
import numpy as np
import pytest
from nids.sprint2_experiments import grouped_stratified_indices,run_experiment,bounded_reservoir

def dataset():
 return pd.DataFrame({'a':list(range(100)),'b':np.arange(100)*.1,'target':['0']*50+['1']*50})

def test_disjoint_and_label_coverage():
 df=dataset();v,(tr,va,te),info=grouped_stratified_indices(df,['a','b'])
 assert len(set(tr)&set(va))==0 and len(set(tr)&set(te))==0 and len(set(va)&set(te))==0
 assert set(df.iloc[te]['target'])=={'0','1'}
 assert len(tr)+len(va)+len(te)==100

def test_duplicate_group_same_split():
 df=dataset();df=pd.concat([df,df.iloc[[0,70]]],ignore_index=True)
 _,masks,_=grouped_stratified_indices(df,['a','b'])
 for a,b in [(0,100),(70,101)]:
  assert any(a in m and b in m for m in masks)

def test_conflicting_same_features_rejected():
 df=dataset();df.loc[99,['a','b']]=df.loc[0,['a','b']]
 with pytest.raises(ValueError,match='conflicting labels'):grouped_stratified_indices(df,['a','b'])

def test_train_and_gaussian():
 result=run_experiment(dataset(),['a','b'],enabled=['rf'])
 assert 'random_forest' in result['metrics']
 assert 'gaussian_anomaly' in result['metrics']
 assert result['split']['test']>0

def test_reservoir_cap(tmp_path):
 for i in range(2):
  pd.DataFrame({'x':range(i*100,(i+1)*100),'Label':['0']*50+['1']*50}).to_csv(tmp_path/f'{i}.csv',index=False)
 out,counts=bounded_reservoir(sorted(tmp_path.glob('*.csv')),['x'],lambda x:str(x),cap=10,chunksize=13)
 assert len(out)==20 and counts=={'0':100,'1':100}
