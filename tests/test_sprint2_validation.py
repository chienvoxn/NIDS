from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
import pytest
from nids.sprint2_validation import leakage_audit, stability_summary, rare_class_warnings

def sample():
    d=pd.DataFrame({'x':range(80),'y':[i*.2 for i in range(80)],
                    'target':['BENIGN']*40+['Bot']*40})
    return pd.concat([d,d.iloc[[0,79]]],ignore_index=True)

def test_duplicate_aware_zero_overlap():
    r=leakage_audit(sample(),['x','y'],42)
    assert all(v==0 for v in r['exact_duplicate_hash_overlap'].values())
    assert sum(r['split'][x] for x in ('train','validation','test'))==82

def test_seed_changes_splits():
    a=leakage_audit(sample(),['x','y'],42)
    b=leakage_audit(sample(),['x','y'],43)
    assert a['exact_duplicate_hash_overlap']==b['exact_duplicate_hash_overlap']
    assert all('BENIGN' in a['label_counts_by_fold'][f] for f in ('train','validation','test'))

def test_conflicting_duplicate_rejected():
    d=sample();d.loc[79,['x','y']]=d.loc[0,['x','y']]
    with pytest.raises(ValueError,match='conflicting labels'):
        leakage_audit(d,['x','y'],42)

def test_stability_numbers():
    data=[{'metrics':{'rf':{'test':{'macro_f1':n}}}} for n in [0.8,0.9,1.0]]
    r=stability_summary(data)['rf']
    assert r['macro_f1_mean']==pytest.approx(.9)
    assert r['macro_f1_std']==pytest.approx(.1)
    assert r['n_seeds']==3

def test_rare_classes():
    r=rare_class_warnings({'Heartbleed':11,'BENIGN':350},threshold=50)
    assert len(r)==1 and r[0]['label']=='Heartbleed'
