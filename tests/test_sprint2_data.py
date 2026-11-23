from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
import numpy as np
import pytest
from nids.sprint2_data import (canonical_cic_label, map_to_checkpoint_index,
    check_cic_header, ctu_label, ctu_numeric_features, sanitize_numeric_chunk)

MANIFEST={'label_order':['BENIGN','Web Attack ï¿½ Brute Force','Web Attack ï¿½ Sql Injection','Web Attack ï¿½ XSS']}

@pytest.mark.parametrize('raw,expected',[
  ('Web Attack � Brute Force','Web Attack - Brute Force'),
  ('Web Attack ï¿½ Brute Force','Web Attack - Brute Force'),
  ('Web Attack – XSS','Web Attack - XSS'),
  ('Web Attack - Sql Injection','Web Attack - Sql Injection'),
  ('BENIGN','BENIGN'),(' DDoS ','DDoS')])
def test_label_variants(raw,expected):
    assert canonical_cic_label(raw)==expected

def test_unknown_web_label_rejected():
    with pytest.raises(ValueError):canonical_cic_label('Web Attack - UnseenType')

def test_checkpoint_indices_immutable():
    assert map_to_checkpoint_index('Web Attack � XSS',MANIFEST)==3
    assert map_to_checkpoint_index('Web Attack ï¿½ Brute Force',MANIFEST)==1

def test_schema_missing_reported():
    result=check_cic_header(['a','Label'],{'feature_order':['a','b']})
    assert result['missing']==['b'] and result['available']==1

def test_ctu_binary():
    assert ctu_label(1)==1 and ctu_label('0')==0
    with pytest.raises(ValueError):ctu_label(2)

def test_ctu_index_dropped():
    assert ctu_numeric_features(['Unnamed: 0','Flow Duration','Label'])==['Flow Duration']

def test_nonfinite_no_fit():
    df=pd.DataFrame({'x':[1,np.inf,'bad']})
    x=sanitize_numeric_chunk(df,['x'])
    assert x['x'].isna().sum()==2
