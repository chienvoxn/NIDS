import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import pytest
from nids.offline import time_ordered_windows,causal_graph_features,ctu_features

def test_real_temporal_order():
 df=pd.DataFrame({'src':['a']*3,'time':['2020-01-01T00:00:03Z','2020-01-01T00:00:01Z','2020-01-01T00:00:02Z'],'feature':[3,1,2],'y':[1,0,0]})
 x,y,_=time_ordered_windows(df,['feature'],'src','time','y',length=2)
 assert x[:,:,0].tolist()==[[1,2],[2,3]] and y.tolist()==[0,1]

def test_missing_timestamp_rejected():
 df=pd.DataFrame({'src':['a','a'],'time':['not a time','2020-01-01'],'feature':[1,2],'y':[0,1]})
 with pytest.raises(ValueError,match='Unparseable'):time_ordered_windows(df,['feature'],'src','time','y',2)

def test_graph_causality_no_same_time_lookahead():
 m=pd.DataFrame({'src':['a','a','a'],'dst':['b','c','d'],'time':pd.to_datetime(['2020-01-01T00:00:00Z','2020-01-01T00:00:00Z','2020-01-01T00:00:01Z'],utc=True),'scenario_id':[5]*3})
 g=causal_graph_features(m)
 assert g.prior_out_flows.tolist()==[0,0,2]

def test_background_is_not_negative():
 df=pd.DataFrame({'SrcAddr':['a','a','a'],'DstAddr':['b','c','d'],'StartTime':['2020-01-01']*3,'Dur':[1,2,3],'Label':['Background','Normal','From-Botnet'],'scenario_id':[5]*3})
 X,y,meta=ctu_features(df)
 assert y.tolist()==[0,1] and len(meta)==2
