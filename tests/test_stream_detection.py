import pandas as pd
import pytest
from nids.stream_detection import process_batch, PROVISIONAL
class Engine:
    metadata={'seed':42,'dataset':'cic'}
    def predict(self,frame,batch_size):
        return pd.DataFrame({'predicted':['Bot']*len(frame),'confidence':[.9]*len(frame),'is_attack':[True]*len(frame)})

def test_replay():
    r=process_batch(Engine(),pd.DataFrame({'x':[1,2]}),offset=32)
    assert list(r.flow_row)==[33,34] and r.alert_status.eq('replay_alert').all()
    assert 'source_ip' not in r and r.batch_processing_ms.ge(0).all()

def test_gate():
    df=pd.DataFrame({'feature_status':[PROVISIONAL]})
    with pytest.raises(ValueError): process_batch(Engine(),df)
    with pytest.raises(ValueError): process_batch(Engine(),df,'live')

def test_shadow():
    df=pd.DataFrame({'feature_status':[PROVISIONAL],'close_reason':['idle']})
    assert process_batch(Engine(),df,'shadow').alert_status.eq('shadow_only_unverified').all()
    with pytest.raises(ValueError): process_batch(Engine(),pd.DataFrame({'x':[1]}),'shadow')

def test_bound():
    with pytest.raises(ValueError): process_batch(Engine(),pd.DataFrame({'x':[1,2]}),batch_size=1)
