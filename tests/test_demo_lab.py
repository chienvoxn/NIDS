import pandas as pd
from nids.demo_lab import scenario,evaluate_predictions,replay,topology,topology_examples

def test_scenarios():
    f=pd.DataFrame({'target':['BENIGN','PortScan','DoS Hulk','Bot']})
    assert len(scenario(f,'Mixed'))==4
    assert scenario(f,'PortScan').target.tolist()==['PortScan']
def test_metrics():
    f=pd.DataFrame({'target':['BENIGN','Bot','Bot'],'predicted':['Bot','BENIGN','Bot']})
    m=evaluate_predictions(f,['BENIGN','Bot'],'BENIGN'); assert m['fp']==1 and m['fn']==1 and m['fpr']==1 and m['detection_rate']==.5
    assert evaluate_predictions(f.iloc[1:],['BENIGN','Bot'],'BENIGN')['fpr'] is None

def test_topology():
    e=topology_examples(); assert topology(e['Centralized'])[0]=='centralized'; assert topology(e['P2P'])[0]=='p2p_like'; assert topology(e['Benign hub (same shape)'])[0]=='centralized'

def test_replay():
    class Fake:
        def predict(self,f): return pd.DataFrame({'predicted':['BENIGN']*len(f)})
    r=pd.concat(list(replay(Fake(),pd.DataFrame({'x':[1,2,3]}),2)))
    assert r.flow_row.tolist()==[1,2,3]
