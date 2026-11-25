import pandas as pd
from nids.lab_evidence import mixed_case,run_cases,topology_evidence
class Engine:
    dataset='cic'; normal='BENIGN'; labels=['BENIGN','PortScan','DoS Hulk','Bot']; metadata={'profile':'test-double'}
    def predict(self,frame):
        return pd.DataFrame({'predicted':frame.target.to_numpy(),'target':frame.target.to_numpy(),'is_attack':frame.target.to_numpy()!='BENIGN','confidence':1.})
def frame(): return pd.DataFrame({'target':Engine.labels*5})
def test_mixed():
    assert mixed_case(frame(),'Bot',3).target.value_counts().to_dict()=={'Bot':3,'BENIGN':3}
def test_evidence(tmp_path):
    d=tmp_path/'run'; r=run_cases(Engine(),frame(),d,3,2); assert len(r)==3 and (r.fpr==0).all() and (r.detection_rate==1).all(); assert (d/'Bot'/'predictions.csv').exists()
def test_topology():
    r=topology_evidence(); assert r.match.all() and (r.maliciousness=='not inferred').all()
