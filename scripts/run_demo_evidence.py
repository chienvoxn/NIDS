import argparse,json
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
from nids.demo_engine import DemoEngine
from nids.lab_evidence import run_cases,topology_evidence

def main():
    p=argparse.ArgumentParser(); p.add_argument('--csv',default='data/demo/cic_cnn_demo.csv'); p.add_argument('--model',default='CIC CNN + MHSA seed 42'); p.add_argument('--out'); a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'configs/demo_models.json').read_text(encoding='utf-8'))[a.model]
    engine=DemoEngine(root/cfg['directory'],cfg['dataset'],cfg['profile'],variant=cfg['variant'])
    dest=Path(a.out) if a.out else root/'outputs/demo_evidence'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    frame=pd.read_csv(a.csv,nrows=20001)
    if len(frame)>20000: raise ValueError('Maximum 20,000 demo rows')
    print(run_cases(engine,frame,dest).to_string(index=False))
    topology_evidence().to_csv(dest/'topology_lab.csv',index=False)
    print('Saved:',dest)
if __name__=='__main__': main()
