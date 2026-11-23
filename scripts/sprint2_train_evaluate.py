#!/usr/bin/env python3
"""CIC multiclass baselines / Gaussian and CTU binary baselines on *CSV-only* internal pilot.
Does NOT train the saved CNN checkpoint; never writes into models/final.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nids.sprint2_data import canonical_cic_label,ctu_label,ctu_numeric_features
from nids.sprint2_experiments import bounded_reservoir,run_experiment
import pandas as pd


def main():
 p=argparse.ArgumentParser()
 p.add_argument('--dataset',choices=('cic','ctu','both'),default='both')
 p.add_argument('--per-class',type=int,default=350,help='Maximum sampled rows per class (pilot scale)')
 p.add_argument('--chunksize',type=int,default=40000)
 p.add_argument('--seed',type=int,default=42)
 p.add_argument('--models',nargs='+',choices=('rf','xgb','lgbm'),default=['rf','xgb','lgbm'])
 p.add_argument('--out',default='reports/sprint2_part2_metrics.json')
 args=p.parse_args()
 if args.per_class<20 or args.chunksize<1:p.error('per-class >=20 and chunksize >=1 required')
 manifest=json.loads((ROOT/'models/final/deployment_manifest.json').read_text(encoding='utf-8'))
 output={'generated_utc':datetime.now(timezone.utc).isoformat(),'methodology':'internal_pilot_training_from_local_csv','parameters':vars(args),'datasets':{}}
 targets=['cic','ctu'] if args.dataset=='both' else [args.dataset]
 for kind in targets:
  folder=ROOT/'data/raw'/('CIC-IDS2017' if kind=='cic' else 'CTU-13')
  files=sorted(folder.rglob('*.csv'))
  if not files:raise FileNotFoundError(f'No CSV files under {folder}')
  if kind=='cic':
   if len(files)!=8:raise ValueError(f'Expected 8 CIC CSVs, found {len(files)}')
   features=manifest['feature_order']; label_fn=canonical_cic_label
  else:
   if len(files)!=2:raise ValueError(f'Expected 2 CTU CSVs, found {len(files)}')
   columns=pd.read_csv(files[0],nrows=0).columns.str.strip().tolist()
   features=ctu_numeric_features(columns)
   label_fn=ctu_label
  sample,distribution=bounded_reservoir(files,features,label_fn,args.per_class,args.chunksize,args.seed)
  result=run_experiment(sample,features,seed=args.seed,enabled=args.models)
  result['original_dataset_class_distribution']=distribution
  result['source_files']=[f.relative_to(ROOT).as_posix() for f in files]
  result['n_features']=len(features)
  result['feature_names']=features
  output['datasets'][kind]=result
  print(f'Completed {kind}: sample={len(sample)} original_rows={sum(distribution.values())}',flush=True)
 out=Path(args.out);out=out if out.is_absolute() else ROOT/out
 out.parent.mkdir(parents=True,exist_ok=True)
 out.write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 print('Saved:',out)
 for kind,results in output['datasets'].items():
  for name,item in results['metrics'].items():print(kind,name,'test_macro_f1=',round(item['test']['macro_f1'],4))
if __name__=='__main__':main()
