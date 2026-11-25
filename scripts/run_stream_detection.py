"""CNN CSV replay or experimental tail of Phase 5B CSV (no verified live alerts)."""
import argparse, csv, json, time, hashlib
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from nids.stream_detection import process_batch


def tail_batches(path, size, seconds, max_rows):
    """Phase 5B writes single-line numeric CSV records and flushes after emission."""
    deadline = time.monotonic()+seconds
    with path.open('r', encoding='utf-8', newline='') as f:
        header=None; rows=[]; seen=0; last_emit=time.monotonic()
        while time.monotonic()<deadline and seen<max_rows:
            pos=f.tell(); line=f.readline(262145)
            if len(line)>262144: raise ValueError('Oversize CSV record')
            if not line.endswith('\n'):
                f.seek(pos)
                if rows and time.monotonic()-last_emit>=0.5:
                    yield pd.DataFrame(rows,columns=header); rows=[]; last_emit=time.monotonic()
                time.sleep(0.1); continue
            row=next(csv.reader([line]))
            if header is None:
                header=row
                if len(header)!=len(set(header)): raise ValueError('Duplicate header')
                continue
            if len(row)!=len(header): raise ValueError('Incomplete or invalid CSV record')
            rows.append(row); seen+=1
            if len(rows)==size:
                yield pd.DataFrame(rows,columns=header); rows=[]; last_emit=time.monotonic()
        if rows: yield pd.DataFrame(rows,columns=header)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',required=True,type=Path)
    p.add_argument('--mode',choices=['replay','shadow'],default='replay')
    p.add_argument('--ack-unverified',action='store_true')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--seconds',type=int,default=60)
    p.add_argument('--max-rows',type=int,default=10000)
    p.add_argument('--delay',type=float,default=0.1)
    p.add_argument('--out',type=Path,default=Path('outputs/stream_detection'))
    a=p.parse_args()
    if not (1<=a.batch_size<=512 and 1<=a.seconds<=300 and 1<=a.max_rows<=100000 and 0<=a.delay<=2): p.error('Invalid limits')
    if a.mode=='shadow' and not a.ack_unverified: p.error('Shadow requires --ack-unverified; not verified live detection')
    from nids.demo_engine import DemoEngine
    engine=DemoEngine('outputs/final/cic_final_runs/seed_42','cic','full')
    if engine.metadata['seed']!=42: raise ValueError('Expected seed 42')
    out=a.out/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'); out.mkdir(parents=True,exist_ok=False)
    stats={'mode':a.mode,'verified_live':False,'rows':0,'attack_predictions':0,'replay_alerts':0,'complete':False,'model':engine.metadata,'source':str(a.csv),'checkpoint_sha256':hashlib.sha256((engine.directory/'cnn_se_mhsa.pt').read_bytes()).hexdigest()}
    try:
        batches=(pd.read_csv(a.csv,chunksize=a.batch_size,nrows=a.max_rows) if a.mode=='replay' else tail_batches(a.csv,a.batch_size,a.seconds,a.max_rows))
        with (out/'predictions.jsonl').open('w',encoding='utf-8') as pred, (out/'replay_alerts.jsonl').open('w',encoding='utf-8') as alerts:
            for frame in batches:
                if frame.empty: continue
                if a.mode=='shadow':
                    # Explicit missing numeric values from the Phase 5B writer.
                    for name in engine.features:
                        if name in frame:
                            frame[name]=frame[name].replace({'nan': float('nan'), '': float('nan')})
                result=process_batch(engine,frame,a.mode,a.batch_size,stats['rows'])
                result.to_csv(out/'predictions.csv',mode='a',header=stats['rows']==0,index=False)
                pred.write(result.to_json(orient='records',lines=True)+'\n'); pred.flush()
                selected=result[result['alert_status']=='replay_alert']
                if not selected.empty: alerts.write(selected.to_json(orient='records',lines=True)+'\n'); alerts.flush()
                stats['rows']+=len(result); stats['attack_predictions']+=int(result['is_attack'].sum()); stats['replay_alerts']+=len(selected)
                print('Processed:',stats['rows'],'mode:',a.mode,flush=True)
                if a.mode=='replay': time.sleep(a.delay)
        stats['complete']=True
    finally:
        (out/'summary.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
        print('Saved:',out,flush=True)

if __name__=='__main__': main()
