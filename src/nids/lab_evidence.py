"""Evidence for labeled flow replay and topology; never claims live packet detection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from nids.demo_lab import replay, evaluate_predictions, topology, topology_examples


def mixed_case(frame, attack, limit=100, seed=42):
    if attack not in ('PortScan','DoS Hulk','Bot') or limit<1:
        raise ValueError('Unknown attack or invalid limit')
    parts=[]
    for label in ('BENIGN',attack):
        rows=frame[frame['target'].astype(str).str.strip()==label]
        if rows.empty: raise ValueError(f'Missing class {label}')
        parts.append(rows.sample(min(len(rows),limit),random_state=seed))
    return pd.concat(parts).sample(frac=1,random_state=seed).reset_index(drop=True)


def run_cases(engine, frame, destination, limit=100, batch=32):
    if engine.dataset!='cic': raise ValueError('CIC model required')
    dest=Path(destination); dest.mkdir(parents=True,exist_ok=False)
    rows=[]
    for attack in ('PortScan','DoS Hulk','Bot'):
        sample=mixed_case(frame,attack,limit)
        result=pd.concat(list(replay(engine,sample,batch)),ignore_index=True)
        metrics=evaluate_predictions(result,engine.labels,engine.normal)
        folder=dest/attack.replace(' ','_'); folder.mkdir()
        result.to_csv(folder/'predictions.csv',index=False)
        metrics.pop('matrix').to_csv(folder/'confusion_matrix.csv')
        metrics.pop('per_class').to_csv(folder/'per_class.csv',index=False)
        # Batch processing duration is repeated for each member; sum once per batch.
        times=result.iloc[::batch].batch_processing_ms.to_numpy()
        metrics.update(scenario=attack,flows=len(result),batch_size=batch,
                       processing_ms_per_flow=float(times.sum()/len(result)),
                       batch_p95_ms=float(np.percentile(times,95)))
        (folder/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        rows.append(metrics)
    pd.DataFrame(rows).to_csv(dest/'summary.csv',index=False)
    (dest/'provenance.json').write_text(json.dumps({'model':engine.metadata,'mode':'labeled CSV replay, not live attack','selection_seed':42,'limit_per_class':limit},indent=2),encoding='utf-8')
    return pd.DataFrame(rows)


def topology_evidence():
    rows=[]
    for case,edges in topology_examples().items():
        shape,g=topology(edges)
        expected='p2p_like' if case=='P2P' else 'centralized'
        rows.append({'case':case,'expected_shape':expected,'predicted_shape':shape,'match':shape==expected,'nodes':g.number_of_nodes(),'edges':g.number_of_edges(),'maliciousness':'not inferred'})
    return pd.DataFrame(rows)
