"""CSV scenario evaluation and controlled communication topology tests."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from time import perf_counter

SCENARIOS={'Normal':['BENIGN'], 'PortScan':['PortScan'], 'DoS':['DoS Hulk','DoS GoldenEye'], 'Bot':['Bot'], 'Mixed':None}

def scenario(frame, name, limit=100, seed=42):
    if name not in SCENARIOS or 'target' not in frame: raise ValueError('Expected labeled CIC data and known scenario')
    if limit<1: raise ValueError('Positive limit required')
    labels=SCENARIOS[name]
    if labels is None: labels=['BENIGN','PortScan','DoS Hulk','Bot']
    parts=[]
    for label in labels:
        group=frame[frame.target.astype(str)==label]
        if group.empty: raise ValueError(f'Missing scenario label: {label}')
        parts.append(group.sample(min(limit,len(group)),random_state=seed))
    return pd.concat(parts).sample(frac=1,random_state=seed).reset_index(drop=True)

def evaluate_predictions(result, labels, normal):
    key=next((k for k in ('target','Label') if k in result),None)
    if key is None: return None
    y=result[key].astype(str).str.strip(); p=result.predicted.astype(str)
    if not set(y).issubset(labels) or not set(p).issubset(labels): raise ValueError('Unknown labels')
    if len(y)==0: raise ValueError('Empty evaluation')
    a,b,c,s=precision_recall_fscore_support(y,p,labels=labels,zero_division=0)
    avg='binary' if normal=='0' else 'weighted'
    kw={'pos_label':'1'} if avg=='binary' else {}
    pr,re,f1,_=precision_recall_fscore_support(y,p,average=avg,zero_division=0,**kw)
    tn,fp,fn,tp=confusion_matrix(y!=normal,p!=normal,labels=[False,True]).ravel()
    return {'accuracy':float(accuracy_score(y,p)), 'precision':float(pr),'recall':float(re),'f1':float(f1),'average':avg,'fpr':float(fp/(fp+tn)) if fp+tn else None,'detection_rate':float(tp/(tp+fn)) if tp+fn else None,'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'per_class':pd.DataFrame({'label':labels,'precision':a,'recall':b,'f1':c,'support':s}),'matrix':pd.DataFrame(confusion_matrix(y,p,labels=labels),index=labels,columns=labels)}

def replay(engine, frame, batch_size=32):
    """Sequential CSV batches, not packet capture or original time replay."""
    if batch_size<1 or frame.empty: raise ValueError('Positive batch and nonempty input required')
    for i in range(0,len(frame),batch_size):
        start=perf_counter(); out=engine.predict(frame.iloc[i:i+batch_size]); elapsed=perf_counter()-start
        out['flow_row']=np.arange(i+1,i+1+len(out)); out['processed_at_utc']=pd.Timestamp.now(tz='UTC').isoformat(); out['batch_processing_ms']=elapsed*1000
        yield out

def topology(edges):
    """Shape heuristic only; a centralized hub is not proof of maliciousness."""
    import networkx as nx
    g=nx.Graph(); g.add_edges_from((str(a),str(b)) for a,b in edges if a!=b)
    n=g.number_of_nodes()
    if n<4 or not nx.is_connected(g): return 'indeterminate',g
    deg=np.array([d for _,d in g.degree()]); concentration=deg.max()/(n-1)
    if concentration>=0.8 and (deg==1).mean()>=0.6: label='centralized'
    elif deg.min()>=2 and deg.max()<=2*deg.min(): label='p2p_like'
    else: label='indeterminate'
    return label,g

def topology_examples():
    return {'Centralized': [('hub',f'peer{i}') for i in range(8)], 'P2P': [(f'peer{i}',f'peer{(i+1)%8}') for i in range(8)], 'Benign hub (same shape)': [('server',f'client{i}') for i in range(8)]}
