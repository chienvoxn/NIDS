"""CTU flow schema adapter, window graph, beaconing and topology lab checks."""
import numpy as np
import pandas as pd
import networkx as nx

ALIASES = {
 'src': ('SrcAddr','srcaddr','src_ip','src','source_ip'), 'dst': ('DstAddr','dstaddr','dst_ip','dst','destination_ip'),
 'time': ('StartTime','starttime','timestamp','time','start_time'), 'sport': ('Sport','sport','src_port','source_port'),
 'dport': ('Dport','dport','dst_port','destination_port'), 'bytes': ('TotBytes','totbytes','bytes'),
 'duration': ('Dur','dur','duration'), 'packets': ('TotPkts','totpkts','packets'),
 'label': ('Label','label','labels')
}
def resolve_columns(df):
    lookup = {c.lower():c for c in df.columns}
    return {key: next((lookup[c.lower()] for c in options if c.lower() in lookup),None)
            for key,options in ALIASES.items()}

def binary_labels(series):
    s = series.astype(str).str.lower()
    pos = s.str.contains('botnet|from-botnet|to-botnet|c&c', regex=True)
    normal = s.str.contains('normal') & ~pos
    # Background is unknown, never auto-negative.
    y = pd.Series(pd.NA, index=s.index, dtype='Int64')
    y.loc[normal] = 0
    y.loc[pos] = 1
    return y

def beaconing_features(df, min_events=5):
    c = resolve_columns(df)
    if any(c[k] is None for k in ('src','dst','time')):
        raise ValueError(f'CTU dataset lacks src/dst/time columns; resolved={c}')
    tmp = df[[c['src'],c['dst'],c['time']]].copy()
    tmp.columns = ['src','dst','time']
    if pd.api.types.is_numeric_dtype(tmp['time']):
        tmp['time'] = pd.to_datetime(
            tmp['time'], unit='s', errors='coerce', utc=True
        )
    else:
        tmp['time'] = pd.to_datetime(
            tmp['time'], errors='coerce', utc=True
        )
    tmp = tmp.dropna().sort_values('time')
    out=[]
    for (src,dst),group in tmp.groupby(['src','dst']):
        times = group['time']
        ts = (
            times - times.iloc[0]
        ).dt.total_seconds().to_numpy()
        gaps = np.diff(ts)
        if len(gaps)+1 < min_events or len(gaps)==0: continue
        mu=float(gaps.mean())
        out.append({'src':str(src),'dst':str(dst),'connections':len(ts),
                    'iat_mean_s':mu,'iat_std_s':float(gaps.std()),
                    'iat_cv':float(gaps.std()/mu) if mu>0 else None})
    return pd.DataFrame(
        out,
        columns=[
            'src', 'dst', 'connections',
            'iat_mean_s', 'iat_std_s', 'iat_cv'
        ]
    )

def communication_graph(df):
    c=resolve_columns(df)
    if not c['src'] or not c['dst']: raise ValueError(f'Missing graph endpoints: {c}')
    g=nx.DiGraph()
    for row in df[[c['src'],c['dst']]].dropna().itertuples(index=False,name=None):
        src,dst=map(str,row)
        if g.has_edge(src,dst):g[src][dst]['count']+=1
        else:g.add_edge(src,dst,count=1)
    return g

def graph_features(g):
    if not g.number_of_nodes():return pd.DataFrame(columns=['node','in_degree','out_degree','unique_peers','clustering'])
    u=g.to_undirected()
    clustering=nx.clustering(u)
    return pd.DataFrame([{'node':n,'in_degree':g.in_degree(n),'out_degree':g.out_degree(n),
                          'unique_peers':u.degree(n),'clustering':clustering[n]} for n in g.nodes])

def topology_lab_cases(n=12):
    """Controlled graph examples; NOT CTU-13 ground-truth topology labels."""
    return {'centralized_star': nx.star_graph(n), 'p2p_cycle': nx.cycle_graph(n+1),
            'p2p_mesh': nx.watts_strogatz_graph(n+1,4,0.25,seed=42),
            'benign_hub': nx.star_graph(n)}
