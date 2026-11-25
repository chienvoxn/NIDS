"""Presentation helpers; no changes to inference or topology classification."""
from collections import Counter
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go


def style(fig, height=320):
    return fig.update_layout(template='plotly_dark', paper_bgcolor='#162338', plot_bgcolor='#162338', height=height, margin=dict(l=20,r=20,t=25,b=35), font=dict(color='#cbd5e1'), legend=dict(orientation='h',y=1.12))


def matrix_chart(mat, percent=False):
    a=mat.to_numpy(); totals=a.sum(axis=1,keepdims=True)
    pct=np.divide(a*100,totals,out=np.zeros_like(a,dtype=float),where=totals!=0)
    text=[[f'{v:.1f}%' if percent else str(int(v)) for v in row] for row in (pct if percent else a)]
    f=go.Figure(go.Heatmap(z=pct if percent else a,x=list(mat.columns),y=list(mat.index),text=text,texttemplate='%{text}',customdata=np.stack([a,pct],axis=-1),hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{customdata[0]}<br>Row share: %{customdata[1]:.1f}%<extra></extra>',colorscale=[[0,'#1e293b'],[1,'#087ca7']],colorbar=dict(title='%' if percent else 'Count')))
    f.update_yaxes(autorange='reversed',title='Actual'); f.update_xaxes(title='Predicted',tickangle=-40)
    return style(f,min(680,max(280,32*len(mat)+140)))


def batch_summary(result,batch):
    x=result.copy(); x['Batch index']=(x.flow_row-1)//batch+1
    g=x.groupby('Batch index'); out=g.is_attack.agg(['size','sum']).rename(columns={'size':'Flows','sum':'Attack predictions'})
    out['Benign predictions']=out.Flows-out['Attack predictions']; out['Cumulative alerts']=out['Attack predictions'].cumsum(); out['Processing ms']=g.batch_processing_ms.first()
    return out.reset_index()


def graph_chart(g,edges):
    counts=Counter((str(a),str(b)) for a,b in edges if a!=b)
    pos=nx.spring_layout(g,seed=42) if len(g)>2 else {n:(i*2-1,0) for i,n in enumerate(g)}
    f=go.Figure(); annotations=[]
    for (a,b),count in counts.items():
        x,y=pos[a]; u,v=pos[b]
        f.add_trace(go.Scatter(x=[x,u],y=[y,v],mode='lines',line=dict(color='#64748b',width=min(5,1+np.log1p(count))),hoverinfo='skip',showlegend=False))
        annotations.append(dict(x=u,y=v,ax=x,ay=y,xref='x',yref='y',axref='x',ayref='y',showarrow=True,arrowhead=3,arrowsize=1,arrowwidth=1,arrowcolor='#38bdf8',standoff=14,startstandoff=14,text=''))
        f.add_trace(go.Scatter(x=[(x+u)/2],y=[(y+v)/2],mode='markers+text',marker=dict(size=10,color='#162338'),text=[str(count)],textposition='top center',hovertext=[f'{a} → {b}: {count} events'],hoverinfo='text',showlegend=False))
    f.add_trace(go.Scatter(x=[pos[n][0] for n in g],y=[pos[n][1] for n in g],mode='markers+text',text=list(g),textposition='bottom center',marker=dict(size=24,color='#38bdf8'),hovertext=[f'{n}: degree {g.degree(n)}' for n in g],hoverinfo='text',showlegend=False))
    f.update_layout(annotations=annotations); f.update_xaxes(visible=False,range=[-1.5,1.5]); f.update_yaxes(visible=False,range=[-1.45,1.45]); return style(f,330)
