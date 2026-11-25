import pandas as pd
import networkx as nx
from nids.soc_charts import matrix_chart,batch_summary,graph_chart

def test_matrix():
    m=pd.DataFrame([[98,2],[0,0]],index=['Normal','Attack'],columns=['Normal','Attack'])
    f=matrix_chart(m,True); assert f.data[0].z.tolist()==[[98.,2.],[0.,0.]]

def test_batches():
    r=pd.DataFrame({'flow_row':[1,2,3],'is_attack':[True,False,True],'batch_processing_ms':[5,5,3]})
    b=batch_summary(r,2); assert b.Flows.tolist()==[2,1]; assert b['Cumulative alerts'].tolist()==[1,2]; assert b['Processing ms'].tolist()==[5,3]

def test_graph():
    g=nx.Graph(); g.add_edge('a','b'); f=graph_chart(g,[('a','b')]*6); assert len(f.layout.annotations)==1; assert f.data[1].text==('6',)
