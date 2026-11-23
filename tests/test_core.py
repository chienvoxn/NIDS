import sys,struct
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from nids.models import SEATCNN,DiagonalGaussian
from nids.botnet import binary_labels,topology_lab_cases,communication_graph,graph_features
from nids.pcap import decode_ethernet_ipv4,process_packets
import pandas as pd

def test_cnn_ablation_shapes():
    for se,at in [(False,False),(True,False),(False,True),(True,True)]:
        m=SEATCNN(7,4,se,at)
        x=torch.randn(2,16,7);mask=torch.tensor([[True]*13+[False]*3,[True]*16])
        assert m(x,mask).shape==(2,4)

def test_gaussian():
    model=DiagonalGaussian().fit(np.array([[1.,2.],[1.,3.],[1.,4.]]))
    assert np.isfinite(model.score_samples([[1.,2.]])).all()

def test_ctu_background_unknown():
    y=binary_labels(pd.Series(['From-Botnet','From-Normal','Background','To-Botnet-C&C']))
    assert y.iloc[0]==1 and y.iloc[1]==0 and pd.isna(y.iloc[2]) and y.iloc[3]==1

def test_graph():
    df=pd.DataFrame({'SrcAddr':['a','a','b'],'DstAddr':['b','c','a']})
    g=communication_graph(df)
    assert g.number_of_nodes()==3 and g.number_of_edges()==3
    assert len(graph_features(g))==3
    assert len(topology_lab_cases())==4

def test_pcap_decode():
    eth=b'\x00'*12+b'\x08\x00'
    ipv4=bytes([0x45,0,0,40,0,0,0,0,64,6,0,0,192,168,1,1,192,168,1,2])
    tcp=struct.pack('!HHIIHHHH',1234,80,0,0,0x5002,4096,0,0)
    packet=eth+ipv4+tcp
    rows=process_packets([(1.,packet),(2.,packet)])
    assert len(rows)==1 and rows[0]['syn_count']==2 and rows[0]['duration_s']==1
