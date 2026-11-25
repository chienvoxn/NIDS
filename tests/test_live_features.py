from types import SimpleNamespace
import pytest
from nids.live_features import LiveFeatures,ordered_row

class FakeFlow:
    def __init__(self,p,t): self.first=p; self.last_t=t; self.all_packets=[]
    def add(self,p): self.all_packets.append(p); self.last_t=p.t
    def features(self): return {'count':len(self.all_packets)}

def packet(t,reverse=False):
    a,b=('a','b') if not reverse else ('b','a')
    x,y=(1,2) if not reverse else (2,1)
    return SimpleNamespace(t=t,src=a,dst=b,sport=x,dport=y,proto=6,flags=0)

def test_bidirectional_and_idle():
    m=LiveFeatures(FakeFlow,idle=5)
    assert not m.add(packet(0)); assert not m.add(packet(1,True))
    row=m.advance(6)[0]
    assert row['count']==2 and row['source_ip']=='a' and m.buffered==0
    assert row['feature_status']=='provisional_not_cnn_ready'

def test_limits_and_order():
    m=LiveFeatures(FakeFlow,max_per_flow=1)
    m.add(packet(1)); assert not m.add(packet(2))
    assert m.invalidated==1 and m.buffered==0
    assert not m.add(packet(0)) and m.dropped==2

def test_partial_and_schema():
    m=LiveFeatures(FakeFlow); m.add(packet(0))
    assert m.flush()[0]['close_reason']=='capture_end_partial'
    names=[str(i) for i in range(60)]
    assert list(ordered_row(dict.fromkeys(names,0),names))==names
    with pytest.raises(ValueError): ordered_row({},names)

def test_real_extractor_if_available():
    mod=pytest.importorskip('nids.cic_flow_extractor')
    m=LiveFeatures()
    m.add(mod.PacketInfo(0,'10.0.0.1','10.0.0.2',1234,80,6,0,40,20,2,1024))
    m.add(mod.PacketInfo(1,'10.0.0.2','10.0.0.1',80,1234,6,10,50,20,16,1024))
    r=m.flush()[0]
    assert r['Total Fwd Packets']==1 and r['Total Backward Packets']==1
    assert r['Flow Duration']==1000000 and r['Total Length of Bwd Packets']==10
    assert len(set(r)-{'source_ip','destination_ip','source_port','destination_port','protocol','start_time','end_time','close_reason','feature_status'})==60


def test_tcp_fin_grace_lifecycle():
    m = LiveFeatures(FakeFlow, fin_grace=1.0)

    m.add(packet(0))

    p = packet(1)
    p.flags = 17
    assert not m.add(p)

    p = packet(2, True)
    p.flags = 17
    assert not m.add(p)

    p = packet(2.1)
    p.flags = 16
    assert not m.add(p)

    assert not m.advance(2.5)

    rows = m.advance(3.2)

    assert len(rows) == 1
    assert rows[0]['count'] == 4
    assert rows[0]['close_reason'] == 'tcp_fin_grace'
    assert m.buffered == 0
    assert not m.active
    assert not m.fin_seen
    assert not m.fin_complete


def test_single_fin_keeps_flow_open():
    m = LiveFeatures(FakeFlow)

    p = packet(0)
    p.flags = 1
    m.add(p)

    assert not m.advance(2)
    assert len(m.active) == 1

    assert m.flush()[0]['close_reason'] == 'capture_end_partial'


def test_rst_cleans_fin_tracking():
    m = LiveFeatures(FakeFlow)

    p = packet(0)
    p.flags = 1
    m.add(p)

    p = packet(1, True)
    p.flags = 4

    rows = m.add(p)

    assert rows[0]['close_reason'] == 'tcp_rst'
    assert not m.fin_seen
    assert not m.fin_complete
    assert m.buffered == 0
