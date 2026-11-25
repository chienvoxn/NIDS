from nids.live_monitor import WindowMonitor


def test_window_and_hosts():
    m = WindowMonitor(5)
    assert m.add(10,'a','b',1,2,6,40) is None
    m.add(11,'a','b',1,2,6,60)
    m.add(12,'b','a',2,1,6,50)
    r = m.add(15,'a','c',1,2,6,70)
    assert len(r['flows']) == 2
    assert r['flows'][0]['ip_bytes'] == 100
    a = next(x for x in r['hosts'] if x['host']=='a')
    assert a['out_packets']==2 and a['in_packets']==1 and a['unique_peers']==1
    assert r['ai_verdict'] is None
    assert m.flush()['flows'][0]['dst']=='c'


def test_bounds_and_late():
    m=WindowMonitor(5,1)
    m.add(10,'a','b',1,2,6,40)
    m.add(11,'a','c',1,2,6,40)
    m.add(9,'a','b',1,2,6,40)
    r=m.advance(20)
    assert r['dropped_packets']==1 and r['late_packets']==1
    assert len(r['flows'])==1 and m.flush() is None


def test_idle_and_empty():
    m=WindowMonitor()
    assert m.flush() is None
    m.add(100,'a','b',1,2,17,28)
    assert m.advance(104) is None
    assert m.advance(105)['flows'][0]['protocol']==17
    assert m.advance(110) is None
