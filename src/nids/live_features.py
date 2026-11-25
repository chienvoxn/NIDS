"""Bounded live bidirectional feature candidates; NOT CICFlowMeter-certified."""
import math

class LiveFeatures:
    def __init__(self, flow_factory=None, idle=120., max_age=120., max_flows=2048,
                 max_packets=100000, max_per_flow=4096, fin_grace=1.0):
        if flow_factory is None:
            from nids.cic_flow_extractor import Flow
            flow_factory = Flow
        if min(idle,max_age,max_flows,max_packets,max_per_flow)<=0:
            raise ValueError('limits must be positive')
        self.factory=flow_factory
        self.idle,self.max_age=idle,max_age
        self.max_flows,self.max_packets,self.max_per_flow=max_flows,max_packets,max_per_flow
        self.active={}; self.buffered=0; self.clock=-math.inf
        self.dropped=0; self.invalidated=0
        if not math.isfinite(fin_grace) or fin_grace <= 0:
            raise ValueError('fin_grace must be positive')
        self.fin_grace = fin_grace
        self.fin_seen = {}
        self.fin_complete = {}
    def _close(self,key,reason,valid=True):
        f=self.active.pop(key); self.buffered-=len(f.all_packets)
        self.fin_seen.pop(key, None)
        self.fin_complete.pop(key, None)
        if not valid:
            self.invalidated+=1
            return None
        p=f.first
        return dict(source_ip=p.src,destination_ip=p.dst,source_port=p.sport,
                    destination_port=p.dport,protocol=p.proto,start_time=p.t,end_time=f.last_t,
                    close_reason=reason,feature_status='provisional_not_cnn_ready',**f.features())
    def advance(self,t):
        if not math.isfinite(t) or t<self.clock: raise ValueError('clock must be nondecreasing')
        self.clock=t; rows=[]
        for k,f in list(self.active.items()):
            if k in self.fin_complete and t-f.last_t>=self.fin_grace:
                rows.append(self._close(k,'tcp_fin_grace'))
            elif t-f.last_t>=self.idle or t-f.first.t>=self.max_age:
                rows.append(self._close(k,'idle' if t-f.last_t>=self.idle else 'max_age'))
        return rows
    def add(self,p):
        if not math.isfinite(p.t) or p.t<self.clock:
            self.dropped+=1; return []
        rows=self.advance(p.t)
        key=(p.proto,*sorted(((p.src,p.sport),(p.dst,p.dport))))
        f=self.active.get(key)
        if f is not None and (len(f.all_packets)>=self.max_per_flow or self.buffered>=self.max_packets):
            self._close(key,'capacity',False); self.dropped+=1; return rows
        if f is None:
            if len(self.active)>=self.max_flows or self.buffered>=self.max_packets:
                self.dropped+=1; return rows
            f=self.factory(p,p.t); self.active[key]=f
        f.add(p); self.buffered+=1
        if p.proto==6 and p.flags & 4:
            rows.append(self._close(key,'tcp_rst'))
        elif p.proto==6 and p.flags & 1:
            directions = self.fin_seen.setdefault(key, set())
            directions.add((p.src,p.sport,p.dst,p.dport))
            if len(directions)==2:
                self.fin_complete[key] = p.t
        return rows
    def flush(self):
        return [self._close(k,'capture_end_partial') for k in list(self.active)]


def ordered_row(row,features):
    if len(features)!=60 or len(set(features))!=60:
        raise ValueError('expected 60 distinct CIC features')
    missing=set(features)-row.keys()
    if missing: raise ValueError('missing formula keys: '+', '.join(sorted(missing)))
    return {k:row[k] for k in features}
