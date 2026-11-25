"""Passive Linux capture into provisional feature CSV; never invokes CNN."""
import argparse,csv,json,platform,time
from pathlib import Path
from nids.live_features import LiveFeatures,ordered_row

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--iface',required=True)
    p.add_argument('--seconds',type=int,default=60)
    p.add_argument('--max-packets',type=int,default=100000)
    p.add_argument('--metrics',default='outputs/final/cic_final_runs/seed_42/cnn_metrics.json')
    p.add_argument('--out',default='outputs/live_features')
    a=p.parse_args()
    if platform.system()!='Linux' or not 1<=a.seconds<=300 or not 1<=a.max_packets<=100000: p.error('Linux and bounded capture required')
    from scapy.all import sniff,IP
    from nids.cic_flow_extractor import parse_packet
    report=json.loads(Path(a.metrics).read_text(encoding='utf-8'))
    if report['dataset']!='cic': raise ValueError('CIC report required')
    features=report['features']; monitor=LiveFeatures()
    out=Path(a.out)/str(time.time_ns()); out.mkdir(parents=True,exist_ok=False)
    meta=['source_ip','destination_ip','source_port','destination_port','protocol','start_time','end_time','close_reason','feature_status']
    stats={'captured':0,'unsupported':0,'rows':0,'cnn_enabled':False}
    with (out/'provisional_features.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,meta+features); writer.writeheader()
        def emit(rows):
            for row in rows:
                writer.writerow({**{k:row[k] for k in meta},**ordered_row(row,features)})
                stats['rows']+=1
            f.flush()
        def receive(pkt):
            stats['captured']+=1
            # Adapt IPv4 bytes in RAM to existing Ethernet parser; no packet files.
            if IP not in pkt:
                stats['unsupported']+=1; return
            info=parse_packet(float(pkt.time),b'\x00'*12+b'\x08\x00'+bytes(pkt[IP]))
            if info is None:
                stats['unsupported']+=1; return
            emit(monitor.add(info))
        try:
            sniff(iface=a.iface,store=False,prn=receive,timeout=a.seconds,count=a.max_packets)
        finally:
            emit(monitor.flush())
            stats.update(dropped=monitor.dropped,invalidated=monitor.invalidated,feature_status='provisional_not_cnn_ready')
            (out/'summary.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print('Saved:',out,'; no CNN predictions')
if __name__=='__main__': main()
