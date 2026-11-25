"""Continuous experimental CNN shadow inference on Linux."""
import argparse,csv,json,platform,time,threading
from pathlib import Path
import hashlib
from nids.demo_engine import DemoEngine
from nids.live_shadow import LiveShadowWorker
from nids.live_features import LiveFeatures,ordered_row

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--iface',required=True)
    p.add_argument('--seconds',type=int,default=60)
    p.add_argument('--max-packets',type=int,default=100000)
    p.add_argument('--metrics',default='outputs/final/cic_final_runs/seed_42/cnn_metrics.json')
    p.add_argument('--out',default='outputs/live_features')
    p.add_argument('--ack-unverified', action='store_true', required=True)
    a=p.parse_args()
    if platform.system()!='Linux' or not 1<=a.seconds<=300 or not 1<=a.max_packets<=100000: p.error('Linux and bounded capture required')
    from scapy.all import AsyncSniffer,IP
    from nids.cic_flow_extractor import parse_packet
    report=json.loads(Path(a.metrics).read_text(encoding='utf-8'))
    if report['dataset']!='cic': raise ValueError('CIC report required')
    features=report['features']; monitor=LiveFeatures()
    engine = DemoEngine(
        Path(a.metrics).parent,
        'cic',
        'full',
        variant='cnn_mhsa'
    )
    if engine.features != features:
        raise ValueError('Capture/model schema mismatch')
    lock=threading.Lock()
    ready=threading.Event()
    out=Path(a.out)/str(time.time_ns()); out.mkdir(parents=True,exist_ok=False)
    meta=['source_ip','destination_ip','source_port','destination_port','protocol','start_time','end_time','close_reason','feature_status']
    stats={'captured':0,'unsupported':0,'rows':0,'cnn_enabled':True,
        'mode':'shadow',
        'verified_live':False,
        'semantic_parity':False}
    worker = LiveShadowWorker(engine, out)
    with worker, (out/'provisional_features.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,meta+features); writer.writeheader()
        def emit(rows):
            for row in rows:
                writer.writerow({**{k:row[k] for k in meta},**ordered_row(row,features)})
                stats['rows']+=1
                worker.submit(row, stats['rows'])
            f.flush()
        def receive_unlocked(pkt):
            stats['captured']+=1
            # Adapt IPv4 bytes in RAM to existing Ethernet parser; no packet files.
            if IP not in pkt:
                stats['unsupported']+=1; return
            info=parse_packet(float(pkt.time),b'\x00'*12+b'\x08\x00'+bytes(pkt[IP]))
            if info is None:
                stats['unsupported']+=1; return
            rows = monitor.add(info)
            stats['packet_triggered_rows'] += len(rows)
            emit(rows)
        def receive(pkt):
            with lock:
                receive_unlocked(pkt)

        stats['packet_triggered_rows'] = 0
        stats['flush_rows'] = 0
        stats['timer_ticks'] = 0
        stats['periodic_rows'] = 0
        stats['final_expiration_rows'] = 0
        stats['capture_complete'] = False

        sniffer=AsyncSniffer(
            iface=a.iface,
            store=False,
            prn=receive,
            count=a.max_packets,
            started_callback=ready.set
        )

        try:
            sniffer.start()

            if not ready.wait(timeout=5):
                if not sniffer.running:
                    sniffer.join()
                raise RuntimeError('Capture startup timeout')

            deadline=time.monotonic()+a.seconds

            while sniffer.running:
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    break

                time.sleep(min(0.25,remaining))

                with lock:
                    cutoff=time.time()-0.5

                    stats['timer_ticks']+=1

                    if cutoff>=monitor.clock:
                        rows=monitor.advance(cutoff)
                        stats['periodic_rows']+=len(rows)
                        emit(rows)

            if sniffer.running:
                sniffer.stop()

            sniffer.join()

            with lock:
                rows=monitor.advance(
                    max(monitor.clock,time.time())
                )
                stats['final_expiration_rows']+=len(rows)
                emit(rows)

            stats['capture_complete']=True
        finally:
            if sniffer.running:
                sniffer.stop()

            sniffer.join()

            stats['capture_stopped_at_epoch'] = time.time()
            with lock:
                rows = monitor.flush()
                stats['flush_rows'] += len(rows)
                emit(rows)
            stats.update(dropped=monitor.dropped,invalidated=monitor.invalidated,feature_status='provisional_not_cnn_ready')
            stats['drop_reasons'] = dict(
                monitor.drop_reasons
            )
            stats['max_lateness_seconds'] = (
                monitor.max_lateness_seconds
            )
            stats.update(worker.finish())
            stats['model'] = engine.metadata
            stats['checkpoint_sha256'] = hashlib.sha256(
                engine.checkpoint_path.read_bytes()
            ).hexdigest()
            stats['pipeline_complete'] = (
                stats['capture_complete']
                and stats['shadow_complete']
                and stats['shadow_processed'] == stats['rows']
                and stats['dropped'] == 0
                and stats['invalidated'] == 0
            )
            (out/'summary.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print(json.dumps(stats, indent=2))
    print('Saved:', out, '; experimental shadow inference only')
    if not stats['pipeline_complete']:
        raise RuntimeError('Incomplete shadow pipeline; inspect summary')
if __name__=='__main__': main()
