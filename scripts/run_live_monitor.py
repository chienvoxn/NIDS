"""Passive IPv4 TCP/UDP capture on an authorized Linux lab interface."""
import argparse
import csv
import json
import platform
import time
from pathlib import Path
from nids.live_monitor import WindowMonitor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--iface', required=True)
    p.add_argument('--seconds', type=int, default=60)
    p.add_argument('--window', type=int, default=5)
    p.add_argument('--max-packets', type=int, default=100000)
    p.add_argument('--out', default='outputs/live_monitor')
    a = p.parse_args()
    if not (1 <= a.seconds <= 300 and 1 <= a.window <= 60 and 1 <= a.max_packets <= 100000):
        p.error('Capture limits exceeded')
    if platform.system() != 'Linux':
        p.error('Run on Linux in an authorized lab')
    from scapy.all import sniff, IP, TCP, UDP
    out = Path(a.out) / str(time.time_ns())
    out.mkdir(parents=True, exist_ok=False)
    monitor = WindowMonitor(a.window)
    fields = ['window_start','src','dst','sport','dport','protocol','time','last_time','packets','ip_bytes']
    host_fields = ['window_start','host','in_packets','out_packets','in_bytes','out_bytes','unique_peers']
    with (out/'flows.csv').open('w', newline='') as f, (out/'hosts.csv').open('w', newline='') as h, (out/'windows.jsonl').open('w') as j:
        fw, hw = csv.DictWriter(f, fields), csv.DictWriter(h, host_fields)
        fw.writeheader(); hw.writeheader()
        def emit(result):
            if result is None: return
            for row in result['flows']: fw.writerow(dict(window_start=result['window_start'], **row))
            for row in result['hosts']: hw.writerow(dict(window_start=result['window_start'], **row))
            j.write(json.dumps({k:v for k,v in result.items() if k not in ('flows','hosts')})+'\n')
            f.flush(); h.flush(); j.flush()
            print(result['window_start'], 'flows', len(result['flows']), 'hosts', len(result['hosts']), 'metadata only', flush=True)
        def receive(pkt):
            if IP not in pkt: return
            ip = pkt[IP]
            if ip.frag or ip.flags.MF: return
            l4 = ip.payload
            if not isinstance(l4, (TCP, UDP)): return
            emit(monitor.add(float(pkt.time), ip.src, ip.dst, int(l4.sport), int(l4.dport), int(ip.proto), int(ip.len or len(ip))))
        deadline = time.monotonic()+a.seconds
        remaining = a.max_packets
        try:
            while remaining > 0 and time.monotonic() < deadline:
                # Count captured packets without storing them; flush on idle intervals too.
                count = [0]
                def callback(pkt):
                    count[0] += 1
                    receive(pkt)
                sniff(iface=a.iface, store=False, prn=callback, count=remaining,
                      timeout=min(1, max(0.01, deadline-time.monotonic())))
                remaining -= count[0]
                emit(monitor.advance(time.time()))
        finally:
            emit(monitor.flush())
    print('Saved metadata:', out, '\nNo PCAP, no payload, no CNN prediction.')

if __name__ == '__main__': main()
