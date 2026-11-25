"""Bounded passive capture of metadata only; not CNN-compatible feature extraction."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--iface',required=True); p.add_argument('--seconds',type=int,default=30); p.add_argument('--max-packets',type=int,default=10000); p.add_argument('--out',default='outputs/live_metadata.csv'); a=p.parse_args()
    if not 1<=a.seconds<=300 or not 1<=a.max_packets<=100000: p.error('seconds 1-300, max-packets 1-100000')
    from scapy.all import sniff, IP, TCP, UDP
    flows=defaultdict(lambda: [0,0]); rows=[]
    def receive(pkt):
        if IP not in pkt or not (TCP in pkt or UDP in pkt): return
        ip=pkt[IP]; l4=pkt[TCP] if TCP in pkt else pkt[UDP]; t=float(pkt.time); key=(ip.src,ip.dst,int(l4.sport),int(l4.dport),int(ip.proto),int(t//5)*5)
        flows[key][0]+=1; flows[key][1]+=int(ip.len or len(ip))
    sniff(iface=a.iface,timeout=a.seconds,count=a.max_packets,store=False,prn=receive)
    for (src,dst,sp,dp,proto,t),(n,b) in sorted(flows.items(),key=lambda x:x[0][-1]):
        rows.append(dict(src=src,dst=dst,sport=sp,dport=dp,protocol=proto,time=datetime.fromtimestamp(t,timezone.utc).isoformat(),packets=n,ip_bytes=b))
    dest=Path(a.out); dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['src','dst','sport','dport','protocol','time','packets','ip_bytes']); w.writeheader(); w.writerows(rows)
    print(f'Saved {len(rows)} directional 5-second metadata groups to {dest}; no PCAP saved; no AI verdict.')
if __name__=='__main__': main()
