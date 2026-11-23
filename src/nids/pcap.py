"""Minimal dependency-free classic PCAP Ethernet/IPv4 flow extractor.
Supports TCP and UDP packets; no pcapng, VLAN, IPv6, fragments or advanced CICFlowMeter features.
Use identical process_packets() on file and live capture adapters for parity.
"""
import csv
import ipaddress
import socket
import struct
from collections import defaultdict

FEATURES = ['source_ip','destination_ip','source_port','destination_port','protocol',
            'start_time','end_time','duration_s','total_packets','total_bytes','syn_count']

def iter_pcap(path):
    with open(path,'rb') as f:
        magic=f.read(4)
        variants={b'\xd4\xc3\xb2\xa1':('<',1e6),b'\xa1\xb2\xc3\xd4':('>',1e6),
                  b'\x4d\x3c\xb2\xa1':('<',1e9),b'\xa1\xb2\x3c\x4d':('>',1e9)}
        if magic not in variants:raise ValueError('Only classic PCAP supported, not pcapng')
        endian,scale=variants[magic]
        header=f.read(20)
        if len(header)!=20:raise ValueError('Truncated global header')
        *_,network=struct.unpack(endian+'HHIIII',header)
        if network!=1:raise ValueError('Only Ethernet linktype supported')
        while True:
            h=f.read(16)
            if not h:break
            if len(h)!=16:raise ValueError('Truncated packet header')
            sec,sub,included,original=struct.unpack(endian+'IIII',h)
            packet=f.read(included)
            if len(packet)!=included:raise ValueError('Truncated packet payload')
            yield sec+sub/scale,packet

def decode_ethernet_ipv4(t,packet):
    if len(packet)<34 or packet[12:14]!=b'\x08\x00':return None
    offset=14
    ihl=(packet[offset]&15)*4
    if ihl<20 or len(packet)<offset+ihl:return None
    proto=packet[offset+9]
    if proto not in (6,17):return None
    # Ignore non-initial IP fragments.
    flags_frag=int.from_bytes(packet[offset+6:offset+8],'big')
    if flags_frag & 0x1fff:return None
    src=str(ipaddress.ip_address(packet[offset+12:offset+16]));dst=str(ipaddress.ip_address(packet[offset+16:offset+20]))
    l4=offset+ihl
    if len(packet)<l4+4:return None
    sp,dp=struct.unpack('!HH',packet[l4:l4+4]); syn=0
    if proto==6 and len(packet)>=l4+14:syn=int(bool(packet[l4+13]&2))
    ip_len=int.from_bytes(packet[offset+2:offset+4],'big')
    return t,src,dst,sp,dp,proto,max(ip_len,0),syn

def process_packets(packets):
    state=defaultdict(lambda:{'start':None,'end':None,'packets':0,'bytes':0,'syn':0})
    for t,pkt in packets:
        decoded=decode_ethernet_ipv4(t,pkt)
        if decoded is None:continue
        t,src,dst,sp,dp,proto,size,syn=decoded
        # Bidirectional flow keyed by canonical endpoint ordering; direction follows first observed packet.
        a=(src,sp); b=(dst,dp)
        key=(proto,*sorted([a,b]))
        st=state[key]
        if st['start'] is None:st['start']=t;st['forward']=(src,dst,sp,dp)
        st['end']=t;st['packets']+=1;st['bytes']+=size;st['syn']+=syn
    result=[]
    for (proto,*_),st in state.items():
        src,dst,sp,dp=st['forward']
        result.append(dict(zip(FEATURES,[src,dst,sp,dp,proto,st['start'],st['end'],
                                        max(0.,st['end']-st['start']),st['packets'],st['bytes'],st['syn']])))
    return sorted(result,key=lambda r:r['start_time'])

def extract_pcap(path):return process_packets(iter_pcap(path))

def save_flows(rows,path):
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FEATURES);writer.writeheader();writer.writerows(rows)
