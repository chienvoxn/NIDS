# Phase 5A - Linux live metadata monitor

## Pham vi

Bat IPv4 TCP/UDP tren interface lab duoc cap phep, tong hop flow mot chieu va host theo cua so co dinh (tumbling window), ghi CSV/JSONL. Khong luu PCAP hay payload, khong dung GeneratedLabelledFlows. Khong thay doi dashboard, CNN hay checkpoint. Phase 4.1 co the kiem thu sau.

## Linux

Trong thu muc NIDS tren may ao Linux (mang lab co lap), tao venv neu chua co, cai scapy:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install scapy pytest
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_live_monitor.py
ip -br link
```

Chon interface lab cua minh, thay `eth0` bang ten thuc te. Quyen capture can duoc quan tri vien cap; vi du chay script thu thap rieng bang sudo:

```bash
sudo env PYTHONPATH="$PWD/src" "$PWD/.venv/bin/python" scripts/run_live_monitor.py --iface eth0 --seconds 60 --window 5 --max-packets 10000
```

Chi chay tren mang lab co quyen. Tao luu luong binh thuong giua cac may ao cua minh, vi du truy cap dich vu HTTP lab. Khong can tan cong mang that. Kiem tra log terminal va `outputs/live_monitor/<run>/`.

- flows.csv: 5-tuple mot chieu, timestamp packet dau/cuoi trong cua so, so packet, IP bytes.
- hosts.csv: packet/byte vao ra, so peer duy nhat trong cua so.
- windows.jsonl: moc cua so, so packet bo qua do gioi han/den muon, trang thai metadata_only, ai_verdict=null.

Cac cua so khong co packet khong duoc ghi. Cua so cuoi co the chua du thoi luong; window_end la bien cua so danh nghia, khong phai thoi diem canh bao. Script gioi han 300 giay, 100000 packet, 10000 flow/cua so. Bo qua IPv6, fragment va giao thuc khac. Capture duoc goi theo dot ngan, co the mat packet giua cac dot; chua do packet loss cua kernel. Khong phai capture production. File CSV chua IP lab, khong push log nhay cam len GitHub.

## Nghiem thu

- Unit tests: 3 test logic khong can Scapy.
- Linux capture: chua chay thuc te trong moi truong tao goi.
- Kiem tra co flow/host tu luu luong lab, timestamp va tong packet/byte hop ly.
- Co the dung src/dst/time cua flows.csv de xem do thi giao tiep; moi dong la nhom packet trong cua so, KHONG phai su kien mo ket noi doc lap. Khong dung de ket luan chu ky beacon cua ket noi that.

## Gate CNN (chua hoan thanh)

Metadata khong du 60 dac trung CIC; khong them cot gia, khong impute cot thieu de ep CNN chay. Can trich xuat bidirectional flow theo dinh nghia training, doi chieu don vi/timeout/packet length/IAT/flags/active-idle voi extractor tham chieu, kiem thu train-serving parity va sau do moi ket noi DemoEngine. Neu khong the chung minh parity, bao cao live metadata va CSV inference la hai tinh nang rieng. Chua co realtime CNN, chua co do tre canh bao end-to-end.
