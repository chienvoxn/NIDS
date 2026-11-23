# PROJECT SPECIFICATION
# Hệ thống phát hiện tấn công mạng dựa trên mô hình CNN mở rộng (SE-AT-CNN)

> **Tên tiếng Anh:** Network Intrusion Detection Using an Extended CNN with Squeeze-and-Excitation and Temporal Attention  
> **Tên đề xuất mô hình:** SE-AT-CNN  
> **Phiên bản tài liệu:** v2.0 — Phương án 2 datasets  
> **Ngày cập nhật:** 11/11/2026  
> **Trạng thái:** Đặc tả và kế hoạch triển khai; **chưa có kết quả thực nghiệm**  
> **Mục tiêu:** Nghiên cứu, xây dựng và triển khai Network Intrusion Detection System (NIDS) trên Linux.  
> **Datasets chính thức:** CIC-IDS2017 (phân loại tấn công đa lớp) + CTU-13 (phát hiện botnet/đồ thị).  
> **Mô hình nghiên cứu chính:** CNN + Squeeze-and-Excitation (SE) + Attention (AT).  
> **5 baseline chốt:** Random Forest, XGBoost, LightGBM, MLP, Vanilla 1D-CNN.

---

## Mục lục

1. [Tóm tắt quyết định kỹ thuật](#1-tóm-tắt-quyết-định-kỹ-thuật)
2. [Bối cảnh, mục tiêu và phạm vi](#2-bối-cảnh-mục-tiêu-và-phạm-vi)
3. [Đối chiếu yêu cầu của giảng viên](#3-đối-chiếu-yêu-cầu-của-giảng-viên)
4. [Bộ dữ liệu: CIC-IDS2017 và CTU-13](#4-bộ-dữ-liệu-cic-ids2017-và-ctu-13)
5. [Bài toán và câu hỏi nghiên cứu](#5-bài-toán-và-câu-hỏi-nghiên-cứu)
6. [Thiết kế dữ liệu và phòng tránh data leakage](#6-thiết-kế-dữ-liệu-và-phòng-tránh-data-leakage)
7. [EDA và tiền xử lý](#7-eda-và-tiền-xử-lý)
8. [Năm baseline và tiêu chí so sánh](#8-năm-baseline-và-tiêu-chí-so-sánh)
9. [Mô hình chính: SE-AT-CNN](#9-mô-hình-chính-se-at-cnn)
10. [Gaussian Anomaly Detection](#10-gaussian-anomaly-detection)
11. [Botnet Detection: temporal và graph](#11-botnet-detection-temporal-và-graph)
12. [Kiến trúc tổng thể](#12-kiến-trúc-tổng-thể)
13. [Hệ thống giám sát realtime trên Linux](#13-hệ-thống-giám-sát-realtime-trên-linux)
14. [Thiết kế thí nghiệm và đánh giá](#14-thiết-kế-thí-nghiệm-và-đánh-giá)
15. [Kiểm thử với mạng lab](#15-kiểm-thử-với-mạng-lab)
16. [Cấu trúc repository](#16-cấu-trúc-repository)
17. [Công nghệ, hạ tầng và quản lý artifacts](#17-công-nghệ-hạ-tầng-và-quản-lý-artifacts)
18. [Lộ trình triển khai](#18-lộ-trình-triển-khai)
19. [Checklist và tiêu chí nghiệm thu](#19-checklist-và-tiêu-chí-nghiệm-thu)
20. [Rủi ro, giới hạn và phương án dự phòng](#20-rủi-ro-giới-hạn-và-phương-án-dự-phòng)
21. [Kế hoạch báo cáo và demo](#21-kế-hoạch-báo-cáo-và-demo)
22. [Hướng dẫn khởi động dự án](#22-hướng-dẫn-khởi-động-dự-án)
23. [Tài liệu tham khảo](#23-tài-liệu-tham-khảo)
24. [Quyết định đã chốt và việc cần xác nhận](#24-quyết-định-đã-chốt-và-việc-cần-xác-nhận)

---

## 1. Tóm tắt quyết định kỹ thuật

| Hạng mục | Quyết định |
|---|---|
| Số dataset nghiên cứu | **2: CIC-IDS2017 + CTU-13** |
| Dataset cho Attack Classification | **CIC-IDS2017** |
| Dataset cho Botnet Detection | **CTU-13, khai thác cả 13 scenario trong đánh giá cuối** |
| Bài toán ML/DL chính | Phân loại traffic đa lớp; phát hiện botnet nhị phân |
| Baselines | **Random Forest, XGBoost, LightGBM, MLP, Vanilla 1D-CNN** |
| Mô hình đề xuất | **SE-AT-CNN = Conv1D + Squeeze-and-Excitation + Temporal Attention** |
| Ablation | CNN; CNN+SE; CNN+AT; CNN+SE+AT |
| Anomaly Detection | Gaussian log-likelihood; có thể đối chứng thêm Isolation Forest |
| Botnet/Graph | NetworkX + đặc trưng đồ thị + mô hình ML/luật thống kê |
| Beaconing | Phân tích chu kỳ kết nối, IAT, độ lệch chuẩn, coefficient of variation |
| Realtime | Python + Scapy + flow/window aggregator + model inference trên Linux |
| Đích demo | PortScan; DoS/SYN traffic mô phỏng có kiểm soát; beaconing giả lập |
| Dashboard | Nên có nếu thời gian cho phép; không phải trọng tâm |
| Auto block IP | Tùy chọn, chỉ dùng trong lab có cơ chế hoàn tác |
| Chỉ số chính | Macro-F1, per-class Recall/F1, PR-AUC, FPR, detection latency |
| Nguyên tắc | Không gộp trực tiếp hai dataset; không tuyên bố SE-AT-CNN chắc chắn tốt hơn baseline |

**Kết luận phạm vi:** CIC-IDS2017 đáp ứng yêu cầu phân loại nhiều kiểu tấn công; CTU-13 bổ sung phần botnet, beaconing và đồ thị. Cả hai pipeline được tích hợp vào một ứng dụng NIDS nhưng có thể có **model và feature schema riêng**.

---

## 2. Bối cảnh, mục tiêu và phạm vi

### 2.1. Vấn đề

Network traffic được thu thập ở dạng packet/PCAP hoặc flow logs; dữ liệu chưa được chuẩn hóa thì các thuật toán ML không thể trực tiếp vận hành nhất quán. Một NIDS có giá trị phải đồng thời:

1. Biến traffic thành dữ liệu có cấu trúc.
2. Nhận diện bất thường, phân loại các cuộc tấn công đã biết.
3. Phát hiện hành vi botnet kéo dài theo thời gian.
4. Chạy được trên Linux với traffic thực tế hoặc replay PCAP.
5. Được đánh giá trung thực: dữ liệu test độc lập, ít báo động giả, độ trễ phù hợp.

### 2.2. Mục tiêu bắt buộc

- Xây dựng pipeline PCAP/log → packet parsing → flow/host/window features → bảng dữ liệu.
- EDA, thống kê nhãn, missing/Inf/duplicate, class imbalance, feature distributions.
- Triển khai và đánh giá **Gaussian Anomaly Detection**.
- Huấn luyện và đánh giá **5 baseline**.
- Xây dựng mô hình chính **SE-AT-CNN**, có ablation chứng minh tác động của SE và AT.
- Đạt bài toán **Attack Classification đa lớp** trên CIC-IDS2017.
- Xây dựng **botnet detection** dựa trên temporal/beaconing/graph features với CTU-13.
- Khảo sát **centralized/P2P topology**, có dữ liệu thử nghiệm với ground truth.
- Triển khai inference realtime trên Linux, xuất cảnh báo và log.
- Test ít nhất **2–3 kịch bản mô phỏng** trong mạng được phép kiểm thử.
- Báo cáo metrics offline, realtime, độ trễ, nhầm lẫn và giới hạn.

### 2.3. Ngoài phạm vi ở phiên bản đầu

- Không bắt buộc nghiên cứu GNN hoặc Transformer lớn.
- Không bắt buộc gộp train CIC-IDS2017 và CTU-13 thành một mô hình.
- Không bắt buộc thêm CIC-DDoS2019 và InSDN.
- Không triển khai tấn công trên mạng công cộng hay hệ thống không sở hữu.
- Không dùng malware executable đi kèm CTU-13 để chạy.
- Không hứa hệ thống phát hiện mọi kiểu tấn công mới hay đạt một ngưỡng accuracy định sẵn.

### 2.4. Sản phẩm cuối

1. Bộ mã nguồn xử lý dữ liệu (PCAP, flow logs, chuẩn hóa).
2. Notebook/script EDA và baseline.
3. PyTorch SE-AT-CNN và checkpoints.
4. Module Gaussian anomaly.
5. Module botnet graph/beaconing.
6. Linux realtime detection engine.
7. Alert log và dashboard cơ bản (nếu có).
8. Test logs, biểu đồ, confusion matrix, báo cáo, video/ảnh demo.
9. README, tài liệu kiến trúc, tài liệu cài đặt và tái lập thí nghiệm.

---

## 3. Đối chiếu yêu cầu của giảng viên

| Yêu cầu | Cách đáp ứng | Minh chứng |
|---|---|---|
| Đọc PCAP, log và chuyển đổi dữ liệu | Parser + flow extractor, dùng CIC-IDS2017 PCAP; CTU-13 flow logs | Demo PCAP→CSV/Parquet; unit test |
| EDA về traffic bình thường và tấn công | Notebook EDA cho cả 2 dataset | Báo cáo và biểu đồ |
| Missing, scaling, imbalance | Fit xử lý chỉ trên train, áp dụng val/test | Pipeline versioned |
| Phát hiện bất thường Gaussian | Fit benign, chọn ngưỡng trên validation | PR/ROC, confusion matrix, FPR |
| Phân loại tấn công đa lớp | CIC-IDS2017, baseline + SE-AT-CNN | Per-class metrics, confusion matrix |
| So sánh nhiều ML | RF, XGBoost, LightGBM, MLP, 1D-CNN | Bảng metric/time/size |
| CNN mở rộng | SE, AT, ablation | CNN vs CNN+SE vs CNN+AT vs SE-AT-CNN |
| Beaconing | IAT statistics, sliding window | Biểu đồ chu kỳ và alert |
| Botnet graph | CTU-13 → NetworkX → graph features → ML | Graph visualization và classification |
| Centralized/P2P | Lab synthetic graph có topology đã biết + phân tích CTU-13 | Test có ground truth; nêu rõ giới hạn |
| So sánh ML phát hiện botnet | RF/XGBoost trên graph/flow, IF/Gaussian phù hợp | Bảng riêng trên CTU-13 |
| Linux realtime | Scapy sniff/replay, feature extraction, inference, alert | Demo/video, JSONL log |
| 2–3 kịch bản | PortScan, DoS/SYN giới hạn, beaconing lab | Timestamp, expected vs predicted |
| Đo latency/FPR | End-to-end timing, false alarms theo đơn vị traffic/time | Báo cáo định lượng |
| Code + tài liệu | Repository + scripts + docs | Checklist nghiệm thu |

**Điểm cần chú ý:** Đề bài ghi "3 bài toán" nhưng liệt kê chưa nhất quán; theo yêu cầu nghiệm thu thực tế, tối thiểu có ba hướng: (i) anomaly, (ii) attack classification, (iii) botnet/graph.

---

## 4. Bộ dữ liệu: CIC-IDS2017 và CTU-13

### 4.1. CIC-IDS2017 — dataset chính

- Nguồn: Canadian Institute for Cybersecurity (UNB).
- Tài liệu: https://www.unb.ca/cic/datasets/ids-2017.html
- Có PCAP và labeled flows/CSV tạo bằng CICFlowMeter; hơn 80 đặc trưng flow trong bản phát hành.
- Chứa benign traffic và nhiều dạng tấn công: DoS, DDoS, PortScan, Brute Force, Web Attack, Botnet và các loại khác tùy nhãn gốc.
- Mục đích: **multiclass attack classification**, Gaussian anomaly, đánh giá mô hình CNN.
- Ghi chú: dữ liệu gốc có vấn đề về flow labeling, duplicate, feature quality; phải audit trước, lưu rõ phiên bản nguồn.

#### Chiến lược nhãn

1. Giữ **nhãn gốc** để kiểm toán và truy vết.
2. Tạo mapping có version trong `configs/labels_cic_ids2017.yaml`.
3. Lựa chọn số lớp cuối cùng **sau EDA** dựa vào tính nhất quán nhãn, số mẫu và khả năng đánh giá; không cố định tùy tiện.
4. Không "gộp" mọi lỗi nhãn vào benign.
5. Báo cáo rõ loại tấn công nào đã được giữ, loại nào bị gộp hoặc loại khỏi thí nghiệm và vì sao.
6. Nếu triển khai demo PortScan, tập training phải thật sự chứa class PortScan trong mapping cuối; nếu không, chỉ đánh giá anomaly cho demo này.

#### Vai trò của PCAP

- Offline: chạy cùng flow extractor dự kiến dùng realtime (ít nhất một tập PCAP nhỏ) và kiểm tra label alignment.
- Online/replay: dùng PCAP với traffic lab được gán nhãn rõ để đo feature parity và hành vi cảnh báo.
- Không được nhận định rằng chỉ đọc CSV sẵn có đã đáp ứng đầy đủ yêu cầu "PCAP → structured data".

### 4.2. CTU-13 — dataset bổ sung

- Nguồn chính thức: https://www.stratosphereips.org/datasets-ctu13
- Có **13 scenario** botnet; gói nén tổng tải chính thức được giới thiệu có kích thước **1,9 GB** (`CTU-13-Dataset.tar.bz2`). Kích thước giải nén có thể lớn hơn nhiều.
- Ưu tiên dữ liệu **bidirectional flows** có nhãn, trong thư mục `detailed-bidirectional-flow-labels`.
- Các nhóm nhãn cần phân biệt: `Botnet`, `Normal`, `Background`, và các nhãn C&C/hành vi chi tiết theo bản phát hành.
- **Background không đồng nghĩa với Normal**: là traffic chưa xác minh đầy đủ.
- PCAP **đầy đủ payload cho toàn bộ Normal/Background/Botnet không công khai đồng nhất**; nguồn chính thức cung cấp full PCAP cho botnet và có các dạng dữ liệu đã xử lý/truncated khác. Không thiết kế thí nghiệm dựa vào giả định sai rằng mọi scenario đều có full PCAP cho cả ba nhóm.
- Một số bản phát hành chứa malware executable gốc: **không tải/chạy các file thực thi nếu không thật sự cần; tuyệt đối không chạy trong môi trường dự án thông thường**.

#### Hai nhiệm vụ chính trên CTU-13

**Bài toán A — binary botnet detection:**
- Positive: flow có nhãn botnet/C&C được xác nhận theo mapping.
- Negative: `Normal` được xác nhận.
- Unknown: `Background` không sử dụng như Negative mặc định; tách riêng hoặc đưa vào phân tích độ nhạy.
- Đánh giá precision/recall/F1/PR-AUC/FPR trên các scenario độc lập.

**Bài toán B — graph-based behavior/topology:**
- Chuyển các kết nối thành graph theo cửa sổ thời gian.
- Phát hiện dấu hiệu beaconing, centralized hub, P2P-like connectivity.
- Kết quả topology chỉ được gọi là `centralized/P2P` khi có **ground truth topology**; CTU-13 không mặc nhiên cung cấp nhãn centralized/P2P tiêu chuẩn cho mọi graph.

#### Dùng toàn bộ CTU-13 thế nào?

- **Phát triển nhanh:** 1–3 scenario đại diện, xử lý theo chunk/batch.
- **Nghiên cứu chính thức:** audit và bao phủ **cả 13 scenario** trong kế hoạch train/val/test hoặc cross-validation.
- **Không hiểu là** nạp đồng thời 1,9 GB file nén hoặc toàn bộ PCAP vào RAM/GPU.
- Cần lưu scenario ID gắn với từng mẫu để phân tách nhóm và tái lập kết quả.

### 4.3. Vì sao không chọn CIC-DDoS2019 và InSDN ngay?

- CIC-DDoS2019 tập trung sâu về DDoS, không phải nguồn lý tưởng duy nhất cho PortScan, Brute Force và botnet.
- InSDN phù hợp môi trường SDN; không thiết yếu cho phạm vi hiện tại.
- Hai dataset được chốt bổ sung cho nhau rõ ràng, giảm chi phí chuẩn hóa dữ liệu và tăng khả năng hoàn thành realtime.

### 4.4. Quy tắc kết hợp hai dataset

- Không `concat` trực tiếp CSV của CIC-IDS2017 và CTU-13 để train một classifier.
- Chuẩn hóa **tên, kiểu, đơn vị, cửa sổ thời gian** trong một `FeatureContract` khi cần tích hợp/replay.
- Dùng hai model/task-specific preprocessors độc lập nếu nguồn dữ liệu/nhãn khác nhau.
- Nếu nghiên cứu cross-dataset transfer, phải công khai schema overlap, harmonized labels và domain shift; xem đây là thí nghiệm bổ sung, không phải nghiệm thu bắt buộc.

---

## 5. Bài toán và câu hỏi nghiên cứu

### 5.1. Task 1 — Network Anomaly Detection

- **Input:** flow-/host-level features của traffic.
- **Output:** anomaly score và `normal/suspicious` ở mức flow/window.
- **Training:** chủ yếu normal/benign, không sử dụng attack labels để fit Gaussian.
- **Validation:** có thể dùng nhãn để chọn ngưỡng, mục tiêu là FPR hợp lý và recall hữu ích.
- **Metric:** Precision, Recall, PR-AUC, FPR, confusion matrix.

### 5.2. Task 2 — Attack Classification (chính)

- **Input:** flow temporal windows và/hoặc các statistics tương ứng.
- **Output:** class trong tập nhãn CIC-IDS2017 đã chốt.
- **Training:** supervised, có kiểm soát mất cân bằng lớp.
- **Metric chính:** **Macro-F1** và per-class Recall; Accuracy chỉ là thông tin bổ trợ.
- **Mô hình nghiên cứu:** **SE-AT-CNN**.

### 5.3. Task 3 — Botnet Detection (CTU-13)

- **Input:** flow/host/temporal/graph features.
- **Output:** `Botnet/Normal`, anomaly score, dấu hiệu beaconing/topology.
- **Mô hình:** graph-feature Random Forest/XGBoost, Gaussian/Isolation Forest; có thể đánh giá SE-AT-CNN nhị phân sau khi tạo sequence input tương thích.
- **Metric:** Botnet Recall, Precision, PR-AUC, FPR, event-level detection rate, detection latency.

### 5.4. Research questions (RQs)

- **RQ1:** SE-AT-CNN có cải thiện Macro-F1 và Recall các lớp tấn công so với 5 baseline trên CIC-IDS2017 không?
- **RQ2:** Mỗi module SE và AT đóng góp bao nhiêu vào khả năng phát hiện tấn công, độ ổn định và chi phí inference?
- **RQ3:** Gaussian anomaly có phát hiện được mẫu khác phân phối benign mà không làm FPR tăng quá mức không?
- **RQ4:** Kết hợp đặc trưng đồ thị và beaconing có giúp phát hiện botnet trên CTU-13 tốt hơn flow-only features không?
- **RQ5:** Mức suy giảm chất lượng khi chạy realtime/replay so với offline là bao nhiêu, và nguyên nhân là gì?

### 5.5. Giả thuyết nghiên cứu

- H1: SE cải thiện học biểu diễn kênh đặc trưng hữu ích.
- H2: Temporal Attention cải thiện việc tập trung vào các sự kiện trong chuỗi flow.
- H3: SE+AT có thể cải thiện phát hiện một số lớp, **nhưng không bảo đảm vượt toàn bộ RF/XGBoost/LightGBM**.
- H4: Đặc trưng temporal và graph làm tăng khả năng phát hiện botnet/cảnh báo sớm trong một số scenario.

Tất cả H1–H4 phải được **kiểm định thực nghiệm**, không mặc định đúng.

---

## 6. Thiết kế dữ liệu và phòng tránh data leakage

### 6.1. Đơn vị dự đoán

Phải xác định rõ:

- **Flow-level:** 1 mẫu = 1 bidirectional flow.
- **Window-level:** 1 mẫu = tập các flow/packet thuộc 1 host, connection pair hoặc time window.
- **Host-level:** 1 mẫu = tổng hợp hoạt động 1 host trong cửa sổ.

**CNN Temporal Attention cần chuỗi thật sự có trật tự thời gian.** Không được giả lập trục thời gian bằng cách đảo/chuỗi các cột tabular theo thứ tự tùy ý.

### 6.2. Định nghĩa chuỗi

Ví dụ đầu vào:

`X.shape = [batch_size, T, F]`

- `T = 16` hoặc `32` flow gần nhất (chỉ là cấu hình thử).
- `F = số feature` xác định sau khi audit.
- Grouping theo `host` hoặc `(source host, destination host/service)` phù hợp mục tiêu.
- Sort tăng dần theo thời gian; không sử dụng flow nằm trong tương lai của thời điểm inference.
- Nếu chưa đủ T flow: sử dụng padding + mask hoặc min-length policy, áp dụng nhất quán cả train/inference.
- Cửa sổ giao nhau không được bị chia qua train và test.

### 6.3. Contract đặc trưng triển khai

Các nhóm đặc trưng khả thi:

| Nhóm | Ví dụ | Ghi chú |
|---|---|---|
| Flow | duration, total packets, total bytes, avg pkt size, packets/s, bytes/s | Chỉ đo khi đủ thông tin |
| Direction | forward/backward packets, bytes | Thống nhất định nghĩa client/server, flow direction |
| Timing | mean/std/min/max packet inter-arrival | Với flow/window chưa hoàn tất cần state |
| TCP | SYN, ACK, FIN, RST counts/ratios | Không áp dụng theo cách TCP cho UDP/ICMP |
| Host | unique dst IPs, unique dst ports, connections/window | Phụ thuộc cửa sổ và host grouping |
| Temporal | consecutive-flow IAT, repeated-connection rate | Dành cho sequence/beaconing |
| Graph | degree, weighted degree, neighbor count, clustering | Tính trên graph window; không trộn tùy tiện |

**Data contract bắt buộc lưu:**

- Feature name, dtype, unit, missing policy.
- Window length, grouping key, direction convention.
- Cách chuẩn hóa, scaler parameters, category mappings.
- Thứ tự cột, model version và label mapping version.
- Quy tắc tạo sample/label từ flow và window.

Ví dụ `feature_contract.json` (minh họa, không phải config chạy được ngay):

```json
{
  "version": "1.0",
  "task": "attack_multiclass",
  "unit": "host_temporal_window",
  "sequence_length": 16,
  "window_policy": "causal_last_n_flows",
  "feature_order": ["duration_s", "total_packets", "total_bytes", "syn_count"],
  "normalizer": "saved_train_only",
  "label_mapping_file": "configs/labels_cic_ids2017.yaml",
  "notes": "Example only; finalize after data audit and feature parity tests"
}
```

### 6.4. Split và data leakage

**Nguyên tắc chung:**

1. Chia train/validation/test **trước khi fit** imputer/scaler/feature selector/SMOTE.
2. Không trùng flow hoặc window có nguồn gốc từ cùng sự kiện ở các tập.
3. Không dùng `Label`, `AttackType`, scenario ID, tên file, timestamps/IDs mang nhãn trực tiếp làm model features.
4. Phân tích ảnh hưởng của IP, port và môi trường: IP cụ thể có thể khiến mô hình học thuộc host; port có thể có giá trị nghiệp vụ nhưng phải kiểm chứng khả năng tổng quát hóa.
5. Dataset imbalance được xử lý **chỉ trong train** (class weighting trước; SMOTE/ADASYN chỉ khi thích hợp).
6. Test set bất khả xâm phạm trong quá trình tuning.

**CIC-IDS2017:**
- Tách theo đợt thu thập/sự kiện/host hoặc theo thời gian có kiểm soát.
- Cẩn thận: split theo **ngày** hoàn toàn có thể khiến một số class chỉ xuất hiện ở một phía; cần bảo đảm khả năng tính metric cho từng class.
- Đề xuất có **hai protocol**: (a) in-distribution evaluation với event/group-aware split; (b) stress test theo ngày/đợt tấn công/host chưa thấy, báo cáo rõ class coverage.
- Lưu manifest ID của mọi record trong train/val/test.

**CTU-13:**
- Tạo `scenario_id` từ metadata.
- Thực hiện cross-scenario evaluation; có thể thử 9 train / 2 validation / 2 test nếu nhãn đủ, nhưng cần kiểm tra coverage.
- Nên thử nhiều fold hoặc leave-one-scenario-out; không random flow split làm kết quả chính.
- Bất kỳ kết quả "botnet family generalization" nào cũng phải nói rõ họ botnet đã xuất hiện trong train hay chưa.

### 6.5. Label leakage tại window-level

Cần ghi cách gán nhãn:
- Ví dụ `positive if any flow in window is attack` sẽ giúp tăng detection sensitivity nhưng gây ambiguity khi attack chiếm rất ít flow.
- Có thể lựa chọn `majority`, `any`, `dominant` hoặc `last-event`, **chọn trước khi train** và thống nhất toàn bộ baseline.
- Với realtime, alert phải chỉ được đưa ra khi window đã quan sát đủ thông tin. Không đo latency bằng cách sử dụng dữ liệu tương lai.

---

## 7. EDA và tiền xử lý

### 7.1. Data audit

- [ ] Liệt kê toàn bộ file, kích thước, checksum, ngày tải, URL gốc.
- [ ] Kiểm kê nhãn, số lượng flow/packet và scenario/ngày.
- [ ] Kiểm tra encoding, separator, datatype và tên cột.
- [ ] Đếm missing, NaN, +/-Inf, zero variance, duplicated records.
- [ ] Tìm trị ngoại lai/âm phi vật lý, duration=0, impossible rate.
- [ ] Phân tích dấu hiệu target leakage qua địa chỉ mạng, thời gian, scenario ID.
- [ ] Kiểm tra tỷ lệ lớp và class coverage theo từng split.
- [ ] Với PCAP, kiểm tra flow reconstruction và label alignment.

### 7.2. EDA cần có biểu đồ

1. Class distribution theo dataset và split.
2. Traffic theo thời gian (events/giờ, bytes/s, packets/s).
3. Phân bố flow duration, packets, bytes, packet rate.
4. Unique destination ports / IPs theo host.
5. IAT distribution giữa normal và botnet/attack.
6. Correlation heatmap (sau khi loại các biến không phù hợp).
7. PCA/UMAP chỉ để trực quan (fit train-only nếu dùng trong pipeline).
8. Botnet graph degree distribution và network snapshots.
9. So sánh feature distributions train/test để phát hiện domain shift.
10. Tỷ lệ lỗi dữ liệu sau cleaning (bảng before/after).

### 7.3. Preprocessing

- Xóa các dòng không có nhãn hoặc lỗi không thể xử lý theo quy tắc đã công bố.
- Với các giá trị lỗi: impute hoặc loại bỏ, không tự ý sửa nhãn.
- Scaling StandardScaler/RobustScaler trên train; trees có thể không cần scaling.
- Categorical protocol/state: định nghĩa mapping ổn định, có unknown category.
- Mất cân bằng lớp: class weighting, focal loss (tùy chọn) và đánh giá per-class.
- Theo dõi số mẫu trước/sau từng bước.
- Lưu pipeline và danh sách feature đã chọn.

---

## 8. Năm baseline và tiêu chí so sánh

### 8.1. Mô hình đã chốt

| ID | Baseline | Loại | Vai trò |
|---|---|---|---|
| B1 | Random Forest | Classical ML | Mốc dễ giải thích, mạnh với tabular |
| B2 | XGBoost | Gradient Boosting | Baseline hiệu năng mạnh |
| B3 | LightGBM | Gradient Boosting | Tối ưu hiệu quả trên dữ liệu lớn |
| B4 | MLP | Deep Learning | Đối chứng DL không convolution |
| B5 | Vanilla 1D-CNN | Deep Learning | Đối chứng trực tiếp với SE-AT-CNN |

Gaussian và graph detectors là **module bắt buộc bổ sung**, không tính là 5 baseline của CNN đa lớp.

### 8.2. So sánh công bằng

- Dùng cùng **train/validation/test manifest** và label mapping.
- Dùng cùng ngân sách đặc trưng đầu vào theo mục tiêu công bằng; nếu model cần biểu diễn khác thì báo cáo rõ.
- Nếu CNN dùng **T flow liên tiếp**, RF/XGBoost/LightGBM/MLP nên được nhận **thông tin temporal tương đương** (ví dụ flattened sequence hoặc statistical aggregation cùng window), tránh so CNN có thêm thông tin với RF chỉ nhìn 1 flow.
- Cố định random seed, lưu hyperparameter search space, compute budget.
- Tune trên validation; không tune bằng test.
- Chạy tối thiểu 3 seeds đối với DL nếu tài nguyên cho phép, báo cáo mean ± std.
- Các bảng kết quả riêng theo **CIC-IDS2017** và **CTU-13**; không so trực tiếp F1 giữa hai nhiệm vụ như cùng điều kiện.

### 8.3. Metrics

- Classification: Accuracy, **Macro-F1**, Weighted-F1, per-class Precision/Recall/F1, confusion matrix, balanced accuracy.
- NIDS: False Positive Rate, detection rate, false alarms per time/flow nếu có.
- Botnet: Botnet Recall/Precision/F1, PR-AUC, ROC-AUC có điều kiện, calibration.
- Deployment: inference latency p50/p95, end-to-end detection latency, throughput flows/s, CPU/RAM.
- Mô hình: parameters, model size, training time.

---

## 9. Mô hình chính: SE-AT-CNN

### 9.1. Kiến trúc dự kiến

```text
Causal Temporal Flow Windows [Batch, T, F]
                  |
              Transpose
                  v
             [Batch, F, T]
                  |
           Conv1D + BN + ReLU
                  |
               SE Block
                  |
           Conv1D + BN + ReLU
                  |
          Temporal Attention
                  |
      Weighted Temporal Pooling
                  |
        Dense + Dropout + FC
                  |
          Logits / Softmax
                  |
        Attack Class Prediction
```

**Chọn Attention cụ thể:** Temporal additive attention hoặc scaled dot-product self-attention nhẹ (ưu tiên additive trong bản đầu để dễ giải thích). Trong tài liệu này, **AT = Attention**, không phải Adversarial Training.

### 9.2. CNN backbone

- Input: `[B, T, F]`, Conv1D chạy trên trục `T`.
- Filters thử: 32, 64, 128.
- Kernel thử: 3, 5.
- BatchNorm / LayerNorm theo cấu trúc tương ứng.
- ReLU hoặc GELU; Dropout/weight decay.
- Optional: residual/multi-scale block khi có thời gian; nếu thêm phải có ablation riêng để quy tác dụng đúng thành phần.

### 9.3. SE — Squeeze-and-Excitation

Giả sử output Conv1D là `U ∈ R^(B×C×T)`.

- **Squeeze:** Global Average Pooling theo chiều thời gian, thu `z ∈ R^(B×C)`.
- **Excitation:** `s = sigmoid(W2 * ReLU(W1 * z))`.
- **Reweight:** `U'_(c,t) = s_c * U_(c,t)`.

Ý nghĩa: học **trọng số của kênh feature maps**, không phải trực tiếp gán hệ số cho từng trường CSV gốc.

Thử SE reduction ratio: `r ∈ {4, 8, 16}`.

### 9.4. AT — Temporal Attention

Sau khi có temporal representations `h_t`, tính score `e_t`, chuẩn hóa:

`alpha_t = softmax_t(e_t)` với **mask** cho vị trí padding.

Vector tổng hợp:

`context = sum_t(alpha_t * h_t)`

Ý nghĩa: chọn mức độ quan trọng của các **bước thời gian**, giúp mô hình ưu tiên chuỗi tín hiệu đáng chú ý. Attention weights không tự động là giải thích nhân quả.

### 9.5. Cấu hình thử nghiệm ban đầu

| Thành phần | Giá trị đề xuất khởi đầu |
|---|---|
| Sequence length T | 16 hoặc 32 |
| CNN filters | 32 → 64 hoặc 64 → 128 |
| Kernel size | 3, 5 |
| SE reduction ratio | 4, 8, 16 |
| Dropout | 0.2–0.5 |
| Optimizer | AdamW |
| Learning rate | 1e-3 hoặc 3e-4 |
| Batch size | Tùy GPU/RAM (ví dụ 64, 128, 256) |
| Loss CIC-IDS2017 | Weighted Cross-Entropy |
| Loss CTU-13 binary | BCEWithLogits hoặc weighted CE |
| Early stopping | Dựa trên validation Macro-F1/loss đã chốt |
| Scheduler | ReduceLROnPlateau hoặc Cosine; thí nghiệm có kiểm soát |

Các giá trị **là search space ban đầu**, không phải thông số đã chứng minh tối ưu.

### 9.6. Ablation study bắt buộc

| ID | Mô hình | Mục tiêu |
|---|---|---|
| A0 | Vanilla CNN | Baseline kiến trúc |
| A1 | CNN + SE | Đo tác động channel reweighting |
| A2 | CNN + AT | Đo tác động temporal attention |
| A3 | **CNN + SE + AT** | Đo hiệu quả kết hợp |

Tất cả các mô hình phải giữ tối đa nhất quán các điều kiện: split, data contract, optimizer policy, training budget. Ghi rõ overhead về parameters/latency.

### 9.7. Kế hoạch cải thiện chất lượng

1. Kiểm tra label validity, flow/window construction và data leakage trước khi tuning.
2. Dùng temporal representation có ý nghĩa thay vì "Conv1D trên thứ tự cột ngẫu nhiên".
3. Điều chỉnh class weights, threshold và calibration thích hợp.
4. HPO vừa phải trên validation, ưu tiên Macro-F1 hơn Accuracy.
5. Xem lỗi ở từng class hiếm, vẽ confusion matrix.
6. Đánh giá trên cross-event/test khó hơn để tránh mô hình học thuộc môi trường.
7. Nếu SE-AT-CNN không vượt XGBoost/LightGBM, báo cáo trung thực; phân tích trade-off về recall, latency và điều kiện áp dụng.

---

## 10. Gaussian Anomaly Detection

### 10.1. Mục tiêu

Từ các đặc trưng của benign traffic, ước lượng phân phối và đánh dấu mẫu có độ bất thường cao.

### 10.2. Thuật toán cơ bản

Với giả định Gaussian độc lập theo feature:

`log p(x) = sum_i log N(x_i | mu_i, sigma_i^2)`

- Fit `mu_i`, `sigma_i` **chỉ bằng train normal**.
- Thêm epsilon variance để tránh chia 0.
- Ưu tiên **log-likelihood** vì tích xác suất dễ underflow.
- Chọn ngưỡng anomaly score trên validation.
- Báo cáo độ nhạy theo nhiều ngưỡng (PR curve/FPR).

### 10.3. Vấn đề giới hạn

- Đặc trưng flow có thể lệch, heavy-tailed và tương quan mạnh.
- Cần transform (log1p cho skewed nonnegative features) hoặc robust scale khi phù hợp.
- Có thể so thêm **Isolation Forest** (không thay thế phần Gaussian bắt buộc).
- **Không bắt buộc Gaussian lọc trước CNN**: nếu Gaussian loại nhầm traffic nguy hiểm, classifier sẽ không được chạy. Kiến trúc khuyên dùng là scoring song song, fusion/correlation ở alert engine.

---

## 11. Botnet Detection: temporal và graph

### 11.1. Temporal / beaconing

- Gom các sự kiện giao tiếp theo `(source host, destination host/service)`.
- Tính các khoảng cách thời gian giữa lần kết nối liên tiếp.
- Features: mean IAT, std IAT, coefficient of variation (`std/mean`), số lần kết nối, median, chu kỳ và tỷ lệ lặp.
- Chỉ phát kết luận khi đủ số lần quan sát.
- Không coi periodicity là bằng chứng độc lập: scheduler, monitoring agent và heartbeat hợp lệ cũng có thể định kỳ.

Ví dụ `10s, 10s, 11s, 10s, 9s` chỉ là **dấu hiệu cần kiểm tra**, không phải nhãn botnet.

### 11.2. Graph xây từ CTU-13

- Node: IP/host được chuẩn hóa trong phạm vi một scenario.
- Edge: flow/connection trong sliding window.
- Edge attributes: count, total bytes, protocol, destination port diversity, time, direction.
- Các kiểu graph: directed weighted graph và undirected projection tùy metric.
- Window thăm dò: 60s, 300s (tinh chỉnh sau EDA).

### 11.3. Graph features

| Feature | Ý nghĩa |
|---|---|
| In-degree / out-degree | Số host giao tiếp vào/ra |
| Weighted degree | Số lượng/khối lượng kết nối |
| Unique peers | Độ đa dạng đối tác |
| Clustering coefficient | Mức kết nối của các hàng xóm |
| Centrality / hubness | Mức tập trung giao tiếp |
| Edge recurrence | Kết nối lặp |
| Graph density / components | Đặc trưng cấu trúc theo cửa sổ |
| Degree distribution metrics | Tính phân tán/tập trung |
| Temporal regularity | Phối hợp graph với beaconing |

### 11.4. So sánh mô hình botnet trên CTU-13

Tối thiểu:
1. Random Forest + flow features.
2. Random Forest + flow + graph features.
3. XGBoost (hoặc LightGBM) + flow + graph features.
4. Gaussian / Isolation Forest unsupervised (đánh giá riêng theo bản chất).
5. Optional SE-AT-CNN binary với temporal flow windows nếu có đủ công sức.

**Đánh giá tăng thêm** khi kết hợp graph features cần so (2) với (1) trên cùng split. Đánh giá unsupervised vs supervised phải nói rõ khác biệt về lượng nhãn dùng lúc train.

### 11.5. Centralized và P2P — điều kiện để kết luận

- Centralized thường có hub/C2 tập trung, P2P có kết nối phân tán hơn.
- **Hình dạng graph không đủ để khẳng định malware hoặc topology.**
- CTU-13 không tự động là benchmark topology centralized/P2P có label chuẩn cho từng subgraph.
- Cần tạo **hai kịch bản kết nối có topology đã biết** trong lab hoặc chọn nguồn bổ sung có ground truth hợp lệ.
- Tạo test cases: centralized star/hub, P2P mesh/partial mesh, benign hub (dịch vụ hợp lệ) và benign P2P-like để kiểm tra false alarms.
- Báo cáo: topology detection accuracy/F1 trên bộ synthetic/lab này, và phân tích định tính trên CTU-13 với mức độ chắc chắn phù hợp.

---

## 12. Kiến trúc tổng thể

### 12.1. Sơ đồ high-level

```text
                        OFFLINE TRAINING
  +------------------+             +--------------------+
  | CIC-IDS2017      |             | CTU-13             |
  | PCAP + CSV       |             | BiFlows + PCAP*    |
  +---------+--------+             +----------+---------+
            |                                 |
            v                                 v
  +-------------------+           +----------------------+
  | Attack Data       |           | Botnet Data          |
  | EDA + Label Audit |           | EDA + Scenario Audit |
  +---------+---------+           +----------+-----------+
            |                                 |
            v                                 v
  +-------------------+           +-----------------------+
  | Flow/Temporal     |           | Flow/Temporal/Graph   |
  | Feature Contract  |           | Feature Contract      |
  +---------+---------+           +----------+------------+
            |                                 |
            v                                 v
  +-------------------+           +------------------------+
  | 5 Baselines       |           | Gaussian / Graph ML    |
  | CNN+SE+AT         |           | Beaconing / Botnet ML  |
  +---------+---------+           +----------+-------------+
            |                                 |
            +----------------+----------------+
                             |
                             v
                 +------------------------+
                 | Model Registry         |
                 | Model + Preprocessor   |
                 | Schema + Label Mapping |
                 +-----------+------------+
                             |
                             v
                         REALTIME
                     Linux / Scapy
                             |
                   Flow + Host Windows
                             |
              +--------------+---------------+
              |                              |
              v                              v
         Attack Model                  Botnet Pipeline
        + Gaussian Score               + Graph/Beacon
              |                              |
              +--------------+---------------+
                             |
                             v
                  Alert Engine + Logs
                 (Dashboard tùy chọn)
```

`PCAP*`: với CTU-13, không giả định có full PCAP của tất cả nhóm traffic.

### 12.2. Nguyên tắc thiết kế

- Tách rõ **data**, **training**, **inference**, **monitoring**, **evaluation**.
- Training và realtime dùng chung logic feature definitions khi phục vụ cùng task.
- Mỗi model đi kèm scaler, label encoder, schema, version.
- Hỗ trợ hai chế độ inference: **PCAP replay** và **live capture**.
- Alert system hỗ trợ nhiều nguồn điểm: classifier, Gaussian, graph/beaconing.
- Log đủ thông tin để truy vết nhưng tôn trọng quyền riêng tư.

---

## 13. Hệ thống giám sát realtime trên Linux

### 13.1. Công nghệ

- OS: Ubuntu Linux hoặc Kali Linux trong lab.
- Capture: Scapy (có thể dùng tshark/pyshark/dpkt/Zeek theo nhu cầu).
- Flow aggregation: Python service; cache key dựa trên 5-tuple + direction + timeout.
- ML inference: PyTorch, scikit-learn, XGBoost/LightGBM.
- Graph: NetworkX.
- Alerts: JSONL; SQLite tùy chọn.
- UI: Streamlit (tùy chọn).

### 13.2. Luồng xử lý chi tiết

1. Chọn interface bắt traffic (hoặc PCAP replay).
2. Packet parser phân tích IP/TCP/UDP/ICMP và timestamp.
3. Flow aggregator cập nhật bộ đếm, duration, flags, bytes, IAT.
4. Host/window aggregator tính unique destinations, rates, temporal features.
5. Feature adapter kiểm tra schema và tạo input tensors.
6. Chạy Gaussian score và SE-AT-CNN; song song đánh giá graph/beaconing theo chu kỳ.
7. Alert policy quyết định loại cảnh báo, severity và event aggregation/deduplication.
8. Ghi JSONL/SQLite và (nếu có) dashboard.

### 13.3. Tương thích offline/realtime — rủi ro quan trọng nhất

**Bắt buộc viết `test_feature_parity.py`:**
- Cùng PCAP đi qua extractor offline và replay pipeline.
- So sánh giá trị từng feature với tolerance xác định.
- Cùng feature order, units, window semantics, missing handling, scaler.
- Mẫu thiếu dữ liệu trong window phải áp dụng mask giống training.

Nếu điều này thất bại, không đưa model vào live inference dù offline accuracy cao.

### 13.4. Output cảnh báo đề xuất

```json
{
  "timestamp_utc": "2026-11-11T14:30:45Z",
  "source_ip": "192.168.56.10",
  "destination_ip": "192.168.56.20",
  "unit": "host_window",
  "window_seconds": 30,
  "detection_source": "se_at_cnn",
  "predicted_label": "PortScan",
  "model_score": 0.94,
  "anomaly_score": 7.2,
  "severity": "warning",
  "model_version": "attack-v1",
  "feature_schema_version": "1.0"
}
```

**Các giá trị trên hoàn toàn minh họa**, không phải output mô hình đã chạy. `model_score` không nên gọi là "xác suất chính xác" nếu chưa calibration. Với các loại cảnh báo khác nhau, trường `predicted_label`/`severity` cần thể hiện rõ mức độ chắc chắn.

### 13.5. Định nghĩa latency và hiệu năng

- **Inference latency:** thời gian chạy `predict/forward` cho một mẫu hoặc batch.
- **Feature delay:** thời gian chờ gom đủ flow/window.
- **End-to-end detection latency:** từ thời điểm hành vi bắt đầu đến cảnh báo đầu tiên đúng sự kiện.
- Báo cáo p50/p95, throughput và resource usage.
- Không đo mỗi `model.predict()` rồi gọi đó là "latency phát hiện realtime".

### 13.6. Lưu ý network visibility

- Sniff trên một máy **không mặc nhiên thấy traffic giữa hai máy khác**.
- Thiết kế VM topology để traffic đi qua sensor, hoặc dùng port mirroring/bridge hợp lệ.
- Capture yêu cầu quyền phù hợp; trong thực tế giới hạn quyền và interface.
- Nếu capture bị mất gói, phải theo dõi dropped packets/capture health.

---

## 14. Thiết kế thí nghiệm và đánh giá

### 14.1. Experiment matrix CIC-IDS2017

| ID | Model | Input protocol | Target | Metrics |
|---|---|---|---|---|
| C1 | Random Forest | Same-window tabular/flatten | Multi-class | Macro-F1, per-class, FPR |
| C2 | XGBoost | Same-window tabular/flatten | Multi-class | Như C1 |
| C3 | LightGBM | Same-window tabular/flatten | Multi-class | Như C1 |
| C4 | MLP | Same-window vector | Multi-class | Như C1 |
| C5 | Vanilla 1D-CNN | Temporal sequence | Multi-class | Như C1 |
| C6 | CNN + SE | Temporal sequence | Multi-class | Như C1 |
| C7 | CNN + AT | Temporal sequence | Multi-class | Như C1 |
| C8 | **SE-AT-CNN** | Temporal sequence | Multi-class | Như C1 + cost |
| C9 | Gaussian | Flow/window features | Anomaly | FPR, recall, PR-AUC |

Dấu `same-window` nghĩa là bảo đảm **cùng lượng thông tin thời gian** khi so sánh nhóm baseline và mô hình đề xuất.

### 14.2. Experiment matrix CTU-13

| ID | Model | Features | Target |
|---|---|---|---|
| T1 | Random Forest | Flow only | Botnet/Normal |
| T2 | Random Forest | Flow + Graph | Botnet/Normal |
| T3 | XGBoost/LightGBM | Flow + Graph | Botnet/Normal |
| T4 | Isolation Forest | Normal-only fit | Anomaly |
| T5 | Gaussian | Normal-only fit | Anomaly |
| T6 | Beaconing heuristic | Temporal IAT | Suspicious periodicity |
| T7 | Optional SE-AT-CNN | Temporal sequences | Botnet/Normal |
| T8 | Graph topology detector | Labeled lab synthetic graphs | Centralized/P2P/Benign |

### 14.3. Bảng kết quả cần điền sau thực nghiệm

**CIC-IDS2017:**

| Model | Accuracy | Macro-F1 | Recall (attack) | FPR | p95 inference |
|---|---:|---:|---:|---:|---:|
| RF | TBD | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD | TBD |
| LightGBM | TBD | TBD | TBD | TBD | TBD |
| MLP | TBD | TBD | TBD | TBD | TBD |
| CNN | TBD | TBD | TBD | TBD | TBD |
| CNN + SE | TBD | TBD | TBD | TBD | TBD |
| CNN + AT | TBD | TBD | TBD | TBD | TBD |
| **SE-AT-CNN** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

**CTU-13** báo cáo bảng khác, đúng protocol cross-scenario và nhóm label.

### 14.4. Tiêu chí thực nghiệm đáng tin cậy

- Có train/val/test manifest và seed.
- Có confusion matrix **từng lớp**.
- Có ít nhất một split stress-test/cross-scenario.
- Có ablation cho SE và AT.
- Có báo cáo thời gian train, inference, model size/parameters.
- Không diễn giải attention score như chứng minh nhân quả.
- Kết quả xấu, FPR cao và class hiếm vẫn phải báo cáo.
- Không đặt mục tiêu "99% accuracy" bằng mọi giá; ưu tiên đánh giá đúng.

---

## 15. Kiểm thử với mạng lab

### 15.1. Nguyên tắc an toàn

Chỉ chạy traffic mô phỏng trên **mạng lab cô lập, máy thuộc quyền kiểm soát và được phép thử nghiệm**. Giới hạn rate, thời gian và phạm vi kết nối. Không thực hiện trên hệ thống công cộng hoặc không được ủy quyền.

### 15.2. Ba kịch bản bắt buộc khuyến nghị

| Test case | Hành vi thử nghiệm | Module kỳ vọng | Bằng chứng |
|---|---|---|---|
| RT-01 PortScan | Nmap quét tập cổng trên host lab | Attack classifier hoặc anomaly | Flow stats, predicted label, alert |
| RT-02 DoS/SYN | Traffic SYN có giới hạn vào host lab | Classifier/anomaly | Spike charts, FPR, latency |
| RT-03 Beaconing | Client kết nối định kỳ tới server lab giả C2 | Temporal/graph detector | IAT chart, suspicious alert |

Ngoài ba kịch bản dương tính, cần **traffic benign đối chứng** (web browsing, ping bình thường, heartbeat hợp lệ) để đánh giá false positives.

### 15.3. Kịch bản graph topology bổ sung

- RT-G1: centralized/star synthetic traffic topology.
- RT-G2: P2P/mesh synthetic topology.
- RT-G3: benign centralized service và benign peer-to-peer.
- Ghi ground-truth graph; kiểm tra accuracy, precision, recall của topology detector.
- Không mô tả kịch bản simulated P2P là phát hiện được toàn bộ P2P botnet thật.

### 15.4. Thu thập bằng chứng

Mỗi run lưu:
- Environment và network topology.
- Script/config sinh traffic (chỉ lưu phiên bản an toàn).
- PCAP của lab và ground-truth event timestamps.
- File `alerts.jsonl`, log capture, ảnh dashboard.
- Detection latency và kết quả đúng/sai.
- CPU, RAM, packet drops nếu có.

---

## 16. Cấu trúc repository

```text
network-intrusion-detection/
├── README.md
├── PROJECT_SPEC.md
├── requirements.txt
├── pyproject.toml                  # nếu chọn packaging
├── .gitignore
├── configs/
│   ├── labels_cic_ids2017.yaml
│   ├── labels_ctu13.yaml
│   ├── feature_contract_attack.yaml
│   ├── feature_contract_botnet.yaml
│   ├── training.yaml
│   └── realtime.yaml
├── data/
│   ├── README.md
│   ├── raw/                        # Không commit dataset lớn
│   │   ├── cic_ids2017/
│   │   └── ctu13/
│   ├── interim/
│   ├── processed/
│   └── splits/                     # split manifests
├── notebooks/
│   ├── 01_cic_eda.ipynb
│   ├── 02_ctu13_eda.ipynb
│   ├── 03_gaussian.ipynb
│   ├── 04_ml_baselines.ipynb
│   ├── 05_vanilla_cnn.ipynb
│   ├── 06_se_at_cnn_ablation.ipynb
│   └── 07_botnet_graph.ipynb
├── src/
│   ├── data/
│   │   ├── pcap_parser.py
│   │   ├── flow_extractor.py
│   │   ├── feature_contract.py
│   │   ├── window_builder.py
│   │   ├── ctu13_loader.py
│   │   └── cic_ids2017_loader.py
│   ├── preprocessing/
│   │   ├── clean.py
│   │   ├── encode.py
│   │   └── splits.py
│   ├── models/
│   │   ├── gaussian.py
│   │   ├── ml_baselines.py
│   │   ├── cnn.py
│   │   ├── se_block.py
│   │   ├── attention.py
│   │   └── se_at_cnn.py
│   ├── botnet/
│   │   ├── beaconing.py
│   │   ├── graph_builder.py
│   │   ├── graph_features.py
│   │   └── graph_detector.py
│   ├── realtime/
│   │   ├── capture.py
│   │   ├── flow_buffer.py
│   │   ├── inference.py
│   │   ├── alert_manager.py
│   │   └── replay.py
│   └── dashboard/
│       └── app.py
├── scripts/
│   ├── inspect_datasets.py
│   ├── prepare_cic.py
│   ├── prepare_ctu13.py
│   ├── train_baselines.py
│   ├── train_se_at_cnn.py
│   ├── evaluate.py
│   ├── replay_pcap.py
│   └── run_realtime.py
├── artifacts/
│   ├── attack/
│   ├── botnet/
│   └── anomaly/
├── tests/
│   ├── test_flow_extractor.py
│   ├── test_feature_parity.py
│   ├── test_temporal_windows.py
│   ├── test_split_leakage.py
│   ├── test_inference.py
│   └── test_alerts.py
├── reports/
│   ├── figures/
│   ├── metrics/
│   ├── logs/
│   └── final_report/
└── docs/
    ├── architecture.md
    ├── datasets.md
    ├── experiments.md
    ├── realtime_deployment.md
    └── demo_plan.md
```

**Quy ước:** `PROJECT_SPEC.md` là đặc tả; cây thư mục này là kế hoạch chứ **chưa phải mã nguồn đã được viết**.

---

## 17. Công nghệ, hạ tầng và quản lý artifacts

### 17.1. Stack

| Hạng mục | Công nghệ |
|---|---|
| Programming | Python 3.x (pin version khi khởi tạo) |
| Data | pandas, NumPy, PyArrow/Parquet |
| EDA/Plotting | Matplotlib, Seaborn hoặc Plotly |
| Classical ML | scikit-learn, XGBoost, LightGBM |
| Deep Learning | PyTorch |
| Graph | NetworkX |
| Packet processing | Scapy; tùy chọn tshark/dpkt/pyshark/Zeek |
| Realtime UI | Streamlit (tùy chọn) |
| Model packaging | `state_dict`/`.pt`; joblib cho models/scalers tin cậy |
| Tracking | CSV/JSON logs; MLflow tùy chọn |
| Version control | Git/GitHub |

### 17.2. Hạ tầng

- Huấn luyện CNN trên **Kaggle GPU** nếu máy không có GPU mạnh.
- CPU laptop/VM có thể chạy EDA theo chunk, huấn luyện cây và inference model vừa.
- Linux VM có interface traffic quan sát được.
- Lưu ý Kaggle runtime/storage giới hạn; không tải toàn bộ PCAP lớn nếu chỉ cần labeled flow data cho thí nghiệm hiện tại.

### 17.3. Artifacts bắt buộc của một model

Mỗi model release phải có:
- Checkpoint (`.pt`, `.joblib`).
- `model_config.json` + commit SHA.
- `feature_contract.yaml/json`.
- `preprocessor.joblib` (chỉ load từ nguồn đáng tin).
- `label_mapping.json`.
- `metrics.json`.
- `split_manifest.json` hoặc chỉ mục tới phiên bản split.
- Training timestamp, dataset version và seed.

Không tải `pickle/joblib` không rõ nguồn vì có nguy cơ thực thi mã độc khi deserialize.

---

## 18. Lộ trình triển khai

### Giai đoạn 1 — Dataset Audit & EDA

**Việc làm**
- Tải CIC-IDS2017 CSV và tập PCAP cần thiết.
- Tải các scenario CTU-13 phục vụ phát triển, kiểm kê đủ 13 scenario.
- Phân tích nhãn, dữ liệu lỗi, class imbalance, cấu trúc flow.
- Đề xuất label mapping và feature contracts.
- Xác định split manifests chống leakage.

**Deliverables**
- `01_cic_eda.ipynb`, `02_ctu13_eda.ipynb`
- Data dictionary, label mapping, split design, biểu đồ chính.
- Một minh chứng PCAP→structured flow.

**Definition of Done**
- Biết rõ số mẫu/nội dung từng lớp, cách xử lý Background, và protocol chia dữ liệu.
- Không còn giả định sai rằng tất cả CTU-13 đều có PCAP đầy đủ cho mọi loại traffic.

### Giai đoạn 2 — Preprocessing + 5 Baselines

**Việc làm**
- Xây canonical feature schema theo task.
- Xây causal window generator.
- Train Random Forest, XGBoost, LightGBM, MLP, Vanilla 1D-CNN trên CIC-IDS2017.
- Gaussian anomaly trên benign.
- Đánh giá và lưu bảng metrics.

**Deliverables**
- Reproducible training scripts, baselines results, confusion matrices.
- Model artifacts và validation pipeline.

**Definition of Done**
- Có bảng baseline trên test độc lập và model có thể reload để predict lại.

### Giai đoạn 3 — SE-AT-CNN (trọng tâm nghiên cứu)

**Việc làm**
- Xây `SEBlock`, `TemporalAttention`, `SEATCNN`.
- Train A0/A1/A2/A3.
- HPO chọn lọc, class imbalance, seed repeats.
- Phân tích cải thiện/failure cases và inference overhead.

**Deliverables**
- Best model, ablation table, learning curves, per-class metrics.
- Notebook/script tái lập kết quả.

**Definition of Done**
- Có đủ 4 ablation models, điều kiện so sánh rõ ràng và kết luận trung thực.

### Giai đoạn 4 — CTU-13 Botnet & Graph

**Việc làm**
- Tiền xử lý 13 scenario, xử lý nhãn Background đúng.
- Tạo temporal/graph features theo từng window.
- So RF flow-only vs RF graph-enhanced vs XGBoost/IF/Gaussian.
- Xây beaconing detector và topology tests centralized/P2P có ground truth.

**Deliverables**
- `graph_detector`, `beaconing`, CTU-13 metrics, topology lab evaluation.

**Definition of Done**
- Có bảng so sánh botnet riêng và minh chứng nhận diện topology theo nhãn đã biết.

### Giai đoạn 5 — Linux Realtime NIDS

**Việc làm**
- Implement packet capture, flow buffer, causal windows, feature adapter.
- Export/reload model + preprocessors.
- Chạy offline replay trước, sau đó live capture.
- Ghi JSONL alerts; dashboard cơ bản nếu đủ thời gian.
- Tạo feature parity regression tests.

**Deliverables**
- `run_realtime.py`, `replay_pcap.py`, logs, config, deployment docs.

**Definition of Done**
- Một PCAP lab cho kết quả feature parity đạt tolerance và có alert hợp lý khi replay.
- Live capture quan sát được traffic thử nghiệm, không cần thiết bị tấn công thật.

### Giai đoạn 6 — Demo, Evaluation & Report

**Việc làm**
- 3 kịch bản lab, benign controls, latency/FPR/throughput.
- Chuẩn hóa bảng metric và biểu đồ offline vs realtime.
- Viết báo cáo, README và slide/demo.
- Review tất cả claim về accuracy, topology và deployment.

**Deliverables**
- Final report, demo assets, test evidence, code đã đóng gói.

**Definition of Done**
- Đối chiếu được từng yêu cầu nghiệm thu trong Mục 3 và checklist Mục 19.

---

## 19. Checklist và tiêu chí nghiệm thu

### 19.1. Dữ liệu

- [ ] Có bản mô tả và nguồn tải CIC-IDS2017/CTU-13.
- [ ] Có EDA đầy đủ 2 datasets.
- [ ] CTU-13 được audit cả **13 scenario** trong phạm vi đánh giá cuối.
- [ ] Không coi CTU-13 Background là Normal mặc định.
- [ ] Có pipeline **PCAP→flow** thực tế.
- [ ] Có feature contract và thử nghiệm offline/replay parity.
- [ ] Có split chống leakage và lưu manifest.

### 19.2. Mô hình

- [ ] Random Forest.
- [ ] XGBoost.
- [ ] LightGBM.
- [ ] MLP.
- [ ] Vanilla 1D-CNN.
- [ ] CNN + SE.
- [ ] CNN + AT.
- [ ] **CNN + SE + AT (SE-AT-CNN)**.
- [ ] Gaussian anomaly và threshold evaluation.
- [ ] Botnet ML trên CTU-13.
- [ ] Graph + beaconing module.
- [ ] Centralized/P2P có test data được gán nhãn topology.

### 19.3. Đánh giá

- [ ] Accuracy/Precision/Recall/F1/confusion matrix cho tấn công đa lớp.
- [ ] Macro-F1, per-class Recall, FPR và PR-AUC khi phù hợp.
- [ ] Có ablation chứng minh phần cải tiến.
- [ ] Có bảng so sánh ML riêng cho botnet.
- [ ] Có ít nhất một test cross-event/cross-scenario.
- [ ] Có inference latency và end-to-end detection latency.
- [ ] Báo cáo limitation và các trường hợp báo động giả.

### 19.4. Triển khai

- [ ] Chạy trên Linux.
- [ ] Sniff interface/replay PCAP.
- [ ] Trích xuất flow/window features đúng schema.
- [ ] Nạp model đã huấn luyện và predict.
- [ ] Ghi alert timestamp, IP nghi ngờ, nhãn, score, version.
- [ ] Kiểm thử **ít nhất 2–3 kịch bản trong lab**.
- [ ] Có hướng dẫn cài đặt và video/ảnh demo.

---

## 20. Rủi ro, giới hạn và phương án dự phòng

| Rủi ro | Mức ưu tiên | Giảm thiểu |
|---|---|---|
| SE-AT-CNN không vượt tree baselines | Cao | Temporal features có ý nghĩa, ablation công bằng, báo cáo F1/Recall/latency |
| Data leakage | Rất cao | Group/event/scenario split, duplicate checks, audit features |
| CSV vs realtime feature mismatch | Rất cao | Shared FeatureContract + PCAP replay parity tests |
| CTU-13 thiếu full PCAP cho normal/background | Cao | Train từ bidirectional flows; lấy lab PCAP phục vụ extractor/realtime |
| CTU-13 không có ground truth topology đầy đủ | Cao | Kịch bản lab synthetic centralized/P2P có nhãn |
| Dataset imbalance và nhãn lỗi | Cao | Per-class audit, class weighting, data quality versioning |
| Khác biệt đặc trưng giữa hai datasets | Cao | Pipelines riêng, không ghép trực tiếp |
| Linux sensor không thấy traffic | Cao | Thiết kế VM bridge/tap/port mirroring có kiểm soát |
| Latency cao do sliding windows | Trung bình | Định nghĩa window nhỏ/hợp lý, batch inference, đo end-to-end |
| Botnet alert sai do heartbeat hợp lệ | Trung bình | Kết hợp temporal+graph, benign controls, nhiều ngưỡng |
| Hết tài nguyên Kaggle | Trung bình | Dùng sampled development set rồi full evaluation, Parquet, batching |
| An ninh/rò rỉ dữ liệu PCAP | Cao | Mạng lab cô lập, hạn chế truy cập, không commit dữ liệu nhạy cảm |
| Malware executable trong CTU-13 | Rất cao | Không chạy executable; ưu tiên flow logs/PCAP cho nghiên cứu |

### Phương án fallback nếu không đủ thời gian

1. **Không cắt bỏ:** Gaussian, Attack multiclass, 5 baselines, SE-AT-CNN ablation, botnet graph, PCAP pipeline và Linux realtime (đây là yêu cầu cốt lõi).
2. Giảm phạm vi optional: dashboard đẹp, GNN, advanced ensemble, tự động block IP.
3. Dùng số ít feature có thể tái tạo chính xác từ PCAP thay vì cố mô phỏng toàn bộ CICFlowMeter.
4. Bắt đầu với 2–3 scenario CTU-13 để debug, cuối cùng đánh giá đủ 13 scenario bằng batch.
5. Nếu chưa có classifier topology mạnh trên CTU-13, phải làm rõ phạm vi: "module topology detector thử nghiệm trên graph lab có ground truth" và không suy diễn sang botnet thật.

---

## 21. Kế hoạch báo cáo và demo

### 21.1. Bố cục báo cáo

1. **Introduction:** bối cảnh NIDS, vấn đề, phạm vi, đóng góp.
2. **Background/Related Work:** classical ML, CNN, SE, Attention, Gaussian, botnet/graph.
3. **Datasets:** CIC-IDS2017, CTU-13, labels, issues, splits.
4. **Methodology:** data pipeline, 5 baselines, SE-AT-CNN, Gaussian, graph/beaconing.
5. **Experimental Setup:** compute, feature schemas, metrics, HPO, leakage prevention.
6. **Results:** CIC baselines và ablation; CTU graph/botnet; anomaly results.
7. **Real-time System:** kiến trúc, feature parity, kịch bản demo, latency/FPR.
8. **Discussion/Limitations:** domain shift, errors, model tradeoffs, graph topology limits.
9. **Conclusion/Future Work:** GNN, multi-dataset adaptation, production-hardening.

### 21.2. Kịch bản demo gợi ý

- Demo 1: đưa file PCAP lab vào pipeline, xem output flow features.
- Demo 2: xem bảng so sánh 5 baseline và ablation SE-AT-CNN.
- Demo 3: chạy replay PCAP hoặc live capture, nhận PortScan/DoS event.
- Demo 4: chạy beaconing script trong lab, hiện IAT và suspicious alert.
- Demo 5: xem graph centralized/P2P synthetic và so sánh cấu trúc.
- Demo 6: mở metrics/logs và giải thích false positives + độ trễ.

### 21.3. Câu hỏi phản biện cần trả lời

- Tại sao dùng 2 datasets mà không dùng 1?
- CNN trên dữ liệu tabular có thật sự hợp lý? Trục convolution là gì?
- SE và Attention giải quyết hai vấn đề khác nhau thế nào?
- Baselines có được tiếp cận cùng lượng thông tin temporal?
- Tại sao Accuracy cao không đủ?
- Dữ liệu có bị leakage qua host/IP/timestamp/scenario không?
- Tại sao CTU-13 Background không phải Normal?
- CTU-13 có đủ ground truth cho centralized/P2P không?
- Feature schema realtime có khớp training không?
- End-to-end latency khác `model.predict()` thế nào?
- Tại sao Gaussian nên chạy song song thay vì luôn làm bộ lọc bắt buộc?
- Hệ thống có thấy được traffic mạng hay chỉ traffic của máy sensor?

---

## 22. Hướng dẫn khởi động dự án

> Đây là **kế hoạch thao tác**, chưa phải hướng dẫn chạy repository hoàn chỉnh vì code chưa được xây dựng.

### Bước 1 — Tạo repository

Tên gợi ý: `network-intrusion-detection-se-at-cnn`.

Commit:
- `PROJECT_SPEC.md`
- `README.md`
- `.gitignore`
- `configs/`
- Cây thư mục `src/`, `tests/`, `notebooks/`

### Bước 2 — Khởi tạo môi trường

- Chọn Python/PyTorch versions tương thích môi trường Linux/Kaggle.
- Tạo virtual environment.
- Cài thư viện đã pin version trong `requirements.txt`/`pyproject.toml`.
- Xác nhận khả năng đọc PCAP trong môi trường lab.

### Bước 3 — Data Audit trước khi code CNN

- Tải CSV của CIC-IDS2017 và một tập PCAP có nhãn.
- Tải CTU-13 từ nguồn chính thức; ưu tiên bidirectional labeled flows.
- Viết `scripts/inspect_datasets.py`.
- Chốt nhãn và split, bảo đảm không leakage.
- Kiểm tra giả thuyết feature schema.

### Bước 4 — Tạo baseline đầu tiên

- Chạy RF và XGBoost trên tập train phát triển; kiểm tra metric đúng.
- Xây minimal MLP/CNN trên temporal windows.
- Lưu artifact versioned; chỉ sau đó thêm SE/AT.

### Bước 5 — Realtime sớm

- Đừng đợi tới cuối dự án mới xử lý PCAP.
- Ngay từ giai đoạn 1–2, tạo PCAP→flow prototype và parity test.
- Dùng replay để xác nhận model nhận đúng input shape và units.

---

## 23. Tài liệu tham khảo

### Dataset và kỹ thuật

1. **CIC-IDS2017 — Canadian Institute for Cybersecurity (UNB):**  
   https://www.unb.ca/cic/datasets/ids-2017.html
2. **CTU-13 — Stratosphere Laboratory:**  
   https://www.stratosphereips.org/datasets-ctu13
3. **Stratosphere Datasets Overview (nhãn và dạng PCAP/flow):**  
   https://www.stratosphereips.org/datasets-overview
4. **Scapy Documentation:**  
   https://scapy.readthedocs.io/
5. **pyshark:**  
   https://github.com/KimiNewt/pyshark
6. **dpkt Documentation:**  
   https://dpkt.readthedocs.io/
7. **scikit-learn Outlier/Novelty Detection:**  
   https://scikit-learn.org/stable/modules/outlier_detection.html
8. **NetworkX Documentation:**  
   https://networkx.org/documentation/stable/
9. **XGBoost Documentation:**  
   https://xgboost.readthedocs.io/
10. **LightGBM Documentation:**  
    https://lightgbm.readthedocs.io/
11. **PyTorch Documentation:**  
    https://pytorch.org/docs/stable/index.html

### Papers nền tảng

12. Hu, J., Shen, L., & Sun, G. (2018). **Squeeze-and-Excitation Networks**. CVPR.  
    https://arxiv.org/abs/1709.01507
13. Garcia, S., Grill, M., Stiborek, J., & Zunino, A. (2014). **An empirical comparison of botnet detection methods**. *Computers & Security*, 45, 100–123.  
    https://doi.org/10.1016/j.cose.2014.05.011
14. Sharafaldin, I., Habibi Lashkari, A., & Ghorbani, A. A. (2018). **Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization**. ICISSP.  
    Xem đường dẫn tài liệu và trích dẫn trong trang CIC-IDS2017.
15. Andrew Ng — Machine Learning Specialization, Anomaly Detection (Gaussian):  
    https://www.coursera.org/learn/machine-learning

> Khi viết báo cáo học thuật, kiểm tra lại metadata xuất bản và dùng đúng phiên bản dataset/mã nguồn đã tải. Các đường dẫn trên là nguồn tham khảo, không hàm ý mô hình đã tái tạo kết quả từ bài báo.

---

## 24. Quyết định đã chốt và việc cần xác nhận

### 24.1. Các quyết định đã chốt

- [x] Nghiên cứu theo phương án **2 datasets: CIC-IDS2017 + CTU-13**.
- [x] **CIC-IDS2017** là dataset chính cho attack classification đa lớp.
- [x] **CTU-13** cho botnet, beaconing, graph; bao phủ 13 scenario trong đánh giá cuối.
- [x] Dùng **5 baseline**: RF, XGBoost, LightGBM, MLP, Vanilla CNN.
- [x] Mô hình nghiên cứu chính: **CNN + SE + AT (SE-AT-CNN)**.
- [x] Có **ablation** 4 kiến trúc CNN.
- [x] Có **Gaussian anomaly** và module **NetworkX graph botnet**.
- [x] Triển khai **Linux realtime** và **2–3 kịch bản lab**.
- [x] Không gộp trực tiếp hai dataset.
- [x] Không nhận định trước rằng CNN cải tiến sẽ vượt baseline.
- [x] Realtime model phải tuân thủ feature contract tương thích training.

### 24.2. Chi tiết chỉ chốt sau EDA và pilot

- [ ] Danh sách nhãn multiclass cuối cùng của CIC-IDS2017.
- [ ] Quy tắc label window-level.
- [ ] Feature list thực tế dùng được cả offline/realtime.
- [ ] Bộ split chính thức cho CIC-IDS2017; folds cross-scenario cho CTU-13.
- [ ] Temporal grouping key, T và sliding window duration.
- [ ] Chính xác cơ chế Temporal Attention (additive vs self-attention).
- [ ] Quy mô hyperparameter search.
- [ ] Ngưỡng cảnh báo/FPR mục tiêu sau validation.
- [ ] Cách tạo hoặc chọn ground truth centralized/P2P.
- [ ] Mạng VM test, interface sniff và phương án quan sát traffic.

**Nguyên tắc cuối:** Tài liệu này là **bản kế hoạch/đặc tả thiết kế** để bắt đầu triển khai. Mọi chỉ số chưa đo được phải ghi `TBD`, không tự tạo số liệu; mọi thay đổi quan trọng đối với dữ liệu, nhãn, split hoặc mô hình phải được cập nhật lại vào `PROJECT_SPEC.md` và changelog.

---

## Changelog

| Phiên bản | Ngày | Nội dung |
|---|---|---|
| v2.0 | 11/11/2026 | Chốt 2 datasets CIC-IDS2017 + CTU-13; 5 baseline; SE-AT-CNN; Gaussian; botnet graph; realtime; ablation; leakage checks; nghiệm thu |
| v1.x | Trước 11/11/2026 | Phương án ban đầu tập trung CTU-13; được thay thế bởi phiên bản này |

**Hết tài liệu.**
