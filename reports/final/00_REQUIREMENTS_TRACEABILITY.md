# 0. Đối chiếu yêu cầu môn An toàn thông tin

**Đề tài:** Phát hiện tấn công mạng dựa trên CNN mở rộng (CNN + SE + MHSA).
**Mốc báo cáo:** 25/11/2026 — hoàn thiện tài liệu giai đoạn offline smoke, chưa nghiệm thu cuối kỳ.

Nguồn yêu cầu: `Project_ATTT.docx` do giảng viên cung cấp. Bảng này chỉ đánh giá bằng chứng hiện có, không mặc định mã nguồn cũ đã đáp ứng mọi tiêu chí.

| Yêu cầu | Minh chứng hiện có | Trạng thái / tiêu chí hoàn thành |
|---|---|---|
| Ít nhất hai dataset | [02 EDA](02_DATASET_EDA.md) | Đã có CIC và CTU CSV |
| EDA, làm sạch, chuẩn hóa | [02](02_DATASET_EDA.md), [03](03_DATA_PREPROCESSING.md) | Đã thực hiện; bổ sung thống kê mô tả |
| Xử lý mất cân bằng | [03](03_DATA_PREPROCESSING.md) | Có class weighting CIC; cần so sánh validation |
| Gaussian anomaly detection | [04](04_GAUSSIAN_ANOMALY.md) | Có smoke; recall thấp, cần đánh giá ngưỡng |
| ML baseline và phân loại đa lớp | [05](05_ML_BASELINES.md) | Bốn baseline smoke; cần full training |
| So sánh thuật toán phát hiện botnet | [05](05_ML_BASELINES.md) | Có CTU binary smoke; chưa đủ kết quả chính thức |
| Accuracy, Precision, Recall, F1, confusion matrix | [04](04_GAUSSIAN_ANOMALY.md), [05](05_ML_BASELINES.md) | Có smoke; bổ sung per-class full |
| CNN + SE + Attention | Chưa có kết quả mới trong gói | Huấn luyện trên cùng split với baseline |
| Chọn siêu tham số và phân tích đóng góp | Chưa có | Tuning validation và 4 ablation: CNN, CNN+SE, CNN+MHSA, CNN+SE+MHSA |
| CNN có metric tốt hơn baseline trên cùng dữ liệu | Chưa chứng minh | So sánh công bằng, không sửa số liệu nếu không vượt |
| Beaconing | Thiếu timestamp/IP trong CSV | Cần PCAP hoặc flow có timestamp và địa chỉ |
| Botnet graph centralized/P2P | Chưa có minh chứng | Dựng graph, kiểm thử hai cấu trúc có ground truth |
| PCAP thành flow-level và host-level features | Chưa có minh chứng | Đối chiếu feature parity với pipeline huấn luyện |
| Linux realtime, sliding window, cảnh báo | Chưa có minh chứng | Capture được phép, inference, log thời gian/IP/loại/confidence |
| Kiểm thử ít nhất 2–3 kịch bản mô phỏng | Chưa có | Lab cô lập, ghi nhãn ground truth, đo detection rate, FPR, latency |
| So sánh offline và realtime, ưu/nhược điểm | Mới có nhận xét offline | Hoàn thiện sau lab |
| Dashboard / firewall | Nâng cao, tùy chọn | Ưu tiên dashboard, không coi firewall là bắt buộc |
| Mã nguồn và báo cáo | 5 báo cáo, 36 ảnh, 2 audit JSON | Bổ sung artifact và commit tương ứng |

## Thứ tự triển khai tiếp theo

1. Hoàn thiện CNN và bốn ablation, thống nhất split/metric, chạy smoke CPU.
2. Kaggle full training; chốt tham số trên validation; xuất checkpoint, metrics, per-class support, loss curves và báo cáo 06–08.
3. Kiểm chứng PCAP/flow parity, bổ sung metadata thật cho graph và beaconing; báo cáo 09–10.
4. Linux realtime và kiểm thử có kiểm soát, dashboard, đo hiệu năng; báo cáo 11–13.
5. Tổng hợp kết luận, ghi rõ giới hạn, hoàn thiện báo cáo chính thức (14).

## Quy ước nghiệm thu

Mỗi phần tiếp theo cần mã nguồn, test, số liệu gốc, hình ảnh, báo cáo Markdown và commit đã kiểm tra. Không tạo file báo cáo kết quả giả cho các hạng mục chưa thực nghiệm. Kết quả hiện tại là smoke trên split nội bộ; không tuyên bố đã triển khai Linux hoặc CNN vượt baseline.
