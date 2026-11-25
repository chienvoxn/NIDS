# 7. Thực nghiệm Ablation Study

Checkpoint được lựa chọn bằng F1-score trên validation.
CIC sử dụng Weighted F1; CTU sử dụng Binary F1.

> Đây là kết quả Smoke Test, không phải kết quả huấn luyện chính thức.

## CIC-IDS2017

| Mô hình | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| CNN | 52.3486% | 83.9756% | 52.3486% | 59.9101% |
| CNN + SE | 23.3260% | 79.1121% | 23.3260% | 27.4130% |
| CNN + MHSA | 63.8217% | 88.5404% | 63.8217% | 69.8308% |
| CNN + SE + MHSA | 74.3754% | 85.1470% | 74.3754% | 77.8075% |

### Nhận xét

Cấu hình có Validation F1 cao nhất: CNN + SE + MHSA.

Mô hình đề xuất đạt Test F1-score 77.8075%.


![01_training_loss](figures/smoke/cic/cnn/01_training_loss.png)


![02_validation_loss](figures/smoke/cic/cnn/02_validation_loss.png)


![03_validation_accuracy](figures/smoke/cic/cnn/03_validation_accuracy.png)


![04_validation_f1](figures/smoke/cic/cnn/04_validation_f1.png)


![05_ablation_f1](figures/smoke/cic/cnn/05_ablation_f1.png)


![06_cnn_metrics](figures/smoke/cic/cnn/06_cnn_metrics.png)


![07_confusion_cnn](figures/smoke/cic/cnn/07_confusion_cnn.png)


![07_confusion_cnn_mhsa](figures/smoke/cic/cnn/07_confusion_cnn_mhsa.png)


![07_confusion_cnn_se](figures/smoke/cic/cnn/07_confusion_cnn_se.png)


![07_confusion_cnn_se_mhsa](figures/smoke/cic/cnn/07_confusion_cnn_se_mhsa.png)


![10_training_time](figures/smoke/cic/cnn/10_training_time.png)


![11_inference_time](figures/smoke/cic/cnn/11_inference_time.png)


![12_model_complexity](figures/smoke/cic/cnn/12_model_complexity.png)


## CTU-13

| Mô hình | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| CNN | 95.0000% | 91.2308% | 95.7481% | 93.4349% |
| CNN + SE | 94.9400% | 92.2591% | 94.2949% | 93.2659% |
| CNN + MHSA | 95.2000% | 91.9606% | 95.4252% | 93.6609% |
| CNN + SE + MHSA | 95.2400% | 91.4534% | 96.1787% | 93.7566% |

### Nhận xét

Cấu hình có Validation F1 cao nhất: CNN + MHSA.

Mô hình đề xuất đạt Test F1-score 93.7566%.


![01_training_loss](figures/smoke/ctu/cnn/01_training_loss.png)


![02_validation_loss](figures/smoke/ctu/cnn/02_validation_loss.png)


![03_validation_accuracy](figures/smoke/ctu/cnn/03_validation_accuracy.png)


![04_validation_f1](figures/smoke/ctu/cnn/04_validation_f1.png)


![05_ablation_f1](figures/smoke/ctu/cnn/05_ablation_f1.png)


![06_cnn_metrics](figures/smoke/ctu/cnn/06_cnn_metrics.png)


![07_confusion_cnn](figures/smoke/ctu/cnn/07_confusion_cnn.png)


![07_confusion_cnn_mhsa](figures/smoke/ctu/cnn/07_confusion_cnn_mhsa.png)


![07_confusion_cnn_se](figures/smoke/ctu/cnn/07_confusion_cnn_se.png)


![07_confusion_cnn_se_mhsa](figures/smoke/ctu/cnn/07_confusion_cnn_se_mhsa.png)


![10_training_time](figures/smoke/ctu/cnn/10_training_time.png)


![11_inference_time](figures/smoke/ctu/cnn/11_inference_time.png)


![12_model_complexity](figures/smoke/ctu/cnn/12_model_complexity.png)
