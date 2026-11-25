# Minh chứng thực nghiệm

`terminal_smoke_excerpt.txt` là trích đoạn log người thực hiện gửi, không phải kết quả chạy lại. Hai audit JSON ở thư mục cha là bản gốc từ ZIP. Các file baseline_metrics.csv/json gốc, Parquet, model, split indices và log CI không nằm trong ZIP gốc; cần bổ sung từ máy phát triển khi nghiệm thu. Không dựng lại JSON đầy đủ từ bảng làm tròn.

Có thể lưu kết quả gốc bằng PowerShell tại repository:

```powershell
foreach ($kind in @('cic','ctu')) {
    New-Item -ItemType Directory -Force "reports/final/evidence/smoke/$kind" | Out-Null
    Copy-Item "outputs/final/smoke/$kind/baseline_metrics.csv" "reports/final/evidence/smoke/$kind/"
    Copy-Item "outputs/final/smoke/$kind/baseline_metrics.json" "reports/final/evidence/smoke/$kind/"
}
python -m pip freeze | Set-Content -Encoding UTF8 reports/final/evidence/environment.txt
 git rev-parse HEAD | Set-Content -Encoding UTF8 reports/final/evidence/commit.txt
```

Kiểm tra git status và nội dung trước khi commit. Không đưa raw data hoặc checkpoint vào gói báo cáo.
