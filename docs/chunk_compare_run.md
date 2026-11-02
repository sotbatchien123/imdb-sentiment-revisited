# Phương án C — runner B2/C1 (chunk_compare_v1)

Kế hoạch gốc: `reports/KE_HOACH_C_CHUNK_ATTENTION_2026-10-26.md`. Chẩn đoán CUDA: `reports/chunk_attention_v1/cuda_diagnosis.json`.

## Thành phần

| File | Vai trò |
|---|---|
| `scripts/chunk_attention_model.py` | Head dùng chung BiLSTM; gộp `mean` (B2) hoặc `attention` 128 chiều (C1); khởi tạo chung theo seed; tiện ích tích lũy gradient và thống kê attention |
| `scripts/train_chunk_compare.py` | Train/validation, pause, checkpoint, resume, status, `error.json`; `--smoke` kiểm tra với BERT thật trong thư mục riêng |
| `scripts/evaluate_chunk_compare.py` | `compare`: gate pilot C1 vs B2 → `reports/chunk_attention_v1/pilot_comparison.json`; `evaluate`: đánh giá checkpoint best, test bị khóa |
| `scripts/tests/test_chunk_attention.py` | 13 unit test CPU |
| `configs/chunk_compare_v1.json` | Giao thức; `mode` đang là `planned` |

B2 và C1 chỉ khác phép gộp. Cùng seed thì: BiLSTM/classifier khởi tạo giống hệt (ghim hash ở `work_dirs/chunk_compare_v1/shared_init_seed{seed}.pt`); attention khởi tạo bằng nguồn RNG riêng; thứ tự review dùng generator riêng; RNG dropout bắt đầu như nhau. Không có đoạn padding: softmax tính riêng trên các đoạn thật của từng review.

## Lệnh

```powershell
# Unit test (CPU)
.venv-train\Scripts\python.exe -m unittest discover -s scripts\tests -v
# Smoke (BERT thật; thư mục work_dirs\chunk_compare_v1\smoke, không tính vào 50 epoch)
.venv-train\Scripts\python.exe scripts\train_chunk_compare.py --smoke --aggregation attention --device cuda
# Lượt chính — chỉ sau khi đổi "mode" thành "frozen"; chạy tuần tự B2 rồi C1
.venv-train\Scripts\python.exe -u scripts\train_chunk_compare.py --aggregation mean --seed 42
.venv-train\Scripts\python.exe -u scripts\train_chunk_compare.py --aggregation attention --seed 42
# Gate pilot sau khi cả hai lượt xong
.venv-train\Scripts\python.exe scripts\evaluate_chunk_compare.py compare --seed 42
```

Không chạy bằng `python -I` (cờ này bỏ thư mục `scripts` khỏi đường dẫn import).

## Vận hành

- Mỗi lượt ghi vào `work_dirs/chunk_compare_v1/{mean|attention}_seed{seed}/`: `status.json`, `history.json`, `latest.pt` (optimizer/RNG/thứ tự batch), `best.pt`, `best_validation_predictions.json` (kèm trọng số đoạn), `result.json`; lỗi → `error.json` (giai đoạn lỗi, epoch hoàn tất cuối). Không tự khởi động lại.
- Tạm dừng: tạo file `PAUSE` trong `work_dirs/chunk_compare_v1/` hoặc thư mục lượt; dừng ở ranh giới epoch. Xóa marker rồi chạy lại đúng lệnh để resume.
- Khóa `runner.lock` chặn hai lượt chạy cùng lúc. Signature gồm config, mã nguồn và dữ liệu: không sửa giữa chừng một lượt.
- Seed 43/44 chỉ chạy được khi `pilot_comparison.json` có gate đạt. Runner không tự deploy và không đọc test.
- Khi gặp lỗi CUDA (xem chẩn đoán): chạy lại lệnh trong tiến trình mới để resume từ epoch hoàn tất cuối. Tránh game/ứng dụng 3D khi chạy dài.
