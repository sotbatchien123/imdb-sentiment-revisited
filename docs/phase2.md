# MovieReview — giai đoạn 2

Phạm vi đã được chấp nhận: phân tích lỗi validation; pilot đọc nhiều đoạn; demo so sánh. Không dịch dataset và không đổi nhãn/split trong giai đoạn này.

## Mở kết quả

- Demo đang phục vụ tại http://127.0.0.1:8765. Khi cần mở lại, chạy `run/run-demo.cmd` rồi truy cập địa chỉ này. Demo dùng CPU, không chiếm GPU huấn luyện. Nếu cổng đang được dùng bởi demo, chỉ cần mở trình duyệt; không khởi động bản thứ hai.
- Báo cáo: `reports/validation_analysis/report.md`.
- Bản đọc 100 trường hợp: `reports/validation_analysis/index.html`.
- JSON đủ 100 trường hợp đã đọc/ghi chú: `reports/validation_analysis/cases_100_reviewed.json`.
- Tổng hợp validation: `reports/validation_analysis/summary.json`.
- Kết quả kiểm tra API: `reports/demo_checks.json`.

## Thí nghiệm nhiều đoạn

`configs/multichunk_pilot.json` chốt giao thức trước khi có kết quả. Dữ liệu giữ nguyên 21.027 train / 3.750 validation. Runner không đọc test.

Frozen BERT-base-uncased ở eval mode, BF16 trên CUDA; BiLSTM 768 mỗi chiều, FP32, dropout 0,5. Chia toàn bộ review thành các đoạn liên tiếp, không chồng lấn, tối đa 254 token nội dung + CLS/SEP. Dùng chung BERT/BiLSTM cho các đoạn; lấy trung bình biểu diễn 1536 chiều của chúng rồi phân loại một nhãn cho cả review. Loss tính một lần cho mỗi review, không biến nhãn review thành nhãn từng đoạn.

Mỗi batch 2 review, tích lũy 16 lần (32 review/update; nhóm cuối được chuẩn hóa đúng số review). Encoder chạy tối đa 4 đoạn/microbatch; activation checkpointing ở head. AdamW lr=1e-4, weight_decay=0,01, clip gradient=1. Huấn luyện từ đầu head seed42 trong 50 epoch, không early stopping; chọn checkpoint tốt nhất bằng Macro-F1 validation.

Train tạo 33.962 đoạn, dài nhất 10 đoạn/review; validation 6.173 đoạn, dài nhất 12 đoạn/review. Không mất token nội dung. Phần encoder tốn thêm tính toán so với đầu–cuối; đây không phải so sánh đồng ngân sách FLOPs. Bộ nhớ GPU cấp phát đỉnh trong smoke test khoảng 787 MiB, không phải tổng VRAM cả hệ thống.

Sau seed42, chỉ chạy tiếp seed43 và seed44 (50 epoch mỗi seed) nếu:

1. Macro-F1 validation của best checkpoint >= head-tail seed42 + 0,005.
2. Macro-F1 trên 1.568 review validation dài không thấp hơn head-tail seed42.

Mốc tham chiếu: overall 0,9173321811; long 0,9036980004. Đây là ngưỡng quyết định thực dụng đã chốt, không phải kiểm định thống kê. Không đạt thì runner kết thúc và ghi `pilot_gate.json`, không tự tìm thêm cấu hình. Đạt thì sau cả ba seed, chọn checkpoint bằng validation và ghi `configs/multichunk_deploy.json`; khởi động lại demo để nạp mô hình thứ ba. Không dùng checkpoint chưa qua giao thức này trong demo mặc định.

## Theo dõi / tiếp tục

- Trạng thái: `work_dirs/multichunk_v1/status_live.json`; có PID thật, thời gian cập nhật, seed, epoch và số review đã xử lý. Trạng thái trên UI là bản ghi, không thay thế kiểm tra PID khi máy ngủ/tắt.
- `work_dirs/multichunk_v1/seed42/history.json` được tạo sau epoch đầu tiên.
- Mỗi seed có `latest.pt` (optimizer/RNG/history), `best.pt`, `best_validation_predictions.json`, `result.json` khi hoàn tất.
- `smoke_results.json`: kiểm tra token coverage, pooling review, backward review train dài nhất, frozen encoder, checkpoint và cập nhật tiếp sau restore optimizer/RNG.
- Log nền: `logs/multichunk_*.out.log` và `.err.log`.
- Để tạm dừng có lưu trạng thái: tạo file rỗng `work_dirs/multichunk_v1/PAUSE`; runner dừng tại ranh giới epoch sau khi lưu. Xóa marker khi đã muốn tiếp tục rồi chạy `run/resume-multichunk.cmd`. Không chạy hai runner đồng thời.
- Khi tắt đột ngột, epoch đang dở sẽ chạy lại; những epoch đã lưu được giữ. Cấu hình, dữ liệu và mã nguồn có signature: không sửa các file `train_multichunk.py`, `phase2_core.py` hoặc config giữa chừng rồi tiếp tục cùng thư mục.
- Máy phải thức và có nguồn điện để job chạy liên tục. Không tự thay cấu hình sleep của hệ điều hành.

## Ý nghĩa kết quả

Hai checkpoint demo chọn bằng validation, không bằng test: head-only seed43/epoch41; head-tail seed42/epoch49. Demo CPU FP32, còn đánh giá offline encoder BF16; các xác suất có thể khác nhẹ. Hai mô hình có seed/epoch khác nhau nên sự khác biệt trên từng review không chứng minh tác động nhân quả của chiến lược cắt văn bản.

Đã chạy validation đủ 3.750 mẫu: F1 91,0398% và 91,7332%. 170 review chuyển sai→đúng, 144 đúng→sai, 166 cả hai sai, 3.270 cả hai đúng. Mẫu chẩn đoán 100 trường hợp được đọc toàn văn và ghi chú bởi trợ lý; chưa có người đánh giá độc lập. Các dấu hiệu nghi ngờ nhãn hoặc văn bản rác không được dùng để tự sửa dữ liệu.

Chưa có kết quả pilot cuối cùng. Sau khi hoàn thành giao thức và cố định lựa chọn, mới quyết định đánh giá test của mô hình cuối. Test IMDb đã được xem trong giai đoạn baseline nên không gọi đây là holdout chưa từng đụng đến. Nếu pilot không tốt hơn, giữ kết quả âm và giải thích giới hạn của gộp trung bình; không mặc định rằng đọc thêm token phải tăng điểm.

## Chạy lại kiểm tra

```powershell
.\.venv-train\Scripts\python.exe scripts\check_demo.py
.\.venv-train\Scripts\python.exe scripts\train_multichunk.py --smoke
```

Smoke là lượt kiểm tra riêng, không được tính vào 50 epoch. Không chạy smoke đồng thời với training trừ khi chủ động muốn chia sẻ GPU. Không cần cài thêm thư viện hoặc tải thêm model.

## Mã mới

`scripts/phase2_core.py`, `scripts/analyze_validation.py`, `scripts/build_analysis_report.py`, `scripts/qualitative_notes.tsv`, `scripts/train_multichunk.py`, `scripts/demo_server.py`, `scripts/check_demo.py`, `demo/` và hai launcher ở thư mục gốc. Mã upstream và baseline cũ được giữ nguyên.
