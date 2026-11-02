# Baseline IMDb Horror — 11/10/2026

## Phạm vi đang thực hiện

Phân loại Positive/Negative cho review tiếng Anh. Dùng đúng một bài tham chiếu BERT–BiLSTM đã chọn; chưa dịch tiếng Việt, chưa fine-tune encoder.

- Dữ liệu gốc được sao chép từ Downloads vào data/raw/aclImdb_v1.tar.gz; không sửa/xóa bản ở Downloads.
- Metadata: https://datasets.imdbws.com/title.basics.tsv.gz; giữ bản tải và SHA256 trong báo cáo.
- Lọc genres chứa Horror và titleType = movie. Loại series, game, video, phim truyền hình và các định dạng khác để có phạm vi phim điện ảnh rõ ràng.
- 393 mã phim trong toàn bộ tập IMDb gốc không ghép được với metadata hiện tại; không tự suy đoán thể loại. Danh sách trong reports/data_preparation.json.

## Tập sử dụng

| Tập | Review | Tiêu cực | Tích cực | Phim |
|---|---:|---:|---:|---:|
| Train | 2667 | 1919 | 748 | 294 |
| Validation | 478 | 343 | 135 | 53 |
| Test | 3291 | 2112 | 1179 | 347 |

Nguồn: data/processed/imdb_horror_movies_v1. Giữ test theo split gốc sau lọc. Validation lấy theo nhóm phim từ train; seed 42, chọn ứng viên gần 15% số review và gần phân bố nhãn của train nhất. Đã loại 4 review có nội dung chuẩn hóa trùng trong train. Không trùng phim hoặc nội dung chuẩn hóa giữa các tập. Trong test có 4 dòng dư do trùng nội dung; giữ nguyên test và báo cáo rõ. Mã phim trùng train/test phát hiện ở IMDb đầy đủ không nằm trong tập phim Horror cuối cùng.

Thư mục data/processed/imdb_horror_v1 là kiểm tra trung gian chỉ lọc genre, có cả game/series; KHÔNG dùng để train. Config trỏ rõ vào imdb_horror_movies_v1.

## Mô hình và cài đặt

- Encoder google-bert/bert-base-uncased, đọc local từ checkpoints; tham số requires_grad=False và luôn eval() để tắt dropout encoder.
- B0: CLS -> Dropout(0.5) -> Linear(768,2).
- B1: chuỗi token -> Dropout(0.5) -> BiLSTM một tầng, 768 units/chiều -> ghép hidden cuối hai chiều -> Dropout(0.5) -> Linear(1536,2). Packed sequence bỏ qua padding.
- Encoder BF16 trên GPU; đầu phân loại FP32. Đây là triển khai PyTorch của thí nghiệm đã chốt, không phải chạy nguyên xi mã TensorFlow của tác giả.
- Max length 256, cắt đuôi; tỷ lệ cắt được ghi trong tokenization.json.
- Batch 8, tích lũy 4; nhóm cuối dùng đúng số mẫu thực tế.
- AdamW lr=1e-4, weight_decay=0.01, gradient clipping norm=1.0, cross-entropy không dùng trọng số lớp.
- Tối đa 10 epoch; early stop sau 2 epoch không tăng validation Macro-F1. Chọn best.pt theo validation, không theo test.
- Chạy tuần tự cls seed42, bilstm seed42, cls seed43, bilstm seed43, cls seed44, bilstm seed44.
- Đánh giá test chỉ sau khi chọn checkpoint của từng lượt. Không đổi cấu hình dựa trên kết quả test. Báo cáo mean và sample standard deviation của 3 seed.
- Seed cố định và cudnn deterministic; không tuyên bố tái lập bit-for-bit trên phần cứng khác.

## Dừng và chạy tiếp

- run-baselines.cmd: khởi động nền hoặc chạy tiếp checkpoint epoch gần nhất. Có khóa hệ điều hành ngăn hai tiến trình training đồng thời.
- pause-baselines.cmd: yêu cầu dừng sau epoch đang chạy và lưu checkpoint. Chờ status.json hiển thị paused rồi mới tắt máy.
- Nếu tiến trình dừng đột ngột giữa epoch: chạy tiếp từ cuối epoch trước, không khẳng định tiếp tục đúng minibatch.
- status-baselines.cmd: xem tiến độ. Trạng thái chi tiết: work_dirs/horror_frozen_bert_v1/status.json.
- Chưa thiết lập tự chạy khi đăng nhập hoặc khởi động máy.

## Kết quả ở đâu?

- Mỗi lượt: work_dirs/horror_frozen_bert_v1/<model>_seed<seed>/ gồm latest.pt, best.pt, history.json, test_metrics.json, test_predictions.jsonl.
- Tổng hợp khi đủ 6 lượt: reports/baseline_results.json.
- Kiểm tra GPU và phần head: work_dirs/horror_frozen_bert_v1/smoke_report.json. Số liệu smoke chỉ là kiểm tra kỹ thuật, không dùng báo cáo chất lượng.

Mã gốc tại third_party/bert_bilstm_original được giữ nguyên để đối chiếu.
