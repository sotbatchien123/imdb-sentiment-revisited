# MovieReview — bắt đầu tại đây

> **Quy định:** đọc `GHI_NHO.md` (trạng thái hiện tại, việc tiếp theo, nhật ký phiên) trước khi làm việc; cập nhật file đó cuối mỗi phiên có thay đổi. File này là ghi chú nội bộ của máy làm việc, không có trên GitHub.

Ngày chuẩn bị: 11/10/2026. Bài tham chiếu duy nhất: Fine-tuning BERT with Bidirectional LSTM for Fine-grained Movie Reviews Sentiment Analysis (Nkhata, Gauch, Anjum, Zhan; 2023; bản arXiv 2502.20682).

## Kết quả mới nhất — 01/11/2026

Macro-F1 trên 25.000 review test chính thức IMDb, trung bình 3 seed:

| Phương pháp | Macro-F1 |
|---|---|
| Tái lập bài báo (theo mã tác giả, BERT đóng băng + BiLSTM, 256 token) | 89,41% |
| Fine-tune BERT-base, cắt đầu–cuối 512 token | 94,38% |
| **Fine-tune RoBERTa-large, cắt đầu–cuối 512 token** | **96,74%** |

Báo cáo chi tiết: `reports/BAO_CAO_CAI_TIEN_2026-11-01.docx`. Kiểm định thống kê: `reports/significance_tests.json`.

## Nguồn và license

Mã do nhóm viết trong repo này phát hành theo giấy phép MIT (xem [LICENSE](LICENSE)). Giấy phép MIT không áp dụng cho dữ liệu, mô hình hay mã của bên thứ ba nêu dưới đây.

Mã tác giả, bộ dữ liệu IMDb, metadata IMDb và các mô hình tiền huấn luyện đều thuộc về chủ sở hữu gốc, mỗi thành phần có điều kiện sử dụng riêng (một số chỉ cho phép dùng phi thương mại). Xem **[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)** trước khi dùng hoặc phân phối lại. Dữ liệu, mô hình và checkpoint không có trong repo; xem `docs/dataset.md` để tự tải.

## Tìm gì ở đâu?
| Thư mục | Nội dung |
|---|---|
| third_party/bert_bilstm_original | Bản clone mã tác giả, chỉ có ở máy cục bộ, không đưa lên Git vì repo gốc không có license; cách tải lại xem THIRD_PARTY_NOTICES.md |
| checkpoints/nllb-200-distilled-600M | Model dịch Anh → Việt, tokenizer và cấu hình tải từ Meta |
| data/raw | Bạn tự đặt dataset IMDb và metadata thể loại vào đây |
| data/processed | Dữ liệu đã lọc Horror và chia tập |
| data/translated | Bản dịch tiếng Việt; giữ liên kết với review tiếng Anh |
| scripts | Công cụ dịch offline |
| configs | Phạm vi nghiên cứu dự kiến |
| docs | Hướng dẫn dữ liệu, sử dụng và các điểm cần sửa mã gốc |
| reports | Biên bản kiểm tra tải xuống |
| work_dirs | Checkpoint/log thí nghiệm sau này |
| .cache | Cache của dự án (không đưa lên Git) |

## Baseline BERT đóng băng trên toàn bộ IMDb — 11–15/10/2026 (đã xong)

Kết quả: mean pooling 85,71%, BiLSTM lấy đầu 90,14%, BiLSTM đầu–cuối 91,87% Macro-F1 (`reports/full_imdb_results.json`). Các dòng dưới đây là ghi chú vận hành lúc chạy.

- Phạm vi mới: toàn bộ nguồn 50.000 review có nhãn, không lọc thể loại; 21.027 train / 3.750 validation / 25.000 test. Loại 223 mẫu train trùng/rò rỉ, giữ test gốc.
- Ba cấu hình: BERT mean pooling, BERT–BiLSTM lấy đầu, BERT–BiLSTM lấy đầu–cuối. Ba seed, chạy đủ 50 epoch/lượt theo yêu cầu mới; tắt early stopping, vẫn chọn best.pt bằng validation Macro-F1.
- Kế hoạch và giới hạn: docs/full_imdb_plan.md. Cấu hình: configs/full_imdb.json.
- Chạy/tiếp tục: run-full-imdb.cmd. Dừng sau epoch: pause-full-imdb.cmd. Xem tiến độ: status-full-imdb.cmd hoặc work_dirs/imdb_full_v1/status_live.json.
- Báo cáo cập nhật sau từng lượt: reports/full_imdb_report.md. Kết quả đủ chín lượt: reports/full_imdb_results.json.
- Bản Horror cũ mới hoàn tất seed42; đã dừng vì khóa file tiến độ Windows. Lỗi cập nhật tiến độ đã được sửa, không tự chạy tiếp hàng đợi cũ trong lượt toàn bộ IMDb.

## Baseline Horror trước đó — 11/10/2026
- Phạm vi: review phim điện ảnh Horror tiếng Anh, BERT đóng băng; so sánh CLS+Linear và BiLSTM, 3 seed (42/43/44).
- Dữ liệu: 2667 train / 478 validation / 3291 test; không trùng phim hoặc nội dung chuẩn hóa giữa các tập. Xem reports/data_preparation.json.
- Hướng dẫn đầy đủ: docs/baseline_run.md. Cấu hình: configs/baseline.json.
- Xem tiến độ thực tế tại work_dirs/horror_frozen_bert_v1/status.json hoặc mở status-baselines.cmd; không suy ra trạng thái đang chạy chỉ từ README.
- Khởi động/chạy tiếp: run-baselines.cmd. Dừng sau epoch: pause-baselines.cmd; đợi trạng thái paused trước khi tắt máy.
- Môi trường training riêng: .venv-train; nguồn BERT và mã băm tại reports/bert_source.json và reports/bert_verification.json.
- NLLB đã tải nhưng chưa dịch dữ liệu; môi trường dịch riêng chưa cài. Đây là model dịch, không phải checkpoint sentiment.
- Mã TensorFlow tác giả được giữ nguyên tại third_party. Không chạy trực tiếp BERT+BiLSTM-SA.py; thí nghiệm hiện tại dùng scripts/train_baselines.py.

## Dịch một tệp văn bản
1. Chạy setup-translation.cmd một lần (cần mạng để cài thư viện; không tải IMDb).
2. Đặt văn bản tiếng Anh UTF-8 vào data/raw/review_en.txt.
3. Chạy translate-en-vi.cmd. Kết quả ở data/translated/review_vi.txt.
4. Công cụ chỉ đọc model trong checkpoints, không tải model bổ sung. Mặc định dùng CPU.

Xem docs/dataset.md để tải và đặt dữ liệu. Dự án dùng tiếng Anh cho thí nghiệm chính; bản Việt là dữ liệu dịch máy cần kiểm tra chất lượng.
