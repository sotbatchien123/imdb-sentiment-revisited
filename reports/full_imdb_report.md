# Báo cáo thực nghiệm IMDb toàn bộ

Trạng thái: **Hoàn thành** (9/9 lượt).

## Phạm vi và dữ liệu

Dùng 50.000 review có nhãn của IMDb làm nguồn, mọi thể loại; không dùng 50.000 review chưa có nhãn. Test giữ đủ 25.000 review gốc. Các mẫu train trùng/rò rỉ được loại có nhật ký.

| Tập | Review | Phim | Tiêu cực | Tích cực |
|---|---:|---:|---:|---:|
| train | 21027 | 2910 | 10467 | 10560 |
| validation | 3750 | 514 | 1867 | 1883 |
| test | 25000 | 3581 | 12500 | 12500 |

Loại khỏi train dẫn xuất: {"duplicate_training_text": 98, "text_in_official_test": 123, "movie_in_official_test": 2}.
Không trùng phim hoặc nội dung chuẩn hóa giữa các tập. Test có thể chứa nội dung trùng bên trong; được giữ nguyên theo nguồn.
Một phần test (Horror) đã được đánh giá trước đây; không mô tả đây là holdout hoàn toàn chưa quan sát.

## Thiết kế

- mean: BERT đóng băng + masked mean pooling + Linear; lấy đầu review.
- bilstm: BERT đóng băng + BiLSTM; lấy đầu review.
- bilstm_headtail: cùng BERT và BiLSTM nhưng giữ đầu–cuối review.
- Cùng 256 token, batch 8 × tích lũy 4, AdamW lr 1e-4, 768 hidden/chiều, dropout 0.5.
- Chạy đủ 50 epoch, tắt early stopping theo yêu cầu người dùng; chọn best.pt bằng validation Macro-F1, không mặc định dùng epoch cuối.
- Ba seed 42/43/44. Cấu hình đã cố định trước khi xem kết quả test của đợt này.
- So bilstm với bilstm_headtail để đo tác động của cách chọn nội dung; không quy mọi khác biệt với mean pooling cho head–tail.

## Kết quả từng lượt

| Mô hình | Seed | Best epoch | Accuracy (%) | Macro-F1 (%) | Thời gian train (phút) |
|---|---:|---:|---:|---:|---:|
| mean | 42 | 48 | 85.72 | 85.71 | 120.24 |
| bilstm | 42 | 48 | 90.02 | 90.02 | 223.33 |
| bilstm_headtail | 42 | 49 | 91.79 | 91.79 | 198.43 |
| mean | 43 | 49 | 85.75 | 85.75 | 63.67 |
| bilstm | 43 | 41 | 90.12 | 90.12 | 492.00 |
| bilstm_headtail | 43 | 39 | 91.76 | 91.76 | 357.84 |
| mean | 44 | 40 | 85.67 | 85.67 | 126.02 |
| bilstm | 44 | 9 | 90.28 | 90.28 | 337.37 |
| bilstm_headtail | 44 | 41 | 92.06 | 92.06 | 156.48 |

## Trung bình theo mô hình

| Mô hình | Số seed | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|---:|
| mean | 3 | 85.71 ± 0.04 | 85.71 ± 0.04 |
| bilstm | 3 | 90.14 ± 0.13 | 90.14 ± 0.13 |
| bilstm_headtail | 3 | 91.87 ± 0.17 | 91.87 ± 0.17 |

Head–tail chênh lệch **+1.73 điểm phần trăm Macro-F1** so với BiLSTM lấy đầu trên tập test này.
Đây là kết quả thực nghiệm; không khẳng định cải thiện nếu chênh lệch âm hoặc bằng 0. Ba seed chưa đủ để khẳng định tính tổng quát trên mọi dữ liệu.

## Review dài và ngắn

| Mô hình | Seed | Nhóm | Số review | Macro-F1 (%) |
|---|---:|---|---:|---:|
| mean | 42 | short | 14974 | 88.49 |
| mean | 42 | long | 10026 | 81.56 |
| bilstm | 42 | short | 14974 | 92.39 |
| bilstm | 42 | long | 10026 | 86.49 |
| bilstm_headtail | 42 | short | 14974 | 92.85 |
| bilstm_headtail | 42 | long | 10026 | 90.21 |
| mean | 43 | short | 14974 | 88.57 |
| mean | 43 | long | 10026 | 81.54 |
| bilstm | 43 | short | 14974 | 92.46 |
| bilstm | 43 | long | 10026 | 86.61 |
| bilstm_headtail | 43 | short | 14974 | 92.87 |
| bilstm_headtail | 43 | long | 10026 | 90.12 |
| mean | 44 | short | 14974 | 88.39 |
| mean | 44 | long | 10026 | 81.59 |
| bilstm | 44 | short | 14974 | 92.49 |
| bilstm | 44 | long | 10026 | 86.97 |
| bilstm_headtail | 44 | short | 14974 | 93.04 |
| bilstm_headtail | 44 | long | 10026 | 90.60 |

Nhóm long là review vượt 256 token trước khi chọn nội dung; short không vượt giới hạn.
Thời gian phụ thuộc tải GPU khác nếu có, không được coi là benchmark tốc độ phần cứng có kiểm soát.
Chi tiết classification_report, confusion_matrix và dự đoán từng review nằm trong thư mục mỗi lượt.
