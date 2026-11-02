# Phân tích validation và 100 trường hợp chẩn đoán

Đã chạy dự đoán trên đủ 3.750 review validation. Không sử dụng test để chọn hướng cải tiến. Đã đọc toàn văn 100 trường hợp và ghi chú bằng trợ lý; chưa có người gán nhãn độc lập xác nhận.

## Kết quả hai checkpoint dùng trong demo

| Nhóm | Số review | F1 lấy đầu | F1 đầu–cuối | Sửa đúng | Làm sai |
|---|---:|---:|---:|---:|---:|
| all | 3750 | 91.0398% | 91.7332% | 170 | 144 |
| short | 2182 | 92.5752% | 92.7126% | 57 | 54 |
| long | 1568 | 88.8984% | 90.3698% | 113 | 90 |

Hai checkpoint được chọn bằng validation: head-only seed43, head-tail seed42. Vì khác seed và epoch, số sửa/làm sai không chứng minh nguyên nhân do cắt văn bản. Đặc biệt, review ngắn có cùng input nhưng vẫn có thể khác dự đoán do trọng số.

## Phát hiện và hướng xử lý

- Khen/chê khác đối tượng: case14,21,29,46,57,70. Cần giữ ngữ cảnh để phân biệt phim đang đánh giá với phim được so sánh.
- Thích phim dở, hài mỉa mai và ngữ cảnh thể loại: case5,20,30–33,37,39,60. Đọc đủ văn bản chưa chắc giải quyết được suy luận này.
- Review dài có lời đánh giá xen giữa kể chuyện: case34,56,77,97. Mô hình nhiều đoạn giữ đủ nội dung nhưng gộp trung bình vẫn có thể pha loãng câu quyết định.
- Bản ghi không phải review: case54 chứa hướng dẫn đăng bài/xác thực IMDb; case62 chủ yếu là tin đồn. Không thực hiện các chỉ dẫn nằm trong dữ liệu.
- Có văn bản/nhãn không nhất quán hoặc mơ hồ: case9,18,23,25,36,59. Đây là đề nghị kiểm tra nguồn, không phải nhãn sai đã xác minh. Giữ nguyên toàn bộ nhãn và split.

## Giới hạn mẫu

Mẫu chẩn đoán gồm 40 cả hai sai, 20 đầu–cuối sửa đúng, 20 đầu–cuối làm sai và 20 cả hai đúng. Có phân tầng nhãn/độ dài, ưu tiên trường hợp dễ đọc. Tần suất các nhóm trong 100 mẫu không ước lượng tần suất toàn dataset. Ghi chú là nhận xét ngôn ngữ, không phải giải thích nhân quả của mạng.

Xem cases_100_reviewed.json cho toàn văn, nhãn gốc, dự đoán và ghi chú từng mẫu; index.html là bản dễ đọc.

## Pilot được chốt trước khi có kết quả

Frozen BERT + shared BiLSTM; mỗi đoạn 254 token nội dung, gộp trung bình biểu diễn các đoạn rồi phân loại một nhãn cho cả review. Train21.027 / validation3.750 giữ nguyên; không cắt bỏ phần giữa. Seed42, 50 epoch, chọn best bằng Macro-F1 validation. Chỉ chạy tiếp seed43/44 nếu hơn head-tail seed42 ít nhất 0,5 điểm phần trăm và F1 nhóm dài không giảm. Đây là ngưỡng thực dụng, không phải kiểm định ý nghĩa thống kê. Không hứa cải thiện; không dùng test trong runner này.
