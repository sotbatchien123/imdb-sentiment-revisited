# Cải tiến baseline và mở rộng toàn bộ IMDb

## Phạm vi người dùng yêu cầu

Dùng toàn bộ nguồn 50.000 review có nhãn, không lọc Horror hoặc loại hình tác phẩm. Đây là thí nghiệm riêng với baseline Horror, không so trực tiếp điểm số trên hai tập khác nhau. Không dùng 50.000 review chưa có nhãn, không dịch tiếng Việt trong lượt này.

Train gốc có 25.000 review: loại 98 bản trùng trong train, 123 bản trùng nội dung với test và 2 review cùng phim với test. Còn 24.777 review, chia theo phim thành 21.027 train và 3.750 validation. Test giữ đúng 25.000 review. Test có 201 dòng dư do nội dung trùng bên trong; giữ nguyên và công bố. Không trùng phim/nội dung chuẩn hóa giữa các tập. Nhật ký nằm trong reports/full_imdb_data.json.

## Cải tiến và đối chứng

| Mã chạy | Thiết kế | Vai trò |
|---|---|---|
| mean | BERT đóng băng, trung bình token có attention_mask=1 (gồm token đặc biệt), dropout và Linear | Baseline đơn giản thay cho chỉ dùng CLS; cần đo, không mặc định tốt hơn |
| bilstm | BERT đóng băng + BiLSTM 768 units mỗi chiều, lấy đầu review | Đối chứng trực tiếp |
| bilstm_headtail | Cùng kiến trúc, giữ 127 token đầu + 127 token cuối khi quá dài, bao quanh bằng CLS/SEP | Phương án cải tiến giữ kết luận review |

Review không quá 256 token được giữ nguyên; review dài dùng ngân sách tổng 256 token kể cả CLS/SEP. BiLSTM dùng packed sequence để bỏ padding. Cùng split, optimizer, lịch seed và thứ tự batch theo seed/epoch. BERT luôn eval và không có gradient; chỉ huấn luyện đầu phân loại. Đây không phải fine-tune toàn bộ BERT.

Batch 8, tích lũy gradient 4 (batch hiệu dụng 32), AdamW lr 1e-4, weight decay 0.01, dropout 0.5, gradient clipping 1.0. Cập nhật theo yêu cầu người dùng: chạy đủ 50 epoch cho mỗi mô hình và mỗi seed, tắt early stopping. Lựa chọn best.pt theo validation Macro-F1, không mặc định dùng epoch 50. Thay đổi từ 15 lên 50 được thực hiện sau epoch 6 của mean seed42; trọng số, optimizer và lịch sử được giữ nguyên. Hồ sơ sao lưu/đối chiếu: reports/epoch_extension_50.json.

Ba seed 42/43/44, chín lượt chạy tuần tự. Cấu hình cố định trong configs/full_imdb.json. Không chỉnh siêu tham số theo test. Test chỉ đánh giá sau khi chọn checkpoint của mỗi lượt. Báo cáo chênh lệch head–tail so với head-only, mean/std ba seed và phân nhóm review dài/ngắn. Nếu không tăng chất lượng, báo cáo đúng kết quả âm.

Một phần test (Horror) đã được đánh giá trước đó. Không gọi đây là tập holdout chưa từng được quan sát. Các số đo benchmark này không chứng minh tính tổng quát trên review IMDb mới ngoài tập dữ liệu.

## Trạng thái baseline cũ

Lượt Horror dừng ở CLS seed43 do Windows khóa file status.json khi cập nhật. Kết quả hoàn tất của CLS/BiLSTM seed42 được giữ. Chưa hoàn tất sáu lượt Horror. Lỗi tiến độ đã sửa bằng retry và file dự phòng status_live.json; lỗi hiển thị không còn kết thúc huấn luyện. Lượt mới sử dụng thư mục work_dirs/imdb_full_v1 riêng.

## Vận hành

- run/run-full-imdb.cmd: bắt đầu hoặc tiếp tục sau checkpoint epoch cuối.
- run/pause-full-imdb.cmd: dừng sau epoch; chờ stage paused mới tắt máy.
- run/status-full-imdb.cmd: đọc trạng thái file dự phòng cập nhật.
- Chạy lại khi crash giữa epoch sẽ lặp lại epoch chưa lưu, không phải tiếp tục minibatch chính xác.
- Không tự chạy SEG song song. Không có tác vụ tự chạy lúc đăng nhập.
- reports/full_imdb_report.md và full_imdb_progress_report.json: cập nhật sau mỗi lượt hoàn tất, ghi rõ báo cáo tạm/final.
- reports/full_imdb_results.json: tổng hợp sau đủ chín lượt.
- Chưa chạy xong thì không xem báo cáo tạm là kết luận.
