# Ghi chú mã tác giả

Nguồn: https://github.com/gnkhata1/Finetuning-BERT-on-Movie-Reviews-Sentiment-Analysis
Bài: https://arxiv.org/abs/2502.20682
Mã trong third_party được giữ nguyên. Chưa chạy hoặc sửa mã huấn luyện.

Các điểm đã đọc và cần xử lý trước thí nghiệm:
- Hàm download_and_load_datasets tự tải IMDb; không chạy khi chưa chuyển sang đọc dữ liệu cục bộ.
- Gộp train/test gốc bằng np.concatenate rồi chia lại: cần giữ official split cho kế hoạch của nhóm.
- Test có shuffle; nhãn lấy ở lượt duyệt khác có thể lệch thứ tự dự đoán.
- return trong extract_labels nằm trong vòng lặp, chỉ lấy batch đầu.
- Code đóng băng lớp BERT bằng model.layers[2].trainable=False; cần ghi đúng chế độ huấn luyện thực tế, không mặc định gọi là fine-tune toàn bộ BERT.
- Tham số và API TensorFlow cũ cần đối chiếu trước khi cài môi trường. Môi trường dịch PyTorch tách biệt với môi trường training gốc TensorFlow.

Kế hoạch: BERT với đầu phân loại thông thường so với BERT–BiLSTM, cùng tập IMDb Horror, báo cáo Accuracy/Macro-F1 và lỗi theo ngữ cảnh. Đây là kế hoạch, chưa phải kết quả.
