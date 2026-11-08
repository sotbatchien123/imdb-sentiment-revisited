# Hướng dẫn kiểm tay nhãn ABSA (tập test)

Mỗi thành viên mở **một** file `annotator_<số>.xlsx` (mỗi người một số), làm trong sheet
**Gan nhan**. Sheet **Huong dan** có định nghĩa nhãn và khía cạnh.

| | |
|---|---|
| Số review mỗi người | 200 (100 review chung cho cả 3 người + 100 review riêng) |
| Tổng số review khác nhau | 400, lấy ngẫu nhiên từ test, phân tầng theo số sao, mỗi phim tối đa một review |
| Độ dài trung bình | khoảng 230 từ một review |
| Khía cạnh | acting, story, visual, sound, direction, overall |
| Nhãn | positive, negative, neutral, none |

## Cách làm

1. Đọc cột **Review**.
2. Mỗi khía cạnh có cột nhãn **LLM**, cột **Bằng chứng LLM** (câu LLM trích ra) và cột vàng
   **NGƯỜI**. Chọn nhãn của bạn trong danh sách ở cột NGƯỜI. Đồng ý với LLM thì chọn đúng nhãn đó.
3. Không bỏ trống ô NGƯỜI nào. Có thắc mắc thì ghi ở cột **Ghi chú**.
4. **Không bàn nhau** khi gán 100 review chung. Các review đó dùng để đo độ đồng thuận giữa người
   với người (Fleiss κ). Thứ tự review trong mỗi file đã được xáo, nên không biết review nào là review chung.
5. Đừng sửa cột `review_id`, đừng đổi tên sheet hay tên cột.

Gợi ý: đọc nhanh review, nhìn câu bằng chứng của LLM để kiểm tra. Lưu ý thêm hai điều:
- Một khía cạnh có thể được nhắc tới mà LLM bỏ sót. Khi đó nhãn LLM là none, câu bằng chứng trống.
- Câu khen/chê chung về cả bộ phim ("great movie") chỉ tính cho overall.

## Khi xong

Lưu file và đặt lại vào thư mục này với đúng tên cũ, rồi chạy:

```bat
"%USERPROFILE%\anaconda3\python.exe" scripts\absa_human_check.py score
```

Kết quả ghi vào `human_eval.json`:
- độ đồng thuận giữa người với người;
- điểm của LLM (Qwen3-8B) so với người;
- điểm của 3 mô hình × 3 seed so với người.

## Lưu ý về phương pháp

File hiện nhãn LLM để kiểm cho nhanh. Vì vậy nhãn người có thể bị kéo về phía nhãn LLM (hiệu
ứng mỏ neo), và điểm "LLM so với người" có thể cao hơn thực tế. Bài báo phải ghi rõ điều này.
Nếu có thời gian, có thể tạo bộ file ẩn nhãn LLM bằng
`scripts\absa_human_check.py prepare --blind --out-dir reports/absa_v1/human_check_blind`.
