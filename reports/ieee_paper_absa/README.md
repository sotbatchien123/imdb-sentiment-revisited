# Bài báo IEEE về ABSA (tiếng Anh)

Bài mới cho bài toán phân tích cảm xúc theo 6 khía cạnh. Bài cũ (phân loại 2 lớp) vẫn giữ nguyên ở
`reports/ieee_paper/`.

- `main.tex`: lớp `IEEEtran` (conference). Tài liệu tham khảo viết sẵn trong file, không cần BibTeX.
- `figures/`: 3 hình (PDF cho LaTeX, PNG để xem nhanh), vẽ bằng `make_figures.py` từ
  `reports/absa_v1/significance.json` và `reports/absa_v1/analysis/error_analysis.json`.
- Sơ đồ quy trình (Hình 1) vẽ bằng TikZ ngay trong `main.tex`.

## Chỗ còn thiếu (tô đỏ trong PDF)

Mục **Human Verification** đang chờ kết quả kiểm tay của nhóm. Khi nhóm điền xong 3 file trong
`reports/absa_v1/human_check/`, chạy `scripts/absa_human_check.py score`, lấy số trong
`human_eval.json` rồi thay đoạn `\pending{...}` bằng bảng kết quả. Sau đó sửa lại câu cuối của
abstract ("A human verification ... is in progress").

## Vẽ lại hình

Từ thư mục gốc dự án (cần matplotlib, có trong anaconda):

```
python scripts/absa_significance.py
python scripts/absa_error_analysis.py
"%USERPROFILE%\anaconda3\python.exe" reports/ieee_paper_absa/make_figures.py
```

## Biên dịch

Tên tác giả có dấu tiếng Việt nên dùng XeLaTeX hoặc Tectonic, giống bài cũ:

- Overleaf: tải thư mục này lên, đổi Compiler thành **XeLaTeX**.
- Tectonic (máy nhóm đã có sẵn bản 0.17.0 ở `E:\tools\tectonic`; bộ nhớ đệm gói TeX khoảng 46 MB đặt ở
  `E:\tools\tectonic-cache`, không để trên C:). Chạy trong thư mục này:

  ```bat
  set TECTONIC_CACHE_DIR=E:\tools\tectonic-cache
  E:\tools\tectonic\tectonic.exe -X compile main.tex
  ```

  Ra `main.pdf` (6 trang). Dòng "Fontconfig error" là vô hại.
