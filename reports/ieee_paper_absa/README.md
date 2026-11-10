# Bài báo IEEE về ABSA (tiếng Anh)

Bài mới cho bài toán phân tích cảm xúc theo 6 khía cạnh. Bài cũ (phân loại 2 lớp) vẫn giữ nguyên ở
`reports/ieee_paper/`.

- `main.tex`: lớp `IEEEtran` (conference). Tài liệu tham khảo viết sẵn trong file, không cần BibTeX.
- `figures/`: 3 hình (PDF cho LaTeX, PNG để xem nhanh), vẽ bằng `make_figures.py` từ
  `reports/absa_v1/significance.json` và `reports/absa_v1/analysis/error_analysis.json`.
- Sơ đồ quy trình (Hình 1) vẽ bằng TikZ ngay trong `main.tex`.

## Kiểm tay bằng người

Mục **Human Verification Protocol** mô tả quy trình kiểm tay 400 review đã chuẩn bị nhưng chưa thực
hiện (ghi là giới hạn, quyết định 10/11). Nếu sau này làm: điền 3 file trong
`reports/absa_v1/human_check/`, chạy `scripts/absa_human_check.py score`, đưa số trong
`human_eval.json` vào mục đó, rồi sửa câu cuối của abstract và mục Limitations.

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
