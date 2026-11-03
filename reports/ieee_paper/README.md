# Bài báo IEEE (tiếng Anh)

- `main.tex`: bài báo theo lớp `IEEEtran` (conference), tài liệu tham khảo viết sẵn trong file nên không cần BibTeX.
- `figures/`: 4 hình (PDF dùng cho LaTeX, PNG để xem nhanh), vẽ bằng `make_figures.py` từ `reports/improvement_report/report_data.json`.

## Biên dịch

Bản `main.pdf` hiện có được biên dịch bằng Tectonic 0.17.0 (XeTeX), 6 trang. Tên tác giả có dấu tiếng Việt nên nên dùng engine XeTeX: khi đó file tự chọn font TeX Gyre Termes (giống Times, đủ dấu tiếng Việt).

Cách 1, Overleaf: New Project → Upload Project → chọn `overleaf_upload.zip`; vào Menu, đổi Compiler thành **XeLaTeX**, rồi bấm Recompile.

Cách 2, Tectonic trên máy (tự tải gói khi cần):

```
tectonic main.tex
```

Với pdfLaTeX, file sẽ chuyển sang `fontenc` T5. Cách này cần gói `vntex`, và nếu font không có dấu tiếng Việt thì chữ có dấu có thể bị thiếu.

Vẽ lại hình (từ thư mục gốc dự án, cần matplotlib và font Times New Roman):

```
python reports/ieee_paper/make_figures.py
```

## Cần điền trước khi nộp

- Khối `\author` đang có họ tên và mã sinh viên; nếu cần, bổ sung khoa, trường, email.
