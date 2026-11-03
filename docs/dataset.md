# Dữ liệu: cách tải và chuẩn bị

Repo không chứa review IMDb. Người dùng tự tải theo các bước dưới đây.

## Large Movie Review Dataset (IMDb)

1. Trang gốc: https://ai.stanford.edu/~amaas/data/sentiment/
2. File cần tải: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
   (SHA-256 `c40f74a18d3b61f90feba1e17730e0d38e8b97c05fde7008942e91923d1658fe`).
3. Đặt file vào `data/raw/aclImdb_v1.tar.gz` (đường dẫn tương đối tính từ thư mục gốc repo). Không cần giải nén.
4. Chạy `scripts/prepare_full_imdb.py`. Script đọc trực tiếp file nén, kể cả `urls_pos.txt` và `urls_neg.txt` để biết mỗi review thuộc phim nào, rồi tạo `data/processed/imdb_full_v1/`.

Kết quả chia tập:

| Tập | Review | Phim | Tiêu cực | Tích cực |
|---|---:|---:|---:|---:|
| train | 21.027 | 2.910 | 10.467 | 10.560 |
| validation | 3.750 | 514 | 1.867 | 1.883 |
| test (chính thức, giữ nguyên) | 25.000 | 3.581 | 12.500 | 12.500 |

Từ tập train gốc, script loại 223 review: 98 review trùng lặp trong train, 123 review có nội dung trùng với test, và 2 review của một phim cũng có trong test. Tập train và validation được chia theo phim, nên không có phim nào nằm ở hai tập.

## Metadata thể loại (chỉ dùng cho thí nghiệm phụ trên phim Horror)

`title.basics.tsv.gz` lấy từ https://datasets.imdbws.com/, đặt vào `data/raw/`. Đây là metadata phim, không phải review. Bộ dữ liệu này chỉ được dùng cho mục đích cá nhân, phi thương mại (xem `THIRD_PARTY_NOTICES.md`).
