# Dataset — người dùng tự tải

Không có review IMDb nào được tải trong bước chuẩn bị này.

1. Trang gốc: https://ai.stanford.edu/~amaas/data/sentiment/
2. Tệp tải: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
3. Đặt tệp tại E:\Project\MovieReview\data\raw\aclImdb_v1.tar.gz và giải nén thành E:\Project\MovieReview\data\raw\aclImdb\ (bên trong có train, test, README).
4. Giữ các tệp urls_pos.txt và urls_neg.txt: dùng liên kết phim để ghép thể loại.
5. Metadata thể loại: https://datasets.imdbws.com/title.basics.tsv.gz — tự tải về data/raw/title.basics.tsv.gz. Đây là metadata, không phải review.

Phạm vi đề xuất: phim có nhãn Horror (có thể đồng thời có thể loại khác). Chưa thực hiện ghép và chưa biết số review còn lại. Giữ train/test gốc sau lọc, lấy validation từ train theo nhóm phim. Không dùng test để lựa chọn cấu hình. Không gộp bản dịch và bản gốc vào các tập khác nhau; giữ cùng review_id và split.
