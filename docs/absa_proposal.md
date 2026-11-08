# Đề xuất chuyển sang phân tích cảm xúc theo khía cạnh (ABSA)

Ngày lập: 04/11/2026. Trạng thái: **đang triển khai**. Cách chạy: [absa_run.md](absa_run.md).

**Quyết định đã chốt (04/11/2026):** review tiếng Anh IMDb; 7 khía cạnh (Acting, Story, Visual,
Sound, Direction, Pacing, Overall); gán nhãn toàn bộ bằng LLM chạy local (Qwen3-8B qua llama.cpp,
không tốn phí), nhóm kiểm tra sau; bộ tham chiếu 300 review validation gán tay; 3 mô hình là
BERT đóng băng + BiLSTM, BERT-base FT và RoBERTa-large FT; thời hạn khoảng 10 ngày. Các mục bên
dưới là phân tích ban đầu; chỗ nào khác với quyết định này thì theo quyết định này.

## 1. Yêu cầu mới

Mỗi review phải được gán nhãn cảm xúc cho **từng khía cạnh**, không chỉ cho cả review:

| Khía cạnh | Nhãn cho mỗi khía cạnh |
|---|---|
| Acting (Diễn xuất) | Positive / Negative / Neutral / *None* (không nhắc tới) |
| Story (Cốt truyện) | như trên |
| Visual (Hình ảnh) | như trên |
| Sound (Âm thanh) | như trên |

Bài toán này là **Aspect-Category Sentiment Analysis (ACSA)**. Dự án hiện tại mới làm
**phân loại cảm xúc cả văn bản, hai lớp**, nên lệch với yêu cầu ở ba điểm: không có khía
cạnh, không có Neutral, và không phân biệt "nhắc tới" với "không nhắc tới".

Nhãn *None* là bắt buộc. Nếu không có nó, mô hình buộc phải đoán cảm xúc cho cả những khía
cạnh mà review không hề nhắc tới. Ví dụ, khoảng 79% review không nói gì về âm thanh (mục 2).

## 2. Bộ IMDb hiện tại có dùng được không?

Kiểm tra trên `data/processed/imdb_full_v1/test.jsonl` (25.000 review):

| Tiêu chí | IMDb aclImdb | Kết luận |
|---|---|---|
| Nhãn theo khía cạnh | Không có. Chỉ có `label` 0/1 và `rating` của cả review | ❌ |
| Lớp Neutral | Không có. Bộ dữ liệu bỏ hẳn review 5–6 sao, chỉ giữ ≤4 và ≥7 | ❌ |
| Văn bản có nói về khía cạnh | Từ khoá gợi ý: Acting 55,7%, Story 54,0%, Visual 27,1%, Sound 20,7%; 18% không trúng từ khoá nào | ✅ dùng được làm **nguồn văn bản** |
| Split sạch theo phim | Đã có (21.027 / 3.750 / 25.000, không trùng phim) | ✅ tái sử dụng được |

**Kết luận:** phần **văn bản** và **split** của IMDb dùng được, còn **nhãn** thì không. Nhãn
0/1 hiện có chỉ giữ lại làm nhãn phụ (cảm xúc chung), không thay được nhãn khía cạnh.

## 3. Các dataset có sẵn

| Dataset | Ngôn ngữ, quy mô | Khía cạnh | Nhãn | Vấn đề |
|---|---|---|---|---|
| [Lowerated/imdb-reviews-rated](https://huggingface.co/datasets/Lowerated/imdb-reviews-rated) | Anh, 48.577 review IMDb | Direction, Cinematography, Unique Concept, Story, Emotions, Characters, Production Design | Số thực từ −1 đến 1 | **Không có Sound**; giá trị 0 gộp chung "neutral" và "không nhắc tới"; không ghi rõ ai gán nhãn (thẻ "AI" gợi ý nhãn do mô hình tạo); chỉ có một split train |
| [Lowerated/lm6-movies-reviews-aspects](https://huggingface.co/datasets/Lowerated/lm6-movies-reviews-aspects) | Anh, 345.800 dòng | 7 khía cạnh như trên | Chỉ có tên khía cạnh, **không có cảm xúc** | Không dùng cho ACSA được |
| [MADTRAS](https://data.mendeley.com/datasets/p59cfx4vx6) | Tamil | Acting, story, screenplay, cinematography, songs, BGM, editing… | Có cảm xúc | Không phải tiếng Anh |
| [M-ABSA](https://arxiv.org/abs/2502.11824) (EMNLP 2025) | 21 ngôn ngữ, 7 lĩnh vực | Theo từng lĩnh vực | Bộ ba khía cạnh–cảm xúc, người gán nhãn | Không chuyên về phim, khía cạnh không khớp 4 khía cạnh cần làm |

**Kết luận:** chưa có bộ tiếng Anh công khai nào khớp đúng 4 khía cạnh × 3 cảm xúc + None.
Bộ gần nhất là Lowerated nhưng thiếu Sound và không tách được Neutral với None. Hướng khả thi
và dễ bảo vệ nhất là **tự xây dataset trên văn bản IMDb**, dùng Lowerated làm dữ liệu phụ.

## 4. Đề xuất dataset: `imdb_absa_v1`

### 4.1 Định nghĩa nhãn

| Nhãn | Khi nào dùng | Ví dụ (Acting) |
|---|---|---|
| Positive | Khen rõ ràng | "Hanks gives a brilliant performance" |
| Negative | Chê rõ ràng | "the acting was wooden" |
| Neutral | Có nhắc tới nhưng không khen chê, **hoặc** vừa khen vừa chê ngang nhau | "the cast includes Tom Hanks"; "great lead, terrible support" |
| None | Không nhắc tới khía cạnh | — |

Phạm vi từng khía cạnh:

- **Acting:** diễn viên, vai diễn, diễn xuất, dàn cast, lồng tiếng.
- **Story:** cốt truyện, kịch bản, thoại, nhân vật xét theo cách viết, nhịp kể, kết thúc.
- **Visual:** quay phim, ánh sáng, kỹ xảo/CGI, bối cảnh, trang phục, hoá trang.
- **Sound:** nhạc nền, soundtrack, bài hát, hiệu ứng âm thanh, chất lượng âm thanh.

Hướng dẫn gán nhãn đầy đủ (kèm ca khó) sẽ viết ở `docs/absa_guideline.md` trước khi gán nhãn.

### 4.2 Quy trình gán nhãn

| Phần | Số review | Ai gán nhãn | Dùng để |
|---|---|---|---|
| Gold test | 1.200 (lấy từ test cũ, phân tầng theo độ dài và theo phim) | 3 thành viên, mỗi người 400; 200 review chung để đo độ đồng thuận | Đánh giá cuối, chỉ mở một lần |
| Gold validation | 600 (từ validation cũ) | 3 thành viên | Chọn mô hình, ngưỡng |
| Silver train | khoảng 21.000 (train cũ) | LLM gán nhãn theo đúng guideline, xuất JSON kèm câu bằng chứng | Huấn luyện |

- Đo độ đồng thuận bằng **Cohen/Fleiss κ** trên 200 review chung; đặt mục tiêu κ ≥ 0,6.
- So nhãn LLM với gold validation. Nếu một khía cạnh đạt dưới khoảng 80%, sửa prompt hoặc
  guideline rồi gán lại. Ghi lại model, prompt và ngày chạy để người khác tái lập được.
- Giữ nguyên split theo phim cũ, để kết quả không bị rò rỉ vì review cùng phim nằm ở hai tập.

Không dùng thẳng nhãn LLM làm test. Nếu làm vậy, mô hình chỉ đang học bắt chước LLM, còn
điểm số lại trông như được đo trên nhãn người.

### 4.3 Khía cạnh mở rộng (tuỳ chọn)

| Khía cạnh | Vì sao nên thêm | Mức ưu tiên |
|---|---|---|
| Direction (Đạo diễn) | Review IMDb hay nhắc tới; Lowerated cũng có | Cao |
| Pacing/Editing (Nhịp phim, dựng phim) | "too long", "boring middle" rất phổ biến; hiện đang bị lẫn vào Story | Trung bình |
| Emotion/Entertainment (Cảm xúc, độ giải trí) | Gần với cảm xúc chung nhưng tách được | Trung bình |
| Originality (Tính mới) | Lowerated có (Unique Concept) | Thấp |
| Overall (Tổng thể) | Lấy từ rating sẵn có; 5–6 sao = Neutral nếu bổ sung review thiếu | Nên giữ làm nhãn phụ |

Đề xuất: làm chắc **4 khía cạnh chính trước**, rồi thêm **Direction**. Mỗi khía cạnh thêm sẽ
làm tăng công gán nhãn gold theo tỉ lệ tương ứng.

## 5. Mô hình và cách chạy

### 5.1 Cách đặt bài toán

1. **Multi-head (chính):** dùng một encoder chung, gắn 4 đầu phân loại, mỗi đầu có 4 lớp
   (Pos/Neg/Neu/None). Chỉ cần một lần forward cho mỗi review nên chạy nhanh.
2. **Sentence-pair:** đưa vào `review [SEP] "acting"`, đầu ra 4 lớp (theo kiểu auxiliary
   sentence của Sun và cộng sự, 2019). Cách này dễ thêm khía cạnh mới, nhưng chậm gấp 4 lần.
3. **LLM zero-shot / few-shot:** dùng làm mốc so sánh và để giải thích kết quả.

### 5.2 Các mô hình so sánh

| # | Mô hình | Vai trò |
|---|---|---|
| B0 | Từ khoá + VADER | Mốc tối thiểu |
| B1 | TF-IDF + Logistic Regression cho mỗi khía cạnh | Mốc cổ điển |
| B2 | BERT đóng băng + BiLSTM đầu–cuối, multi-head | Nối tiếp kết quả cũ (91,87%) |
| M1 | BERT-base fine-tune, multi-head | Mô hình chính, nhẹ |
| M2 | RoBERTa-large hoặc DeBERTa-v3-base fine-tune, multi-head | Mô hình mạnh nhất (GPU 8 GB: fp16 + gradient accumulation) |
| M3 | M1 theo kiểu sentence-pair | So sánh cách đặt bài toán |
| L0 | LLM zero-shot | Mốc tham chiếu |

### 5.3 Độ đo

- Với mỗi khía cạnh: **Macro-F1 trên 4 lớp**. Báo cáo thêm hai phần tách riêng:
  - **phát hiện**: F1 của "có nhắc tới hay không";
  - **cảm xúc**: Macro-F1 trên 3 lớp, chỉ tính các review có nhắc tới khía cạnh đó.
- **Điểm chung:** trung bình Macro-F1 của 4 khía cạnh. Chạy 3 seed (42/43/44) như cũ.
- Kiểm định: bootstrap ghép cặp theo phim kèm hiệu chỉnh Holm. Tái sử dụng mã hiện có.

### 5.4 Thứ tự chạy

1. Viết guideline, gán thử 50 review, cả nhóm thống nhất các ca khó.
2. Gán nhãn gold (validation 600, test 1.200) và đo κ.
3. Gán nhãn silver bằng LLM rồi kiểm tra với gold validation.
4. Chạy B0–B2, rồi M1 với 3 seed. Mỗi lượt BERT-base mất khoảng 20–40 phút.
5. Chạy M2, M3, L0.
6. Mở gold test **một lần**, chạy kiểm định, phân tích lỗi theo từng khía cạnh.

## 6. Hướng cải tiến (có cơ sở từ kết quả cũ)

| Cải tiến | Lý do |
|---|---|
| Lấy đầu–cuối khi review dài hơn 512 token | Dự án cũ đã cho thấy cách này tăng 3,6 điểm ở review bị cắt; nhận xét về âm nhạc, hình ảnh hay nằm cuối review |
| Multi-task thêm nhãn cảm xúc chung (0/1 sẵn có) | Có sẵn nhãn miễn phí cho 50.000 review, giúp encoder học chiều cảm xúc |
| Trọng số lớp hoặc focal loss | None chiếm đa số, Neutral rất hiếm |
| Attention theo từng khía cạnh | Mỗi đầu tự chọn câu liên quan, đồng thời cho ra bằng chứng để giải thích |
| Học tiếp MLM trên miền IMDb (50k unsup) | Hướng "tiếp theo" của dự án cũ, nay có chỗ áp dụng |
| Lọc nhãn silver theo độ tin cậy, hoặc huấn luyện trước trên silver rồi fine-tune trên gold validation | Giảm ảnh hưởng của nhiễu từ nhãn LLM |
| Pretrain trên Lowerated (Story↔Story, Cinematography↔Visual, Characters↔Acting) | Có thêm dữ liệu khía cạnh, nhưng phải ghi rõ ánh xạ chỉ gần đúng |
| Ensemble M1 + M2 | Thường tăng thêm một chút với chi phí thấp |

## 7. Ảnh hưởng tới sản phẩm đã làm

- Kết quả hai lớp hiện có (84,97% → 96,74%) **vẫn giữ được**, và trở thành phần
  "cảm xúc tổng thể", tức Giai đoạn 1. Phần ABSA là đóng góp chính mới.
- Cần viết lại: bài báo IEEE (bài toán, dataset, kết quả), slide 21 trang, README.
- Mã tái sử dụng được: split theo phim, cắt đầu–cuối, huấn luyện 3 seed, bootstrap + Holm, demo.
