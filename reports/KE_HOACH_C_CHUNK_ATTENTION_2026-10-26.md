# KẾ HOẠCH TRIỂN KHAI PHƯƠNG ÁN C
## Phân loại cảm xúc IMDb bằng BERT đóng băng, BiLSTM và attention giữa các đoạn

Ngày lập: 26/10/2026. Trạng thái: kế hoạch chuẩn bị triển khai, chưa sửa mã huấn luyện hoặc khởi chạy thí nghiệm trong lượt lập kế hoạch này.

## 1. Mục tiêu và câu hỏi nghiên cứu

Câu hỏi chính: khi đã giữ toàn bộ nội dung review, việc học trọng số cho từng đoạn có tốt hơn coi mọi đoạn quan trọng ngang nhau không?

Giả thuyết: review thường chứa nhiều câu kể chuyện hoặc thông tin ngoài lề; một số đoạn chứa đánh giá quyết định. Attention có thể học cách tăng ảnh hưởng của các đoạn này. Đây là giả thuyết phải kiểm chứng, không phải kết luận đã có.

Mục tiêu sản phẩm: một mô hình mới có thể đánh giá tái lập, so sánh công bằng với mô hình gộp trung bình, và trả về trọng số đoạn cho demo. Bài toán vẫn là phân loại toàn review thành tích cực/tiêu cực; không tạo nhãn aspect hoặc nhãn cảm xúc từng đoạn.

Phạm vi giữ nguyên: IMDb tiếng Anh, train21.027/validation3.750/test25.000, cùng manifest và nhãn. Kế hoạch này giả định tiếp tục full IMDb theo phạm vi đã triển khai. Nếu học phần quay lại yêu cầu một thể loại, phải chốt split và chạy lại cả B/C; không trộn kết quả hai phạm vi.

Không đưa vào C v1: fine-tune BERT, dịch tiếng Việt, đổi encoder, sentence-based chunking, overlap, positional embedding, GRU giữa các đoạn, gán lại nhãn hoặc ensemble. Mỗi thay đổi thêm sẽ gây khó xác định attention có thực sự đóng góp hay không.

## 2. Điểm xuất phát đã đối chiếu với mã nguồn

- `scripts/phase2_core.py`: `ChunkHead.represent()` tạo vector 1536 chiều cho một đoạn; `ChunkHead.classify()` hiện gộp bằng tổng/chia số đoạn.
- `forward_chunks()` giữ danh sách owner để biết đoạn nào thuộc review nào; BERT xử lý theo microbatch, còn loss tính theo review.
- BiLSTM768 mỗi chiều, dropout0,5; BERT đóng băng và eval mode, encoder CUDA BF16, head FP32.
- `configs/multichunk_pilot.json`: review batch2, accumulation16, chunk microbatch4, AdamW1e-4, weight_decay0,01, 50 epoch, không early stopping.
- Lượt nhiều đoạn cũ lưu được 5 epoch rồi báo illegal memory access. Không dùng kết quả 5 epoch để thay cho đối chứng 50 epoch.
- Runner cũ có signature gồm mã nguồn/config/dữ liệu, điều kiện mở rộng seed và tự ghi cấu hình deploy. Không chỉnh trực tiếp để tránh làm mất khả năng truy vết hoặc triển khai ngoài ý muốn.

## 3. Kiến trúc C v1

Luồng xử lý:

Review → tokenizer BERT → các đoạn 254 token nội dung → thêm CLS/SEP → frozen BERT → shared BiLSTM → vector từng đoạn → masked additive attention → vector review → dropout → Linear(1536,2) → logits.

Với review có m đoạn và vector h_i thuộc R^1536:

    u_i = tanh(W h_i + b)          W: 128 × 1536; b: 128
    e_i = vᵀ u_i                  v: 128; không thêm scalar bias dư thừa
    alpha_i = softmax(e)_i        chuẩn hóa riêng trên m đoạn hợp lệ của review
    z = sum_i(alpha_i * h_i)      z thuộc R^1536
    logits = Linear(Dropout(z))   2 lớp

Attention hidden size chốt ban đầu: 128. Số tham số thêm là 1536×128 + 128 + 128 = 196.864. Không thêm attention dropout hoặc temperature ở v1 để hạn chế số lựa chọn cần dò.

Quy tắc quan trọng:

1. Softmax phải tính riêng từng review, không tính chung các đoạn trong batch.
2. Đoạn padding bị mask khỏi softmax; không được nhận trọng số.
3. Dùng FP32 cho score, softmax và phép gộp; mọi trọng số phải hữu hạn, không âm và tổng gần1.
4. Review chỉ có một đoạn phải cho alpha=[1] và tương đương gộp trung bình khi phần trọng số dùng chung giống nhau ở eval mode.
5. Dùng cross-entropy một nhãn/review. Không nhân số mẫu theo số đoạn hoặc ưu tiên review dài chỉ vì có nhiều đoạn.
6. Giữ nguyên mọi token nội dung; không cap số đoạn hoặc cắt review dài âm thầm.
7. Trả attention theo đúng thứ tự đoạn gốc để demo đối chiếu; chỉ detach khi lưu/hiển thị, không detach khỏi đồ thị gradient lúc train.
8. C v1 không mô hình hóa thứ tự giữa các đoạn: đổi thứ tự các vector đoạn chỉ đổi thứ tự trọng số, không đổi vector gộp. BiLSTM vẫn mô hình hóa thứ tự token bên trong mỗi đoạn. Đây là giới hạn được ghi rõ.

Cơ sở phương pháp: ý tưởng attention phân cấp cho phân loại tài liệu trong HAN, https://aclanthology.org/N16-1174/. Ở đây là điều chỉnh attention cấp đoạn trên kiến trúc BERT–BiLSTM hiện có, không phải tái lập nguyên bản HAN hoặc tuyên bố phát minh attention.

## 4. Thiết kế đối chứng và khởi tạo

| Mã | Cấu hình | Vai trò |
|---|---|---|
| A | BiLSTM đầu–cuối đã hoàn thành | Mốc hiệu quả thực tế của dự án |
| B2 | Toàn review, gộp trung bình | Đối chứng trực tiếp trong runner mới |
| C1 | Toàn review, gộp attention128 | Phương án cải tiến |

B2 và C1 dùng chung pipeline trong runner mới, chỉ khác aggregation. Đổi tên B2 để phân biệt với lượt B cũ dừng sau5 epoch.

Mỗi seed, tạo một bản khởi tạo chung cho BiLSTM và classifier; lưu/hash state này rồi nạp giống nhau vào B2/C1. Attention dùng nguồn RNG khởi tạo riêng. Dùng generator riêng cho thứ tự review để việc khởi tạo thêm attention không vô tình đổi thứ tự batch. Seed_all giống nhau đơn thuần chưa bảo đảm điều đó.

Cả B2 và C1 bắt đầu từ đầu head; không dùng best head-tail hoặc checkpoint B epoch5 để khởi tạo riêng C1. Thống nhất dropout, optimizer, gradient clipping, batch, precision, microbatch, lịch validation và thuật toán chọn best. Độ lệch loss sau train là điều bình thường; không kỳ vọng hai model có trọng số chung còn giống nhau sau cập nhật.

Chạy tuần tự B2_seed42 rồi C1_seed42 để giảm tranh GPU. Không chạy song song với SEG trong thử nghiệm đo tài nguyên; không tự dừng tác vụ khác. Nếu GPU đang bận, ghi nhận và hoãn launch thay vì thay cấu hình một phía.

Mỗi lượt50 epoch, không early stopping; best theo Macro-F1 validation, tie chọn epoch sớm hơn. Giữ ngưỡng phân lớp argmax, không tune threshold. Test không được đọc trong development/pilot.

## 5. Kế hoạch theo giai đoạn và điều kiện hoàn thành

### Giai đoạn 0 — bảo toàn bằng chứng, điều tra CUDA

Đầu ra: `reports/chunk_attention_v1/cuda_diagnosis.json` và log tái hiện.

- Ghi phiên bản Python/PyTorch/CUDA/driver/GPU, tiến trình đang dùng GPU, VRAM đầu/cuối, checksum dữ liệu/config/mã/checkpoint. Chụp lại trạng thái lỗi cũ.
- Giữ nguyên `work_dirs/multichunk_v1` và scripts cũ; diagnostic chạy trong thư mục riêng, không ghi đè history/latest/best.
- Dùng tiến trình mới sau illegal memory access; không tiếp tục sử dụng CUDA context đã lỗi.
- Tái hiện từ checkpoint sauepoch5 trên bản sao chẩn đoán, bật CUDA_LAUNCH_BLOCKING=1 chỉ cho tiến trình thử. Ghi epoch/batch/review_id/chunk counts và kích thước tensor quanh bước thất bại; không cần log toàn văn review.
- Kiểm tra token ID trong khoảng vocab, dtype long, độ dài mask hợp lệ, owner ID, tensor/device và finite loss/gradient.
- Nếu cần cô lập nguyên nhân, thay từng yếu tố: activation checkpointing, microbatch, BF16/FP32, backend attention. Không đồng thời thay nhiều biến rồi coi là đã tìm nguyên nhân.
- Nếu phải đổi hành vi số học/batching, ghi phiên bản vận hành mới và áp dụng giống nhau cho B2/C1. Không hạ/nâng hàng loạt dependency để che lỗi.

Điều kiện chuyển bước: tái hiện vùng lỗi hoặc có cấu hình vận hành vượt qua cùng vùng batch đã lỗi và stress test, với nguyên nhân/giới hạn được ghi nhận trung thực. Nếu chưa biết nguyên nhân, ghi “workaround đã kiểm tra”, không ghi “đã sửa triệt để”. Không bảo đảm hết lỗi chỉ vì smoke vài bước chạy được.

### Giai đoạn 1 — triển khai head và runner B2/C1

Đầu ra dự kiến:

    scripts/chunk_attention_model.py
    scripts/train_chunk_compare.py
    scripts/evaluate_chunk_compare.py
    scripts/tests/test_chunk_attention.py
    configs/chunk_compare_v1.json
    work_dirs/chunk_compare_v1/mean_seed42/
    work_dirs/chunk_compare_v1/attention_seed42/
    reports/chunk_attention_v1/

Đây là tên dự kiến, chưa phải các file đã tồn tại.

- `chunk_attention_model.py`: phần biểu diễn đoạn dùng chung, hai phép gộp mean/attention, API trả logits và tùy chọn weights/owner IDs.
- `train_chunk_compare.py`: tokenization/split dùng chung; riêng generator batch và initialization; train, validation, pause, checkpoint, resume và status.
- `evaluate_chunk_compare.py`: đánh giá checkpoint đã chốt, xuất dự đoán và bảng đối chiếu; bắt buộc chỉ rõ split, không mặc định test.
- Không kế thừa cơ chế tự deploy/gate của runner cũ. Promotion là bước riêng sau khi báo cáo đủ kết quả.
- Config lưu aggregation, attention_dim128, toàn bộ thông số baseline; ghi mode=planned trước triển khai, rồi đóng băng config/signature khi bắt đầu lượt chính.

Điều kiện hoàn thành: B2/C1 dùng cùng train loop, review partition, trọng số khởi tạo chung, tham số vận hành; thay aggregation là khác biệt kiến trúc duy nhất.

### Giai đoạn 2 — kiểm thử tính đúng và ổn định

Các kiểm tra bắt buộc:

| Kiểm tra | Điều cần chứng minh |
|---|---|
| Single chunk | Mean và attention cho cùng logits ở eval với weights chung |
| Uniform scores | Khi mọi score=0, weighted pooling bằng mean |
| Mask padding | Thêm đoạn padding không làm đổi output; alpha của padding=0 |
| Review isolation | Gộp một review không dùng đoạn của review khác |
| Permutation | Đổi thứ tự đoạn không đổi pooled vector, weights đổi theo tương ứng |
| Gradient | Attention/BiLSTM/classifier nhận gradient; BERT không nhận gradient |
| Microbatch | Chia microbatch khác nhau cho output eval gần nhau trong dung sai công bố |
| Token coverage | Ghép nội dung các đoạn tái tạo đúng token gốc |
| Long reviews | Chạy train backward trên review train dài nhất; eval trên validation dài nhất |
| Resume | Lưu/khôi phục model, optimizer, RNG, thứ tự batch; bước tiếp theo khớp trong dung sai |
| Accumulation | Nhóm batch cuối không đủ32 review vẫn chuẩn hóa loss đúng |
| Status/lock | Mở viewer không làm chết job; runner thứ hai không ghi đè trạng thái runner đang sống |

Dùng dữ liệu train cho thử overfit tập nhỏ để phát hiện lỗi gradient; không biến độ chính xác trên tập nhỏ thành kết quả benchmark. Dung sai CPU FP32 và CUDA BF16 phải được nêu riêng; kiểm tra deterministic trong phạm vi backend cho phép, không hứa bitwise mọi phần cứng.

Stress test có backward, optimizer và save/resume trên nhiều batch với độ dài đa dạng; ưu tiên đi qua vùng gây lỗi cũ. Pilot epoch đầu cần quan sát đến khi validation và checkpoint được ghi thành công. Các epoch smoke dùng thư mục riêng, không cộng vào50 epoch chính.

### Giai đoạn 3 — pilot seed42

Ngân sách đề xuất: B2 50epoch + C1 50epoch, tối đa100 epoch chính cho pilot. Đây là ngân sách kế hoạch, chưa phải lệnh khởi chạy.

Thông số chốt: frozen BERT, BiLSTM768×2, dropout0,5, AdamW1e-4, weight_decay0,01, clip_grad_norm1, batch2, accumulation16, chunk microbatch4, content254, encoderBF16/headFP32. Nếu bước CUDA yêu cầu thay thông số, chốt lại trước cả hai lượt và ghi deviation.

Mỗi epoch lưu train loss, validation loss/Accuracy/Macro-F1, confusion matrix, F1 ngắn/dài, thời gian, peak allocated/reserved VRAM, seed và checkpoint signature. Theo dõi loss để thấy overfitting nhưng vẫn chạy đủ ngân sách trừ lỗi tài nguyên hoặc yêu cầu dừng.

Với attention, thêm số đoạn, normalized entropy cho review nhiều hơn1 đoạn và phân bố trọng số. Phân bố lệch mạnh không tự chứng minh attention tốt; entropy một đoạn phải được xử lý riêng, tránh chia log(1).

Checkpoint: latest có optimizer/RNG/history/current epoch; best có validation và provenance. Lưu theo atomic replace có retry hữu hạn; file status khóa không được làm hỏng checkpoint. Khi job fail ghi error.json, failed stage và last_completed_epoch. Không tự restart vô hạn. Pause sauepoch giữ nguyên trạng thái; resume tiếp từ epoch hoàn tất cuối.

### Giai đoạn 4 — điều kiện mở rộng và đánh giá

Tách hai mục tiêu để tránh kết luận nhầm:

- Đóng góp attention: C1 phải tốt hơn B2 trên cùng giao thức.
- Thay thế mô hình đang dùng: C1 còn cần cạnh tranh được với A đầu–cuối.

Gate pilot đề xuất, cố định trước launch:

1. F1 validation tổng của C1_seed42 − B2_seed42 >=0,003, tức +0,3 điểm phần trăm.
2. F1 validation nhóm dài của C1_seed42 không thấp hơn B2_seed42.
3. Đủ50 epoch, không có lỗi tính đúng hoặc dữ liệu bị thay đổi.

Nếu đạt, đề xuất mở rộng C1 và B2 cùng seed43/44: thêm4 lượt×50=200 epoch, tối đa6 lượt/300 epoch chính toàn chiến dịch. Nếu không đạt, dừng mở rộng và ghi kết quả âm; không lặp nhiều hidden_size/lr đến khi có điểm đẹp.

Gate này là đề xuất riêng để đánh giá attention, KHÔNG sửa ngầm gate cũ của `multichunk_pilot.json`. Ngưỡng0,3 điểm là tiêu chí thực dụng, không phải bảo đảm ý nghĩa thống kê. Nếu chỉ C vượt B nhưng chưa vượt A, vẫn có thể báo attention cải thiện pooling trong pipeline nhiều đoạn; chưa đủ cơ sở thay demo.

Đối với mục tiêu thay A, giữ mốc pilot đã trao đổi: seed42 F1 tổng ít nhất0,9223321811 (92,2332%) và long F1 không dưới0,9036980004 (90,3698%). Sau3seed xem cả mean±sample SD và chênh lệch cùng seed; không chọn seed tốt nhất theo test để quảng bá mô hình.

Chốt phương pháp/checkpoint/threshold trên validation trước khi chạy test. Khi kết thúc chiến dịch, đánh giá các cấu hình B2/C1 được đưa vào so sánh cuối cùng trên test, kể cả kết quả không cải thiện. Không chỉ báo cáo cấu hình thắng; nếu chỉ có pilot seed42 phải ghi rõ chưa xác minh độ ổn định qua seed.

Metric chính: Macro-F1 toàn review. Metric phụ: Accuracy, F1 lớp0/1, confusion matrix, F1 dài/ngắn, theo số đoạn (1;2;3–4;>=5, ghi số mẫu), thời gian suy luận và peak VRAM. Nếu cần khoảng tin cậy của chênh lệch, bootstrap theo movie_id trên dự đoán ghép cặp, không coi nhiều review cùng phim hoàn toàn độc lập; báo rõ đó là biến thiên theo mẫu phim, khác với SD qua seed.

### Giai đoạn 5 — phân tích và demo

- Chọn mẫu phân tích định tính từ validation, gồm C đúng/B sai, C sai/B đúng, cả hai sai và đúng.
- Hiển thị review, nhãn gốc, dự đoán B/C, alpha và nguyên văn từng đoạn; không biến điểm attention thành “độ tích cực của đoạn”.
- Có thể làm kiểm tra loại bỏ đoạn alpha lớn và đoạn alpha nhỏ để mô tả độ nhạy; vì đây là input bị can thiệp, không coi là chứng minh nhân quả hoặc chất lượng giải thích tuyệt đối.
- Demo đọc cấu hình model type rõ ràng để nạp đúng head. Không dùng loader mean hiện tại để nạp attention checkpoint.
- CPU inference, giới hạn đầu vào và xử lý lỗi giữ như demo cũ; kiểm tra predictions/attention sau save-load trên CPU.
- Chỉ thêm C vào demo sau khi đánh giá xong và ghi rõ phiên bản; không tự thay checkpoint đang dùng bằng best tạm thời.

## 6. Chi phí và lịch dự kiến

Khối lượng công việc kỹ thuật dự kiến: chẩn đoán CUDA0,5–2 ngày; triển khai và test1–2 ngày; phân tích/demo0,5–1 ngày. Đây là ước lượng công việc, không phải cam kết thời điểm hoàn tất; lỗi CUDA chưa rõ nguyên nhân là yếu tố có thể kéo dài.

Chưa chốt số giờ GPU. Đo2 epoch ổn định của B2/C1 gồm validation và lưu checkpoint, báo ETA theo median/range quan sát; không lấy tốc độ của lượt cũ đang lỗi hoặc lúc tranhGPU làm cam kết. Không cần tải model mới. Chi phí chính là đọc mọi đoạn, không phải riêng196.864 tham số attention thêm.

Không cache toàn bộ hidden state BERT ở v1: tuy encoder đóng băng cho phép tối ưu này, nó thêm chi phí đĩa và một thay đổi pipeline cần kiểm chứng riêng. Tối ưu sau khi có kết quả đúng.

## 7. Sản phẩm và tiêu chí nghiệm thu

Đầu ra tối thiểu trước chạy dài:

- Config và protocol đóng băng, split hashes, shared-init hashes, environment snapshot.
- Hai head mean/attention trong cùng runner; test report đạt và log kiểm tra CUDA.
- Cơ chế checkpoint/resume/pause/status hoạt động, đường dẫn output riêng.

Đầu ra sau pilot:

- History, latest/best, validation predictions cho B2/C1 seed42.
- `pilot_comparison.json` ghi delta tổng/dài, gate, reason, test_evaluated=false.
- Báo cáo quyết định mở rộng hoặc dừng, không tuyên bố trước kết quả.

Đầu ra cuối chiến dịch:

- Bảng từng seed và mean±SD; đường cong train/validation; metric ngắn/dài; chi phí vận hành.
- Test predictions/results khi đã khóa phương pháp, phân tích mẫu lỗi và giới hạn nghiên cứu.
- Demo có attention nếu được chọn đưa vào trình bày, kèm nhắc rõ trọng số không phải lời giải thích bảo đảm.

Nghiệm thu triển khai không phụ thuộc việc tăng điểm: mã đúng, dữ liệu/split không đổi, chạy có provenance, kết quả đầy đủ và báo cáo trung thực đều bắt buộc. Thành công về hiệu quả chỉ được kết luận sau thí nghiệm.

## 8. Thứ tự triển khai đề xuất

1. Bảo toàn lượt cũ, chẩn đoán CUDA và chốt cấu hình ổn định.
2. Tạo runner đối chứng cùng head attention, kiểm thử trước.
3. Chốt protocol seed42 và ngân sách100 epoch pilot.
4. Chạy B2/C1 seed42, tổng hợp validation và quyết định gate.
5. Khi đủ cơ sở mới mở rộng seed, đánh giá cuối và cập nhật demo.

Lượt hiện tại chỉ lập kế hoạch. Các file code/config mang tên dự kiến chưa được tạo và không có lượt GPU mới được khởi động.