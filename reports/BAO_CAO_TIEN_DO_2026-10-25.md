# BÁO CÁO TIẾN ĐỘ DỰ ÁN MOVIEREVIEW

**Ngày tổng hợp:** 25/10/2026  
**Thư mục dự án:** `D:\MovieReview`  
**Nội dung:** Các công việc đã thực hiện, kết quả thực nghiệm, sản phẩm bàn giao và phần chưa hoàn tất.

## 1. Tóm tắt kết quả

Dự án đã xây dựng được quy trình phân loại cảm xúc toàn bộ review phim tiếng Anh trên IMDb, chuẩn bị dữ liệu, huấn luyện và đánh giá ba cấu hình baseline. Cả ba cấu hình đều đã hoàn thành 50 epoch với ba seed 42, 43, 44, tương ứng 9 lượt chạy và 450 epoch được ghi trong lịch sử.

Trong các cấu hình đã hoàn tất, **BERT đóng băng + BiLSTM lấy đầu–cuối** đạt kết quả test tốt nhất theo Macro-F1 trung bình ba seed: **91,87% ± 0,17 điểm phần trăm**. So với BiLSTM chỉ lấy phần đầu, mức tăng là **1,73 điểm phần trăm**.

Giai đoạn tiếp theo đã hoàn thành phân tích validation và demo so sánh. Thí nghiệm đọc nhiều đoạn đã được triển khai nhưng **chưa hoàn tất**: seed42 lưu được 5 epoch, sau đó dừng do lỗi CUDA ngày 16/10/2026. Tại thời điểm kiểm tra ngày 25/10/2026, không tìm thấy tiến trình huấn luyện MovieReview phù hợp; demo ở cổng 8765 cũng không phản hồi.

| Hạng mục | Trạng thái được xác minh |
|---|---|
| Chuẩn bị môi trường, mã nguồn và checkpoint | Đã có trong dự án |
| Chuẩn bị và kiểm tra split IMDb đầy đủ | Hoàn tất |
| Mean pooling, BiLSTM lấy đầu, BiLSTM đầu–cuối | Hoàn tất 9/9 lượt, mỗi lượt 50 epoch |
| Đánh giá test baseline | Hoàn tất trên 25.000 review |
| Phân tích validation và 100 mẫu chẩn đoán | Hoàn tất; ghi chú bởi trợ lý, chưa có người đánh giá độc lập |
| Pilot nhiều đoạn seed42 | Dừng sau 5 epoch được lưu; chưa đủ 50 epoch |
| Nhiều đoạn seed43/44 | Chưa chạy; điều kiện mở rộng chưa được xét cuối pilot |
| Demo so sánh | Đã xây dựng và kiểm thử trước đây; hiện không chạy |
| Chuyển dataset sang tiếng Việt | Chưa thực hiện trong phạm vi hiện tại |

## 2. Mục tiêu và phạm vi

Mục tiêu hiện tại là dự đoán cảm xúc **tích cực hoặc tiêu cực cho cả review**, không phải trích xuất aspect hoặc phân loại cảm xúc từng aspect. Nhãn 0 là tiêu cực, nhãn 1 là tích cực.

Hướng ban đầu có xem xét aspect và một thể loại riêng; phạm vi thực nghiệm sau đó được chuyển sang phân loại toàn review và chạy full IMDb theo quyết định trong quá trình triển khai. Vì vậy, kết quả trong báo cáo này **không phải kết quả riêng cho Horror**. Nếu yêu cầu học phần vẫn bắt buộc một thể loại, cần xác nhận lại phạm vi trước khi dùng các số liệu này làm kết quả nộp chính thức.

Dự án có checkpoint BERT-base-uncased và NLLB-200-distilled-600M. NLLB được chuẩn bị cho hướng Anh–Việt nhưng chưa được dùng để tạo bộ dữ liệu tiếng Việt trong các thí nghiệm báo cáo ở đây. Mã baseline cục bộ dùng BERT đóng băng; không trình bày kết quả này như một phép tái lập hoàn toàn phương pháp hoặc điểm số của bài báo tham khảo.

## 3. Dữ liệu và tiền xử lý

Dữ liệu được tổ chức trong `data/processed/imdb_full_v1`, gồm các file JSONL và manifest để theo dõi nguồn gốc, phân chia và kiểm tra trùng lặp.

| Tập | Số review | Tiêu cực | Tích cực |
|---|---:|---:|---:|
| Train | 21.027 | 10.467 | 10.560 |
| Validation | 3.750 | 1.867 | 1.883 |
| Test chính thức | 25.000 | 12.500 | 12.500 |

Các công việc đã làm:

- Chuẩn hóa nội dung bằng cách bỏ thẻ HTML, giải mã HTML entity và chuẩn hóa khoảng trắng.
- Loại khỏi phần train gốc 98 bản ghi trùng văn bản train, 123 bản ghi trùng văn bản test và 2 bản ghi có phim trùng test.
- Tách train/validation theo nhóm phim, nhằm tránh review của cùng phim nằm ở hai tập.
- Lưu định danh review, phim, nhãn và mã băm để kiểm tra tính toàn vẹn.
- Manifest ghi nhận không có giao nhau về phim hoặc văn bản giữa các split.
- Giữ nguyên test chính thức; tập này vẫn có 201 bản ghi trùng văn bản dư bên trong chính tập test.

Test đã được sử dụng trong giai đoạn baseline và có phần Horror được đánh giá trước đó. Do đó, không gọi test này là tập độc lập chưa từng được xem. Không dùng dữ liệu không nhãn trong các lượt huấn luyện được báo cáo.

Bằng chứng: [manifest dữ liệu](../data/processed/imdb_full_v1/manifest.json).

## 4. Baseline đã xây dựng

Cả ba cấu hình dùng cùng BERT-base-uncased đóng băng, luôn ở eval mode. Chỉ phần phân loại phía sau được huấn luyện.

| Cấu hình | Cách xử lý review |
|---|---|
| Mean pooling | Lấy tối đa 254 token đầu, thêm CLS/SEP; trung bình biểu diễn token rồi phân loại |
| BiLSTM lấy đầu | Cùng giới hạn đầu vào; BiLSTM đọc chuỗi biểu diễn BERT rồi phân loại |
| BiLSTM đầu–cuối | Review dài: giữ 127 token đầu và 127 token cuối, thêm CLS/SEP; dùng BiLSTM |

Review không vượt giới hạn được giữ nguyên. Mỗi đầu vào tối đa 256 token tính cả token đặc biệt. BiLSTM có hidden size 768 mỗi chiều; dropout 0,5. Encoder dùng BF16 trên GPU, head dùng FP32.

Cấu hình chung: AdamW, learning rate 0,0001; weight decay 0,01; batch size 8 và gradient accumulation 4, tương đương 32 review mỗi lần cập nhật thông thường. Mỗi seed chạy 50 epoch, tắt early stopping. Các trường patience còn trong config không có nghĩa early stopping đang bật.

Checkpoint tốt nhất được chọn theo Macro-F1 validation, **không mặc định lấy epoch 50** và không chọn theo test. Seed42/43/44 giúp quan sát biến thiên do khởi tạo và thứ tự huấn luyện; không phải ba dataset khác nhau.

Bằng chứng: [config baseline](../configs/full_imdb.json), lịch sử và checkpoint trong `work_dirs/imdb_full_v1/`.

## 5. Kết quả baseline trên test

Macro-F1 là trung bình F1 của hai lớp, phản ánh đồng thời khả năng dự đoán lớp tích cực và tiêu cực. Bảng dưới dùng mean ± độ lệch chuẩn mẫu qua ba seed; đây không phải khoảng tin cậy.

| Mô hình | Accuracy trung bình | Macro-F1 trung bình ± SD |
|---|---:|---:|
| Mean pooling | 85,7133% | 85,7118% ± 0,0407 điểm % |
| BiLSTM lấy đầu | 90,1387% | 90,1384% ± 0,1277 điểm % |
| BiLSTM đầu–cuối | **91,8733%** | **91,8731% ± 0,1659 điểm %** |

Chi tiết từng lượt, tất cả đều có 50 epoch trong history:

| Mô hình | Seed | Epoch được chọn bằng validation | Macro-F1 test |
|---|---:|---:|---:|
| Mean pooling | 42 | 48 | 85,7136% |
| Mean pooling | 43 | 49 | 85,7516% |
| Mean pooling | 44 | 40 | 85,6702% |
| BiLSTM lấy đầu | 42 | 48 | 90,0233% |
| BiLSTM lấy đầu | 43 | 41 | 90,1160% |
| BiLSTM lấy đầu | 44 | 9 | 90,2758% |
| BiLSTM đầu–cuối | 42 | 49 | 91,7916% |
| BiLSTM đầu–cuối | 43 | 39 | 91,7638% |
| BiLSTM đầu–cuối | 44 | 41 | 92,0640% |

Kết quả cho thấy đầu–cuối tốt hơn lấy đầu ở cả ba seed trên test hiện tại. Trung bình mức tăng là 1,7348 điểm phần trăm Macro-F1. Đây là kết quả thực nghiệm cục bộ; chưa có kiểm định để khẳng định ý nghĩa thống kê hoặc khả năng tổng quát sang review tiếng Việt.

Bằng chứng: [JSON kết quả đầy đủ](full_imdb_results.json).

## 6. Phân tích lỗi validation

Đã đánh giá đủ 3.750 review validation và tạo mẫu chẩn đoán 100 trường hợp. Hai checkpoint dùng cho demo được chọn riêng theo validation: BiLSTM lấy đầu seed43/epoch41 và đầu–cuối seed42/epoch49.

| Nhóm validation | Số review | F1 lấy đầu | F1 đầu–cuối |
|---|---:|---:|---:|
| Toàn bộ | 3.750 | 91,04% | 91,73% |
| Ngắn | 2.182 | 92,58% | 92,71% |
| Dài, trên 256 token | 1.568 | 88,90% | 90,37% |

Trong 3.750 review, có 3.270 trường hợp cả hai đúng, 166 cả hai sai, 170 trường hợp đầu–cuối đúng trong khi lấy đầu sai, và 144 trường hợp ngược lại.

Mẫu chẩn đoán gồm 40 trường hợp cả hai sai, 20 đầu–cuối sửa đúng, 20 đầu–cuối làm sai và 20 cả hai đúng. Đã đọc toàn văn, ghi chú từng trường hợp và kiểm tra văn bản/nhãn vẫn trùng với dữ liệu gốc.

Các hiện tượng đáng chú ý gồm khen chê lẫn nhau, khen phim khác trong cùng review, phủ định hoặc lời trích dẫn, thích phim dở vì tính giải trí, và từ ngữ kinh dị không phản ánh thái độ tiêu cực của người xem. Case54 là nội dung hướng dẫn đăng bài/xác thực IMDb thay vì một review; một số trường hợp khác có văn bản và nhãn cần con người rà soát.

Không tự sửa nhãn hoặc xóa mẫu. Ghi chú được tạo bởi trợ lý, chưa có người đánh giá độc lập. Mẫu 100 trường hợp không đại diện thống kê cho dataset. Hai checkpoint khác seed/epoch, nên khác biệt dự đoán trên từng trường hợp không chứng minh nguyên nhân là cách cắt văn bản.

Bằng chứng: [tổng hợp JSON](validation_analysis/summary.json), [100 mẫu có ghi chú](validation_analysis/cases_100_reviewed.json), [bản đọc dễ xem](validation_analysis/index.html).

## 7. Thí nghiệm cải tiến đọc nhiều đoạn

Đã triển khai mô hình chia toàn bộ review thành các đoạn liên tiếp, mỗi đoạn tối đa 254 token nội dung. Mỗi đoạn đi qua BERT đóng băng và BiLSTM dùng chung; lấy trung bình biểu diễn các đoạn để phân loại một nhãn cho cả review. Không gán nhãn riêng cho từng đoạn.

Train tạo 33.962 đoạn, tối đa 10 đoạn/review; validation tạo 6.173 đoạn, tối đa 12 đoạn/review. Kiểm tra token coverage xác nhận không mất token nội dung. Cấu hình batch 2 review, tích lũy 16 lần, microbatch encoder 4 đoạn và activation checkpointing ở head.

Các kiểm thử ban đầu đã đạt: tương đương baseline khi chỉ có một đoạn; gộp đúng theo review; backward trên review train dài nhất; encoder không nhận gradient; lưu/khôi phục dự đoán và cập nhật tiếp sau khôi phục optimizer/RNG. Các kiểm thử này không bảo đảm toàn bộ lượt dài sẽ ổn định.

**Tiến độ thực tế:** seed42 đã lưu 5/50 epoch. Sau đó tiến trình báo `CUDA error: an illegal memory access was encountered` lúc 23:42:31 ngày 16/10/2026, múi giờ +07:00. Theo vòng lặp và history, lỗi xuất hiện trong lượt huấn luyện tiếp theo sau epoch5. CUDA báo lỗi bất đồng bộ nên vị trí traceback chưa xác định được nguyên nhân gốc. Không kết luận đây là hết VRAM.

| Epoch đã lưu | Train loss | Macro-F1 validation |
|---|---:|---:|
| 1 | 0,3695 | 84,4209% |
| 2 | 0,2914 | 87,4393% |
| 3 | 0,2807 | 86,2563% |
| 4 | 0,2551 | 87,2209% |
| 5 | 0,2363 | **90,6354%** |

Best trong phần đã chạy là epoch5. F1 nhóm review dài đạt 90,6869%. So với head-tail seed42 trên cùng validation, F1 tổng thấp hơn khoảng 1,10 điểm phần trăm, nhưng nhóm dài cao hơn khoảng 0,32 điểm. Đây chỉ là so sánh tạm thời sau 5 epoch với baseline đã chạy 50 epoch, không đủ để kết luận hướng nhiều đoạn thất bại hoặc tốt hơn.

Quy tắc đã chốt: sau khi hoàn thành 50 epoch seed42, chỉ mở rộng seed43/44 nếu F1 validation tổng tăng ít nhất 0,5 điểm phần trăm so với head-tail seed42 và nhóm dài không giảm. Pilot chưa hoàn thành nên chưa có quyết định cuối theo quy tắc này; chưa có test của mô hình nhiều đoạn và chưa triển khai nó trong demo.

Các file `best.pt`, `latest.pt`, `history.json` và dự đoán validation tốt nhất còn trên đĩa. Lượt kiểm tra báo cáo này chỉ xác nhận file và history, không khởi động lại training hoặc kiểm tra lại khả năng resume sau sự cố CUDA.

Bằng chứng: [trạng thái lỗi](../work_dirs/multichunk_v1/status_live.json), [history seed42](../work_dirs/multichunk_v1/seed42/history.json), [smoke test](../work_dirs/multichunk_v1/smoke_results.json).

## 8. Demo và công cụ bàn giao

Đã xây dựng demo chạy cục bộ bằng CPU, dùng checkpoint thật của hai mô hình. Người dùng nhập review tiếng Anh để xem nhãn, xác suất chưa hiệu chỉnh, số token giữ/bỏ và phần văn bản đầu–cuối đã đưa vào mô hình. Demo có ví dụ minh họa và đọc trạng thái thí nghiệm từ JSON.

Kiểm thử API trước đây đã đạt với ví dụ tích cực, tiêu cực, review dài và HTML; kiểm tra từ chối nội dung rỗng/quá dài, payload không hợp lệ, nguồn truy cập ngoài và đường dẫn file tùy ý. Đã kiểm tra thao tác dự đoán và mở văn bản đầu–cuối trên trình duyệt; ảnh kiểm chứng được lưu.

**Ngày 25/10/2026, demo không chạy:** kết nối `127.0.0.1:8765/health` bị từ chối. Điều này khác với việc mã demo chưa hoàn thành. Có thể mở lại bằng `D:\MovieReview\run-demo.cmd` khi cần sử dụng; báo cáo này không tự mở lại dịch vụ.

Bằng chứng: [kiểm thử API](demo_checks.json), [ảnh demo trước đây](demo-preview.png), [hướng dẫn vận hành](../PHASE2_README.md). Các mô tả “đang chạy” trong README/JSON bàn giao cũ là trạng thái ngày triển khai, không phải trạng thái hiện tại.

## 9. Việc còn lại và thứ tự ưu tiên

1. Điều tra lỗi CUDA bằng một lượt tái hiện có kiểm soát, giữ nguyên checkpoint và giao thức; chưa quy lỗi cho tensor padding chỉ dựa vào traceback.
2. Sau khi xác minh cách khắc phục, kiểm tra resume từ epoch5 rồi hoàn thành ngân sách 50 epoch seed42. Nếu thay đổi phương pháp/cấu hình, ghi thành phiên bản thí nghiệm mới để giữ khả năng đối chiếu.
3. Áp dụng điều kiện mở rộng đã chốt; chỉ chạy seed43/44 khi đáp ứng điều kiện sau pilot đầy đủ.
4. Cố định lựa chọn dựa trên validation trước khi đánh giá test mô hình cuối; công bố cả kết quả không cải thiện nếu có.
5. Nhờ thành viên nhóm rà soát độc lập mẫu lỗi và trường hợp nghi ngờ dữ liệu. Không sửa nhãn chỉ để tăng điểm.
6. Mở demo khi cần trình bày; chỉ thêm mô hình nhiều đoạn sau khi đủ bằng chứng đánh giá.

Báo cáo được lập từ file cấu hình, manifest, history, kết quả JSON, log lỗi và kiểm tra tiến trình/dịch vụ hiện tại. Các bước trên là công việc còn lại, chưa được thực hiện trong lượt tạo báo cáo này.