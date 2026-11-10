# Chạy pipeline ABSA (6 khía cạnh × 4 nhãn)

Đề xuất và lý do: [absa_proposal.md](absa_proposal.md). Tài liệu này chỉ ghi **cách chạy**.

| Khía cạnh huấn luyện | acting, story, visual, sound, direction, overall (pacing vẫn được LLM gán nhưng bỏ khỏi huấn luyện, quyết định 05/11) |
|---|---|
| Nhãn | positive, negative, neutral, none (không nhắc tới) |
| Dữ liệu | văn bản và split của `data/processed/imdb_full_v1` (21.027 / 3.750 / 25.000) |

## 0. Chuẩn bị (một lần)

- llama.cpp bản build b12081 (CUDA 13.4, Windows x64): `llama-b12081-bin-win-cuda-13.4-x64.zip`
  và `cudart-llama-bin-win-cuda-13.4-x64.zip` tại <https://github.com/ggml-org/llama.cpp/releases>,
  giải nén vào `E:\llm\bin\`.
- Model `Qwen3-8B-Q4_K_M.gguf` (5,03 GB, Apache-2.0) tại
  <https://huggingface.co/Qwen/Qwen3-8B-GGUF>, đặt vào `E:\llm\models\`.
- GPU cần trống khoảng 6,5 GB VRAM. Không chạy chung với job huấn luyện khác.

## 1. Gán nhãn silver bằng LLM local

```bat
run\absa-llm-server.cmd      :: cửa sổ 1: mở server ở http://127.0.0.1:8080
run\absa-label.cmd           :: cửa sổ 2: gán nhãn validation -> train -> test
```

- Bản dùng chính thức: prompt `configs/absa_prompt_v3.txt`, cấu hình `configs/absa_labeling_v2.json`.
  Với mỗi khía cạnh, LLM phải trích **câu bằng chứng** trước rồi mới ra nhãn. Nhãn khác none mà
  câu bằng chứng không có trong review (khớp nguyên văn hoặc ít nhất 70% từ) sẽ bị ép về none;
  overall được miễn kiểm tra này. Đầu ra JSON bị ép theo schema nên luôn đủ 7 khoá và đúng nhãn.
- So với 300 review tham chiếu: prompt v2 (không bằng chứng) đạt kappa trung bình 0,52; prompt v3
  đạt 0,72 (`reports/absa_v1/agreement_qwen_v{2,3}.json`). Tốc độ trên RTX 5070 Laptop
  (2 luồng song song): v2 khoảng 1,7 review/s, v3 khoảng 0,64 review/s (toàn bộ khoảng 21 giờ).
- Đầu ra: `data/processed/imdb_absa_v1/silver_qwen3_8b_v3/{split}.jsonl`, có trường `evidence`
  để nhóm kiểm tay. Mỗi dòng ghi lại model,
  phiên bản prompt và mã băm prompt. Review lỗi được ghi vào `errors.jsonl`.
- Có thể dừng (Ctrl+C) và chạy lại bất cứ lúc nào: review đã có nhãn sẽ được bỏ qua.
- Review dài hơn 12.000 ký tự (3 review) được giữ phần đầu và phần cuối.

Chạy thử trước khi gán toàn bộ:

```bat
.venv-train\Scripts\python.exe scripts\absa_label_llm.py --config configs\absa_labeling_v2.json --splits validation ^
  --ids-file reports\absa_v1\reference\sample_ids.json
```

## 2. Kiểm tra chất lượng nhãn

Bộ tham chiếu gồm 300 review validation (150 tích cực, 150 tiêu cực, chọn ngẫu nhiên với seed
2026). Bộ này được gán tay theo đúng `absa_prompt_v2.txt`, lưu tại
`reports/absa_v1/reference/validation_300_claude.jsonl`. Nhãn overall của bộ tham chiếu khớp
99,3% với nhãn sao IMDb (tính trên 95% review có kết luận rõ).

```bat
.venv-train\Scripts\python.exe scripts\absa_compare_labels.py ^
  reports\absa_v1\reference\validation_300_claude.jsonl ^
  data\processed\imdb_absa_v1\silver_qwen3_8b\validation.jsonl ^
  --dataset data\processed\imdb_full_v1\validation.jsonl --out reports\absa_v1\agreement_qwen_v2.json
```

Script in ra tỉ lệ khớp, Cohen's kappa và Macro-F1 cho từng khía cạnh. Nếu kappa của một khía
cạnh dưới 0,6, cần sửa prompt (tạo `absa_prompt_v3.txt`, tăng `prompt_version`) rồi gán lại.

## 3. Huấn luyện 3 mô hình

| Config | Mô hình | Ước tính mỗi seed |
|---|---|---|
| `configs/absa_bilstm_v1.json` | BERT-base đóng băng + BiLSTM đầu–cuối | 20–60 phút |
| `configs/absa_bert_base_v1.json` | BERT-base fine-tune | 20–40 phút |
| `configs/absa_roberta_large_v1.json` | RoBERTa-large fine-tune | 1–1,5 giờ |

```bat
.venv-train\Scripts\python.exe scripts\train_absa.py --config configs\absa_bert_base_v1.json --smoke
run\absa-train.cmd
```

- Tất cả dùng cắt đầu–cuối 512 token (128 đầu + 382 cuối). Mỗi khía cạnh có trọng số lớp
  `sqrt_inverse` để lớp none không lấn át các lớp khác.
- Epoch tốt nhất được chọn theo Macro-F1 trung bình 6 khía cạnh trên validation. Test chỉ chạy
  một lần trên checkpoint đó.
- Checkpoint lưu ở `E:/MovieReview_checkpoints/<version>/` (`latest.pt` của 9 seed đã train để ở `D:/MovieReview_checkpoints`) (ngoài thư mục dự án vì RoBERTa-large nặng khoảng 5 GB mỗi seed). Kết quả lưu ở
  `work_dirs/<version>/seed*/result.json`, tổng hợp ở `reports/absa_v1/<version>/results.json`.
- Muốn tạm dừng thì tạo file `work_dirs/<version>/PAUSE`; runner sẽ dừng ở cuối epoch. Chạy
  lại để tiếp tục.

## 4. Độ đo trong `result.json`

- `macro_f1`: Macro-F1 trên 4 nhãn, cho từng khía cạnh và trung bình 7 khía cạnh (độ đo chính).
- `detection_f1`: F1 của việc nhận ra khía cạnh có được nhắc tới hay không.
- `sentiment_macro_f1`: Macro-F1 trên 3 cảm xúc, chỉ tính các review có nhắc tới khía cạnh đó.
- `overall_vs_imdb_rating`: overall dự đoán so với nhãn sao IMDb. Đây là phép kiểm tra độc lập
  với LLM, vì nhãn sao do chính người viết review cho.

**Lưu ý:** nhãn test là nhãn silver do LLM gán. Khi báo cáo phải ghi rõ điều này. Quy trình kiểm tay
(mục 5) đã chuẩn bị nhưng chưa thực hiện trong thời gian dự án (quyết định 10/11).

## Test

```bat
.venv-train\Scripts\python.exe -m unittest discover -s scripts\tests -v
```

## 5. Kiểm định, phân tích lỗi, kiểm tay

```bat
.venv-train\Scripts\python.exe scripts\absa_significance.py          :: bootstrap theo phim + Holm, khoảng 5 phút
.venv-train\Scripts\python.exe scripts\absa_error_analysis.py        :: F1 từng lớp, kiểu nhầm, độ dài, số sao, ca khó
"%USERPROFILE%\anaconda3\python.exe" scripts\absa_human_check.py prepare   :: tạo 3 file Excel cho nhóm (đã tạo 06/11)
"%USERPROFILE%\anaconda3\python.exe" scripts\absa_human_check.py score     :: chấm điểm khi nhóm điền xong
```

- Kết quả kiểm định: `reports/absa_v1/significance.json`. Kết quả phân tích lỗi:
  `reports/absa_v1/analysis/error_analysis.{json,md}`.
- Kiểm tay: `reports/absa_v1/human_check/` (xem `HUONG_DAN.md`). Có 400 review test, mỗi người 200
  review (100 chung + 100 riêng). Script `prepare` không ghi đè file đã có, trừ khi thêm `--overwrite`.
- `absa_human_check.py` cần openpyxl, có trong anaconda nhưng `.venv-train` không có.
- Bài báo ABSA: `reports/ieee_paper_absa/` (xem README trong đó).

## 6. Demo

```bat
run\absa-demo.cmd                 :: http://127.0.0.1:8766, CPU; thêm --device cuda nếu GPU rảnh
```

Demo nạp best.pt của RoBERTa-large và BERT-base (seed có Macro-F1 validation cao nhất), hiện nhãn và
xác suất cho 6 khía cạnh. Mã: `scripts/absa_demo_server.py`, giao diện: `demo_absa/`.
