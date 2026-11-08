# Nguồn gốc và license của thành phần bên thứ ba

Tài liệu này liệt kê mọi mã nguồn, dữ liệu, mô hình và thư viện của bên thứ ba mà dự án MovieReview sử dụng, kèm điều kiện sử dụng của từng thành phần. Các thành phần này **thuộc về tác giả gốc**. Dự án không cấp lại quyền cho chúng; người dùng phải tuân theo license gốc của từng thành phần.

Kiểm tra lần cuối: 01/11/2026.

## Tóm tắt

| Thành phần | Chủ sở hữu | License / điều kiện | Có trong repo? |
|---|---|---|---|
| Bài báo BERT+BiLSTM (arXiv 2502.20682) | Nkhata, Gauch, Anjum, Zhan | CC BY-NC-ND 4.0 (bản arXiv) | Không, chỉ trích dẫn |
| Mã tác giả `gnkhata1/Finetuning-BERT-on-Movie-Reviews-Sentiment-Analysis` | Gibson Nkhata | **Không có file license** (mặc định giữ mọi quyền) | Không, chỉ hướng dẫn tải |
| Large Movie Review Dataset (IMDb, aclImdb v1) | Maas et al., Stanford | Không ghi license; tác giả yêu cầu trích dẫn | Không; chỉ có thống kê, dự đoán và 100 review mẫu để phân tích lỗi |
| IMDb Non-Commercial Datasets (`title.basics.tsv.gz`) | IMDb.com, Inc. | Chỉ dùng cho mục đích cá nhân, phi thương mại | Không |
| `google-bert/bert-base-uncased` | Google | Apache-2.0 | Không |
| `FacebookAI/roberta-large` | Meta AI (Facebook AI) | MIT | Không |
| `facebook/nllb-200-distilled-600M` | Meta AI | CC BY-NC 4.0 | Không |
| Thư viện Python/Node | xem mục 5 | BSD / Apache-2.0 / MIT / PSF | Không, cài qua pip/npm |

---

## 1. Bài báo gốc

- **Tiêu đề:** Fine-tuning BERT with Bidirectional LSTM for Fine-grained Movie Reviews Sentiment Analysis
- **Tác giả:** Gibson Nkhata, Susan Gauch, Usman Anjum, Justin Zhan
- **Công bố:** International Journal On Advances in Systems and Measurements, vol. 16, no. 3 & 4, 2023; bản arXiv: https://arxiv.org/abs/2502.20682 (28/02/2025)
- **License bản arXiv:** Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 (https://creativecommons.org/licenses/by-nc-nd/4.0/)
- **Cách dự án sử dụng:** chỉ trích dẫn và tái lập phương pháp. Repo không chứa bản sao bài báo, hình hay bảng của bài báo.

```bibtex
@article{nkhata2023bertbilstm,
  title   = {Fine-tuning BERT with Bidirectional LSTM for Fine-grained Movie Reviews Sentiment Analysis},
  author  = {Nkhata, Gibson and Gauch, Susan and Anjum, Usman and Zhan, Justin},
  journal = {International Journal On Advances in Systems and Measurements},
  volume  = {16},
  number  = {3 and 4},
  year    = {2023},
  note    = {arXiv:2502.20682}
}
```

## 2. Mã nguồn của tác giả

- **Nguồn:** https://github.com/gnkhata1/Finetuning-BERT-on-Movie-Reviews-Sentiment-Analysis, commit `c4cbd9a179a9526ee952d3e58247d3f1466ff0c6`
- **License:** repo gốc **không có file LICENSE**. Theo luật bản quyền, khi không có license, tác giả giữ mọi quyền; người khác không được phân phối lại mã.
- **Cách dự án sử dụng:** bản sao cục bộ ở `third_party/bert_bilstm_original/` chỉ để đọc và đối chiếu. Thư mục này **bị loại khỏi Git** (`.gitignore`) để không phân phối lại mã của tác giả. Mã PyTorch tái lập trong `scripts/train_reproduction.py` được viết lại hoàn toàn, chỉ bám theo hành vi đã mô tả trong `docs/code_audit.md`; không sao chép mã TensorFlow gốc.
- **Tải lại để đối chiếu:**

```bash
git clone https://github.com/gnkhata1/Finetuning-BERT-on-Movie-Reviews-Sentiment-Analysis.git third_party/bert_bilstm_original
git -C third_party/bert_bilstm_original checkout c4cbd9a179a9526ee952d3e58247d3f1466ff0c6
```

## 3. Dữ liệu

### 3.1. Large Movie Review Dataset v1.0 (IMDb)

- **Nguồn:** https://ai.stanford.edu/~amaas/data/sentiment/, file `aclImdb_v1.tar.gz` (SHA-256 `c40f74a18d3b61f90feba1e17730e0d38e8b97c05fde7008942e91923d1658fe`)
- **Chủ sở hữu:** Andrew L. Maas và cộng sự (Stanford). Nội dung review thuộc về người viết trên IMDb.
- **License:** file README của bộ dữ liệu không ghi license; tác giả yêu cầu trích dẫn bài ACL 2011 khi sử dụng.
- **Cách dự án sử dụng:** tải về `data/raw/` (bị loại khỏi Git), tạo split sạch ở `data/processed/imdb_full_v1/` (bị loại khỏi Git). Repo chỉ chứa manifest, thống kê, dự đoán theo `review_id`, và văn bản của 100 review validation dùng làm ví dụ phân tích lỗi (`reports/validation_analysis/cases_100*.json`, `index.html`). File `reports/validation_analysis/predictions.jsonl` chứa toàn văn 3.750 review validation nên bị loại khỏi Git. Ai muốn chạy lại phải tự tải bộ dữ liệu từ nguồn gốc.

```bibtex
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L. and Daly, Raymond E. and Pham, Peter T. and Huang, Dan and Ng, Andrew Y. and Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
```

### 3.2. IMDb Non-Commercial Datasets — `title.basics.tsv.gz`

- **Nguồn:** https://datasets.imdbws.com/title.basics.tsv.gz (trang mô tả: https://data.imdb.com/non-commercial-datasets/)
- **Chủ sở hữu:** IMDb.com, Inc.
- **Điều kiện:** chỉ dùng cho mục đích **cá nhân và phi thương mại**, theo điều khoản của IMDb: https://help.imdb.com/article/imdb/general-information/can-i-use-imdb-data-in-my-software/G5JTRESSHJBBHTGX, https://www.imdb.com/conditions, http://www.imdb.com/Copyright
- **Cách dự án sử dụng:** chỉ dùng thông tin thể loại phim để lọc tập con Horror ở giai đoạn đầu. File nằm trong `data/raw/` và không được phân phối lại.

## 4. Mô hình tiền huấn luyện

Mọi trọng số mô hình nằm trong `checkpoints/` và **bị loại khỏi Git**. Checkpoint fine-tune trong `work_dirs/` cũng không được đưa lên. Đây là sản phẩm phái sinh nên chịu license của mô hình gốc.

| Mô hình | Nguồn (revision) | License | Vai trò trong dự án | Trích dẫn |
|---|---|---|---|---|
| bert-base-uncased | https://huggingface.co/google-bert/bert-base-uncased (`86b5e0934494bd15c9632b12f734a8a67f723594`) | Apache-2.0 | Encoder cho tái lập, baseline và hướng B | Devlin et al., 2019, *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL |
| roberta-large | https://huggingface.co/FacebookAI/roberta-large (`722cf37b1afa9454edce342e7895e588b6ff1d59`) | MIT | Encoder cho hướng D | Liu et al., 2019, *RoBERTa: A Robustly Optimized BERT Pretraining Approach*, arXiv:1907.11692 |
| nllb-200-distilled-600M | https://huggingface.co/facebook/nllb-200-distilled-600M (`f8d333a098d19b4fd9a8b18f94170487ad3f821d`) | CC BY-NC 4.0 (chỉ phi thương mại) | Dịch Anh–Việt (chưa dùng cho kết quả) | NLLB Team, 2022, *No Language Left Behind*, arXiv:2207.04672 |
| Qwen3-8B (bản GGUF Q4_K_M) | https://huggingface.co/Qwen/Qwen3-8B-GGUF | Apache-2.0 | Gán nhãn silver theo khía cạnh cho phần ABSA (chạy cục bộ). Nhãn silver nằm trong `data/` và không được phân phối lại | Qwen Team, 2025, *Qwen3 Technical Report*, arXiv:2505.09388 |

Phương pháp cắt đầu–cuối 512 token theo: Sun, Qiu, Xu, Huang, 2019, *How to Fine-Tune BERT for Text Classification?*, CCL 2019, arXiv:1905.05583.

## 5. Thư viện phần mềm

Các thư viện không nằm trong repo. Chúng được cài qua `requirements-*.txt` hoặc pip/npm và giữ nguyên license của mình.

| Thư viện | License |
|---|---|
| PyTorch | BSD-3-Clause |
| Hugging Face transformers, tokenizers, safetensors, huggingface_hub | Apache-2.0 |
| NumPy, SciPy, scikit-learn | BSD-3-Clause |
| sentencepiece | Apache-2.0 |
| matplotlib (vẽ biểu đồ báo cáo) | Matplotlib License (dựa trên PSF) |
| docx (npm, tạo file Word báo cáo) | MIT |
| llama.cpp (chạy Qwen3-8B để gán nhãn, cài riêng, không nằm trong repo) | MIT |
| openpyxl (file Excel kiểm tay) | MIT |

## 6. Lưu ý khi sử dụng

- Vì có các thành phần **chỉ cho phép phi thương mại** (IMDb Non-Commercial Datasets, NLLB) và nguồn dữ liệu không ghi license rõ ràng (aclImdb), kết quả, checkpoint và dữ liệu phái sinh của dự án chỉ nên dùng cho **nghiên cứu và học tập**.
- Khi trích dẫn kết quả, cần trích dẫn bài báo gốc (mục 1), bộ dữ liệu (mục 3.1) và các mô hình đã dùng (mục 4).
- License của phần mã do dự án tự viết (nếu có) được ghi riêng trong file `LICENSE` và không áp dụng cho các thành phần ở trên.
