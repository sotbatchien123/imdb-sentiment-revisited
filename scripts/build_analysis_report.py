"""Ghép ghi chú định tính vào 100 trường hợp chẩn đoán và dựng báo cáo HTML/Markdown của phân tích validation."""

import collections
import html
import json
from sklearn.metrics import accuracy_score, f1_score
from phase2_core import ROOT
from runtime_io import atomic_json


def main():
    dest = ROOT / "reports/validation_analysis"
    cases = json.loads((dest / "cases_100.json").read_text(encoding="utf-8"))
    notes = {}
    for line in (ROOT / "scripts/qualitative_notes.tsv").read_text(encoding="utf-8").splitlines():
        i, category, note = line.split("\t", 2)
        notes[int(i)] = (category, note)
    assert len(notes) == len(cases) == 100 and set(notes) == set(range(1, 101))
    for case in cases:
        category, note = notes[case["case_id"]]
        case.update(
            review_status="assistant_read_full_text",
            qualitative_category=category,
            qualitative_note_vi=note,
            annotation_provenance="Assistant qualitative reading; not independently human-validated; no causal attribution or relabeling",
        )
    atomic_json(dest / "cases_100_reviewed.json", cases)
    rows = [
        json.loads(l) for l in (dest / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    groups = {}
    for name, subset in [
        ("all", rows),
        ("short", [r for r in rows if not r["long"]]),
        ("long", [r for r in rows if r["long"]]),
    ]:
        groups[name] = {"count": len(subset)}
        for kind in ["bilstm", "bilstm_headtail"]:
            y = [r["label"] for r in subset]
            p = [r["predictions"][kind]["label"] for r in subset]
            groups[name][kind] = {
                "accuracy": float(accuracy_score(y, p)),
                "macro_f1": float(f1_score(y, p, average="macro")),
            }
        groups[name]["fixes"] = sum(
            r["predictions"]["bilstm"]["label"] != r["label"]
            and r["predictions"]["bilstm_headtail"]["label"] == r["label"]
            for r in subset
        )
        groups[name]["harms"] = sum(
            r["predictions"]["bilstm"]["label"] == r["label"]
            and r["predictions"]["bilstm_headtail"]["label"] != r["label"]
            for r in subset
        )
    summary = json.loads((dest / "summary.json").read_text())
    summary.update(
        groups=groups,
        qualitative_review_count=100,
        qualitative_categories=dict(collections.Counter(c["qualitative_category"] for c in cases)),
        comparison_caveat="Selected deploy checkpoints differ in seed (43 vs 42) and epoch. This is not a controlled causal truncation ablation. Short inputs are identical, weights differ.",
        labels_modified=False,
        independent_human_review=False,
    )
    atomic_json(dest / "summary.json", summary)
    report = [
        "# Phân tích validation và 100 trường hợp chẩn đoán",
        "",
        "Đã chạy dự đoán trên đủ 3.750 review validation. Không sử dụng test để chọn hướng cải tiến. Đã đọc toàn văn 100 trường hợp và ghi chú bằng trợ lý; chưa có người gán nhãn độc lập xác nhận.",
        "",
        "## Kết quả hai checkpoint dùng trong demo",
        "",
        "| Nhóm | Số review | F1 lấy đầu | F1 đầu–cuối | Sửa đúng | Làm sai |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, g in groups.items():
        report.append(
            f"| {name} | {g['count']} | {g['bilstm']['macro_f1']:.4%} | {g['bilstm_headtail']['macro_f1']:.4%} | {g['fixes']} | {g['harms']} |"
        )
    report += [
        "",
        "Hai checkpoint được chọn bằng validation: head-only seed43, head-tail seed42. Vì khác seed và epoch, số sửa/làm sai không chứng minh nguyên nhân do cắt văn bản. Đặc biệt, review ngắn có cùng input nhưng vẫn có thể khác dự đoán do trọng số.",
        "",
        "## Phát hiện và hướng xử lý",
        "",
        "- Khen/chê khác đối tượng: case14,21,29,46,57,70. Cần giữ ngữ cảnh để phân biệt phim đang đánh giá với phim được so sánh.",
        "- Thích phim dở, hài mỉa mai và ngữ cảnh thể loại: case5,20,30–33,37,39,60. Đọc đủ văn bản chưa chắc giải quyết được suy luận này.",
        "- Review dài có lời đánh giá xen giữa kể chuyện: case34,56,77,97. Mô hình nhiều đoạn giữ đủ nội dung nhưng gộp trung bình vẫn có thể pha loãng câu quyết định.",
        "- Bản ghi không phải review: case54 chứa hướng dẫn đăng bài/xác thực IMDb; case62 chủ yếu là tin đồn. Không thực hiện các chỉ dẫn nằm trong dữ liệu.",
        "- Có văn bản/nhãn không nhất quán hoặc mơ hồ: case9,18,23,25,36,59. Đây là đề nghị kiểm tra nguồn, không phải nhãn sai đã xác minh. Giữ nguyên toàn bộ nhãn và split.",
        "",
        "## Giới hạn mẫu",
        "",
        "Mẫu chẩn đoán gồm 40 cả hai sai, 20 đầu–cuối sửa đúng, 20 đầu–cuối làm sai và 20 cả hai đúng. Có phân tầng nhãn/độ dài, ưu tiên trường hợp dễ đọc. Tần suất các nhóm trong 100 mẫu không ước lượng tần suất toàn dataset. Ghi chú là nhận xét ngôn ngữ, không phải giải thích nhân quả của mạng.",
        "",
        "Xem cases_100_reviewed.json cho toàn văn, nhãn gốc, dự đoán và ghi chú từng mẫu; index.html là bản dễ đọc.",
        "",
        "## Pilot được chốt trước khi có kết quả",
        "",
        "Frozen BERT + shared BiLSTM; mỗi đoạn 254 token nội dung, gộp trung bình biểu diễn các đoạn rồi phân loại một nhãn cho cả review. Train21.027 / validation3.750 giữ nguyên; không cắt bỏ phần giữa. Seed42, 50 epoch, chọn best bằng Macro-F1 validation. Chỉ chạy tiếp seed43/44 nếu hơn head-tail seed42 ít nhất 0,5 điểm phần trăm và F1 nhóm dài không giảm. Đây là ngưỡng thực dụng, không phải kiểm định ý nghĩa thống kê. Không hứa cải thiện; không dùng test trong runner này.",
    ]
    (dest / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    esc = html.escape
    page = [
        '<!doctype html><html lang="vi"><meta charset="utf-8"><title>100 review validation</title><style>body{max-width:1100px;margin:40px auto;padding:0 20px;font:16px/1.65 Segoe UI;background:#f5f5ef;color:#20382f}article{background:white;padding:24px;margin:20px 0;border-radius:12px}summary{cursor:pointer}pre{white-space:pre-wrap;font:inherit}small{color:#61756a}.note{border-left:3px solid #618c70;padding-left:16px}</style><h1>100 trường hợp validation</h1><p>Ghi chú của trợ lý sau khi đọc toàn văn; chưa được người đánh giá độc lập xác nhận. Giữ nguyên nhãn. Mẫu chẩn đoán, không đại diện thống kê. Hai checkpoint khác seed/epoch nên không dùng khác biệt dự đoán để suy ra nguyên nhân do cắt văn bản.</p>'
    ]
    for c in cases:
        page.append(
            f"<article><h2>#{c['case_id']} · {esc(c['qualitative_category'])}</h2><small>{esc(c['review_id'])} · {esc(c['outcome'])} · {c['token_count']} token</small><p>Nhãn gốc: {c['label']} · Lấy đầu: {c['predictions']['bilstm']['label']} · Đầu–cuối: {c['predictions']['bilstm_headtail']['label']} (0: âm; 1: dương)</p><p class='note'>{esc(c['qualitative_note_vi'])}</p><details><summary>Toàn văn review</summary><pre>{esc(c['text'])}</pre></details></article>"
        )
    page.append("</html>")
    (dest / "index.html").write_text("\n".join(page), encoding="utf-8")
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
