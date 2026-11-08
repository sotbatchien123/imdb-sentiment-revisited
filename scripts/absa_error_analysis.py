"""Phân tích lỗi cho 3 mô hình ABSA trên tập test (chỉ đọc, chạy CPU).

    python scripts/absa_error_analysis.py

Đầu ra: reports/absa_v1/analysis/error_analysis.json và error_analysis.md
- F1 từng lớp (positive/negative/neutral/none) cho từng khía cạnh, trung bình 3 seed.
- Ma trận nhầm lẫn (hàng = nhãn silver, cột = dự đoán), cộng 3 seed.
- Các kiểu nhầm phổ biến nhất của từng khía cạnh.
- Kết quả theo độ dài (review vượt 512 token hay không).
- Overall dự đoán so với số sao IMDb (1–4 và 7–10).
- "Ca khó": review mà cả 9 lượt chạy (3 mô hình × 3 seed) đều khác nhãn silver, ứng viên cho
  nhãn silver sai; kèm vài ví dụ có câu bằng chứng của LLM.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from absa_core import LABELS  # noqa: E402

SEEDS = [42, 43, 44]
SYSTEMS = {
    "roberta_large": ("RoBERTa-large", "work_dirs/absa_roberta_large_v1"),
    "bert_base": ("BERT-base", "work_dirs/absa_bert_base_v1"),
    "bilstm": ("BERT frozen + BiLSTM", "work_dirs/absa_bilstm_v1"),
}
SILVER = "data/processed/imdb_absa_v1/silver_qwen3_8b_v3/test.jsonl"
DATASET = "data/processed/imdb_full_v1/test.jsonl"
K = len(LABELS)
IDX = {lab: i for i, lab in enumerate(LABELS)}


def read_jsonl(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return {r["review_id"]: r for r in (json.loads(line) for line in f if line.strip())}


def class_f1(cm: np.ndarray) -> list:
    """F1 từng lớp; None nếu lớp không xuất hiện ở cả nhãn lẫn dự đoán."""
    out = []
    for k in range(K):
        denom = cm[k].sum() + cm[:, k].sum()
        out.append(float(2 * cm[k, k] / denom) if denom else None)
    return out


def snippet(text: str, evidence: str, width: int = 260) -> str:
    text = " ".join(text.replace("<br />", " ").split())
    pos = text.lower().find(evidence.lower()) if evidence else -1
    if pos < 0 or len(text) <= width:
        return text[:width] + ("…" if len(text) > width else "")
    start = max(0, pos - width // 3)
    end = min(len(text), start + width)
    return ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default="reports/absa_v1/analysis")
    ap.add_argument("--examples", type=int, default=3, help="số ví dụ ca khó mỗi khía cạnh")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    runs = {
        name: [read_jsonl(ROOT / d / f"seed{s}" / "test_predictions.jsonl") for s in SEEDS]
        for name, (_, d) in SYSTEMS.items()
    }
    results = {
        name: [
            json.loads((ROOT / d / f"seed{s}" / "result.json").read_text(encoding="utf-8"))
            for s in SEEDS
        ]
        for name, (_, d) in SYSTEMS.items()
    }
    ref = runs["roberta_large"][0]
    ids = sorted(ref)
    aspects = list(ref[ids[0]]["silver"])
    silver_full = read_jsonl(ROOT / SILVER)
    data = read_jsonl(ROOT / DATASET)

    y = {a: np.array([IDX[ref[i]["silver"][a]] for i in ids]) for a in aspects}
    pred = {
        name: {
            a: np.stack([np.array([IDX[run[i]["prediction"][a]] for i in ids]) for run in rs])
            for a in aspects
        }
        for name, rs in runs.items()
    }

    report = {
        "test_reviews": len(ids),
        "aspects": aspects,
        "labels": LABELS,
        "label_source": "silver (Qwen3-8B, prompt v3) — chưa phải nhãn người",
        "silver_distribution": {
            a: {lab: int((y[a] == k).sum()) for k, lab in enumerate(LABELS)} for a in aspects
        },
        "systems": {},
    }

    for name, (title, _) in SYSTEMS.items():
        sysrep = {"title": title, "per_aspect": {}}
        for a in aspects:
            cms = [
                np.bincount(y[a] * K + pred[name][a][s], minlength=K * K).reshape(K, K)
                for s in range(len(SEEDS))
            ]
            f1s = np.array([[np.nan if v is None else v for v in class_f1(cm)] for cm in cms])
            cm = sum(cms)
            errors = Counter()
            for t in range(K):
                for p in range(K):
                    if t != p:
                        errors[(LABELS[t], LABELS[p])] = int(cm[t, p])
            total_err = sum(errors.values())
            sysrep["per_aspect"][a] = {
                "class_f1": {
                    lab: (None if np.isnan(f1s[:, k]).all() else float(np.nanmean(f1s[:, k])))
                    for k, lab in enumerate(LABELS)
                },
                "confusion_sum_3_seeds": cm.tolist(),
                "top_errors": [
                    {
                        "silver": t,
                        "prediction": p,
                        "count_per_seed": c / len(SEEDS),
                        "share_of_errors": c / total_err if total_err else 0.0,
                    }
                    for (t, p), c in errors.most_common(4)
                ],
            }
        sysrep["length_groups"] = {
            g: {
                k: float(np.mean([r["test_length_groups"][g][k] for r in results[name]]))
                for k in results[name][0]["test_length_groups"][g]
            }
            for g in results[name][0]["test_length_groups"]
        }
        sysrep["overall_vs_imdb_rating"] = {
            k: float(np.mean([r["overall_vs_imdb_rating"][k] for r in results[name]]))
            for k in results[name][0]["overall_vs_imdb_rating"]
        }
        report["systems"][name] = sysrep

    # Overall dự đoán theo số sao (RoBERTa-large, cộng 3 seed)
    ratings = np.array([data[i]["rating"] for i in ids])
    k_over = aspects.index("overall")
    by_rating = {}
    for r in sorted(set(ratings.tolist())):
        m = ratings == r
        p = pred["roberta_large"]["overall"][:, m].ravel()
        by_rating[str(r)] = {
            "reviews": int(m.sum()),
            **{lab: float((p == k).mean()) for k, lab in enumerate(LABELS)},
            "silver": {lab: float((y["overall"][m] == k).mean()) for k, lab in enumerate(LABELS)},
        }
    report["roberta_overall_by_star_rating"] = by_rating
    assert k_over >= 0

    # Ca khó: cả 9 lượt chạy đều khác nhãn silver
    hard = {}
    rng = np.random.default_rng(2026)
    for a in aspects:
        allp = np.concatenate([pred[n][a] for n in SYSTEMS])  # (9, N)
        wrong_all = (allp != y[a]).all(0)
        agree = (allp == allp[0]).all(0)
        cand = np.where(wrong_all & agree)[0]
        pairs = Counter((LABELS[y[a][j]], LABELS[allp[0, j]]) for j in cand)
        examples = []
        for j in rng.permutation(cand)[: args.examples]:
            rid = ids[j]
            ev = silver_full[rid].get("evidence", {}).get(a, "")
            examples.append(
                {
                    "review_id": rid,
                    "rating": int(data[rid]["rating"]),
                    "silver": LABELS[y[a][j]],
                    "all_models_predict": LABELS[allp[0, j]],
                    "llm_evidence": ev,
                    "snippet": snippet(data[rid]["text"], ev),
                }
            )
        hard[a] = {
            "all_9_runs_wrong": int(wrong_all.sum()),
            "all_9_runs_wrong_and_agree": int(len(cand)),
            "share_of_reviews": float(len(cand) / len(ids)),
            "top_pairs": [
                {"silver": t, "models": p, "count": c} for (t, p), c in pairs.most_common(3)
            ],
            "examples": examples,
        }
    report["hard_cases"] = hard

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "error_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "error_analysis.md").write_text(to_markdown(report), encoding="utf-8")
    print(to_markdown(report))
    return 0


def pct(v) -> str:
    return "–" if v is None else f"{v * 100:.1f}"


def to_markdown(r: dict) -> str:
    A, L = r["aspects"], r["labels"]
    out = [
        "# Phân tích lỗi ABSA trên tập test",
        "",
        f"{r['test_reviews']:,} review test. ".replace(",", ".")
        + f"Nhãn so sánh: {r['label_source']}. Số liệu là trung bình 3 seed (42/43/44).",
        "",
        "## 1. F1 từng lớp (RoBERTa-large)",
        "",
        "| Khía cạnh | " + " | ".join(L) + " | Tỉ lệ none (silver) |",
        "|---|" + "---|" * (len(L) + 1),
    ]
    rob = r["systems"]["roberta_large"]["per_aspect"]
    for a in A:
        dist = r["silver_distribution"][a]
        none_share = dist["none"] / sum(dist.values())
        out.append(
            f"| {a} | "
            + " | ".join(pct(rob[a]["class_f1"][lab]) for lab in L)
            + f" | {none_share * 100:.0f}% |"
        )
    out += [
        "",
        "## 2. F1 lớp neutral theo mô hình",
        "",
        "| Khía cạnh | " + " | ".join(s["title"] for s in r["systems"].values()) + " |",
        "|---|" + "---|" * len(r["systems"]),
    ]
    for a in A:
        out.append(
            f"| {a} | "
            + " | ".join(
                pct(s["per_aspect"][a]["class_f1"]["neutral"]) for s in r["systems"].values()
            )
            + " |"
        )
    out += ["", "## 3. Kiểu nhầm phổ biến nhất (RoBERTa-large)", ""]
    for a in A:
        errs = ", ".join(
            f"{e['silver']}→{e['prediction']} {e['share_of_errors'] * 100:.0f}%"
            for e in rob[a]["top_errors"][:3]
        )
        out.append(f"- **{a}:** {errs}")
    out += [
        "",
        "## 4. Theo độ dài review (Macro-F1 trung bình 6 khía cạnh)",
        "",
        "| Mô hình | ≤ 512 token | > 512 token |",
        "|---|---|---|",
    ]
    for s in r["systems"].values():
        g = s["length_groups"]
        out.append(
            f"| {s['title']} | {pct(g['short']['macro_f1'])} | {pct(g['long']['macro_f1'])} |"
        )
    out += [
        "",
        "## 5. Overall dự đoán theo số sao IMDb (RoBERTa-large, % dự đoán)",
        "",
        "| Sao | Số review | positive | negative | neutral | none |",
        "|---|---|---|---|---|---|",
    ]
    for star, v in r["roberta_overall_by_star_rating"].items():
        out.append(
            f"| {star} | {v['reviews']} | {pct(v['positive'])} | {pct(v['negative'])} | "
            f"{pct(v['neutral'])} | {pct(v['none'])} |"
        )
    out += [
        "",
        "## 6. Ca khó: cả 9 lượt chạy cùng dự đoán một nhãn khác silver",
        "",
        "Đây là ứng viên cho nhãn silver sai, nên ưu tiên khi nhóm kiểm tay.",
        "",
        "| Khía cạnh | Số review | Tỉ lệ | Cặp (silver → mô hình) phổ biến |",
        "|---|---|---|---|",
    ]
    for a in A:
        h = r["hard_cases"][a]
        pairs = ", ".join(f"{p['silver']}→{p['models']} ({p['count']})" for p in h["top_pairs"])
        out.append(
            f"| {a} | {h['all_9_runs_wrong_and_agree']} | {h['share_of_reviews'] * 100:.1f}% | {pairs} |"
        )
    out += ["", "### Ví dụ", ""]
    for a in A:
        for e in r["hard_cases"][a]["examples"]:
            out.append(
                f"- **{a}** `{e['review_id']}` ({e['rating']} sao): silver **{e['silver']}**, "
                f"mô hình **{e['all_models_predict']}**. Bằng chứng LLM: “{e['llm_evidence']}”. "
                f"Trích: “{e['snippet']}”"
            )
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.exit(main())
