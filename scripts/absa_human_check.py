"""Kiểm tay nhãn ABSA trên tập test: tạo file Excel cho nhóm, rồi chấm điểm khi nhóm gán xong.

Cần openpyxl (có trong anaconda; .venv-train không có):

    python scripts/absa_human_check.py prepare            # tạo annotator_1..3.xlsx
    python scripts/absa_human_check.py score              # đọc file đã điền, tính điểm

Lấy mẫu: ngẫu nhiên phân tầng theo số sao (1–4, 7–10), mỗi phim tối đa một review, seed cố định.
Mặc định 100 review chung (cả 3 người cùng gán, để đo độ đồng thuận) + 100 review riêng mỗi người,
tức 400 review khác nhau, mỗi người 200 review.

Chế độ mặc định là "kiểm tra": file hiện nhãn và câu bằng chứng của LLM, người gán điền nhãn của
mình (chép lại nhãn LLM nếu đồng ý). Dùng --blind để ẩn nhãn LLM (chậm hơn nhưng không bị gợi ý).

Khi chấm: nhãn người của review chung lấy theo đa số (≥ 2/3); review không có đa số bị bỏ ở
khía cạnh đó. Điểm của LLM và của 9 lượt chạy mô hình được tính trên nhãn người này.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["positive", "negative", "neutral", "none"]
ASPECTS = ["acting", "story", "visual", "sound", "direction", "overall"]
SILVER = "data/processed/imdb_absa_v1/silver_qwen3_8b_v3/test.jsonl"
DATASET = "data/processed/imdb_full_v1/test.jsonl"
SEEDS = [42, 43, 44]
SYSTEMS = {
    "roberta_large": "work_dirs/absa_roberta_large_v1",
    "bert_base": "work_dirs/absa_bert_base_v1",
    "bilstm": "work_dirs/absa_bilstm_v1",
}
OUT_DIR = "reports/absa_v1/human_check"

GUIDE = [
    ("Mục đích", "Kiểm tra nhãn do LLM (Qwen3-8B) gán cho 6 khía cạnh của review IMDb."),
    (
        "Cách làm",
        "Đọc review. Với mỗi khía cạnh, điền nhãn của bạn vào cột 'NGƯỜI: <khía cạnh>' "
        "(chọn trong danh sách). Đồng ý với LLM thì chọn đúng nhãn đó; không đồng ý thì chọn nhãn khác.",
    ),
    ("Không bỏ trống", "Mọi ô 'NGƯỜI' phải có nhãn. Có thể ghi chú ở cột 'Ghi chú'."),
    (
        "Không bàn nhau",
        "Các review chung (cả 3 người cùng gán) dùng để đo độ đồng thuận, nên mỗi "
        "người gán độc lập.",
    ),
    ("positive", "Người viết khen rõ khía cạnh này."),
    ("negative", "Người viết chê rõ khía cạnh này."),
    (
        "neutral",
        "Có nói tới khía cạnh này nhưng không khen/chê rõ, HOẶC khen và chê ngang nhau. "
        "Chỉ nêu tên diễn viên/vai mà không đánh giá → acting neutral.",
    ),
    ("none", "Review không nói tới khía cạnh này (phổ biến nhất ở visual, sound, direction)."),
    ("acting", "Diễn xuất, dàn diễn viên, chọn vai, lồng tiếng."),
    ("story", "Cốt truyện, kịch bản, thoại, nhân vật xét theo cách viết, twist, kết thúc, logic."),
    ("visual", "Quay phim, ánh sáng, kỹ xảo/CGI, bối cảnh, trang phục, hoá trang, hình ảnh."),
    ("sound", "Nhạc nền, soundtrack, bài hát, hiệu ứng âm thanh, chất lượng âm thanh."),
    ("direction", "Công việc/lựa chọn của đạo diễn. Chỉ nêu tên đạo diễn thì chưa đủ."),
    (
        "overall",
        "Đánh giá chung về bộ phim. Điểm số quyết định: 1–4/10 negative, 7–10/10 positive. "
        "Hầu như không bao giờ là none.",
    ),
    (
        "Quy tắc chung",
        "Chỉ dựa vào chữ trong review, không suy từ giọng điệu chung. Câu chung chung "
        "như 'great movie', 'waste of time' chỉ tính cho overall. Mỉa mai tính theo ý thật.",
    ),
]


def read_jsonl(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return {r["review_id"]: r for r in (json.loads(line) for line in f if line.strip())}


# ---------------------------------------------------------------- lấy mẫu


def sample_reviews(data: dict, total: int, seed: int) -> list[str]:
    """Phân tầng theo số sao, mỗi phim tối đa một review."""
    rng = np.random.default_rng(seed)
    by_star = defaultdict(list)
    for rid, r in data.items():
        by_star[r["rating"]].append(rid)
    stars = sorted(by_star)
    n_all = sum(len(v) for v in by_star.values())
    exact = {s: total * len(by_star[s]) / n_all for s in stars}
    quota = {s: int(exact[s]) for s in stars}
    for s in sorted(stars, key=lambda s: exact[s] - quota[s], reverse=True)[
        : total - sum(quota.values())
    ]:
        quota[s] += 1  # chia phần dư cho tầng có phần lẻ lớn nhất
    used_movies, chosen = set(), []
    for s in stars:
        picked = 0
        for rid in rng.permutation(sorted(by_star[s])):
            if picked == quota[s]:
                break
            movie = data[rid]["movie_id"]
            if movie in used_movies:
                continue
            used_movies.add(movie)
            chosen.append(str(rid))
            picked += 1
    return chosen


def assign(ids: list[str], shared: int, per: int, annotators: int, seed: int) -> dict:
    rng = np.random.default_rng(seed + 1)
    ids = [ids[i] for i in rng.permutation(len(ids))]
    common = ids[:shared]
    plan = {}
    for a in range(annotators):
        own = ids[shared + a * per : shared + (a + 1) * per]
        order = common + own
        plan[f"annotator_{a + 1}"] = {
            "shared": common,
            "own": own,
            "order": [
                order[i] for i in np.random.default_rng(seed + 10 + a).permutation(len(order))
            ],
        }
    return plan


def clean(text: str) -> str:
    return text.replace("<br /><br />", "\n").replace("<br />", "\n").replace("<br/>", "\n").strip()


def write_workbook(path: Path, order: list[str], data: dict, silver: dict, blind: bool) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "Gan nhan"
    head = ["STT", "review_id", "Review"]
    for a in ASPECTS:
        if not blind:
            head += [f"LLM: {a}", f"Bằng chứng LLM: {a}"]
        head.append(f"NGƯỜI: {a}")
    head.append("Ghi chú")
    ws.append(head)
    bold = Font(bold=True)
    human_fill = PatternFill("solid", fgColor="FFF2CC")
    for c in ws[1]:
        c.font = bold
        c.alignment = Alignment(wrap_text=True, vertical="top")
    for n, rid in enumerate(order, 1):
        row = [n, rid, clean(data[rid]["text"])]
        for a in ASPECTS:
            if not blind:
                row += [silver[rid]["labels"][a], silver[rid].get("evidence", {}).get(a, "")]
            row.append(None)
        row.append(None)
        ws.append(row)
    dv = DataValidation(type="list", formula1='"' + ",".join(LABELS) + '"', allow_blank=False)
    dv.error, dv.errorTitle = "Chỉ chọn positive, negative, neutral hoặc none", "Nhãn không hợp lệ"
    ws.add_data_validation(dv)
    from openpyxl.utils import get_column_letter

    for col, name in enumerate(head, 1):
        letter = get_column_letter(col)
        if name == "Review":
            ws.column_dimensions[letter].width = 90
        elif name.startswith("Bằng chứng"):
            ws.column_dimensions[letter].width = 28
        elif name.startswith("NGƯỜI"):
            ws.column_dimensions[letter].width = 14
            dv.add(f"{letter}2:{letter}{len(order) + 1}")
            for r in range(2, len(order) + 2):
                ws[f"{letter}{r}"].fill = human_fill
        else:
            ws.column_dimensions[letter].width = 12 if name != "Ghi chú" else 30
    for r in ws.iter_rows(min_row=2):
        for c in r:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "D2"

    g = wb.create_sheet("Huong dan")
    g.append(["Mục", "Nội dung"])
    for k, v in GUIDE:
        g.append([k, v])
    g.column_dimensions["A"].width = 16
    g.column_dimensions["B"].width = 110
    for r in g.iter_rows():
        for c in r:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    g["A1"].font = g["B1"].font = bold
    wb.move_sheet("Huong dan", offset=-1)
    wb.active = 1
    wb.save(path)


def cmd_prepare(args) -> int:
    data = read_jsonl(ROOT / DATASET)
    silver = read_jsonl(ROOT / SILVER)
    total = args.shared + args.per_annotator * args.annotators
    ids = sample_reviews(data, total, args.seed)
    plan = assign(ids, args.shared, args.per_annotator, args.annotators, args.seed)
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    for name, p in plan.items():
        path = out / f"{name}.xlsx"
        if path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{path} đã có (có thể đã được điền); thêm --overwrite nếu chắc chắn"
            )
        write_workbook(path, p["order"], data, silver, args.blind)
    manifest = {
        "seed": args.seed,
        "blind": args.blind,
        "shared": args.shared,
        "per_annotator": args.per_annotator,
        "annotators": args.annotators,
        "unique_reviews": len(ids),
        "aspects": ASPECTS,
        "labels": LABELS,
        "stars": dict(sorted(Counter(data[i]["rating"] for i in ids).items())),
        "plan": plan,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    words = [len(data[i]["text"].split()) for i in ids]
    print(
        f"Đã tạo {args.annotators} file trong {out} — {len(ids)} review khác nhau, "
        f"mỗi người {args.shared + args.per_annotator} review; trung bình {np.mean(words):.0f} từ/review."
    )
    return 0


# ---------------------------------------------------------------- chấm điểm


def fleiss_kappa(table: np.ndarray) -> float:
    """table[item, category] = số người chọn; mọi item có cùng số người."""
    n = table.sum(1)[0]
    p_cat = table.sum(0) / table.sum()
    p_item = ((table**2).sum(1) - n) / (n * (n - 1))
    pe = (p_cat**2).sum()
    return float((p_item.mean() - pe) / (1 - pe)) if pe < 1 else 1.0


def majority(labels: list[str]) -> str | None:
    lab, c = Counter(labels).most_common(1)[0]
    return lab if c * 2 > len(labels) else None


def macro_f1(y: list[str], p: list[str]) -> float:
    present = sorted(set(y) | set(p))
    f1 = []
    for lab in present:
        tp = sum(a == lab and b == lab for a, b in zip(y, p))
        denom = sum(a == lab for a in y) + sum(b == lab for b in p)
        f1.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1)) if f1 else float("nan")


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in LABELS) / (n * n)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def read_workbook(path: Path) -> dict:
    from openpyxl import load_workbook

    ws = load_workbook(path, read_only=True)["Gan nhan"]
    rows = list(ws.iter_rows(values_only=True))
    head = list(rows[0])
    col = {a: head.index(f"NGƯỜI: {a}") for a in ASPECTS}
    rid_col = head.index("review_id")
    out, problems = {}, []
    for r in rows[1:]:
        if not r or r[rid_col] is None:
            continue
        labels = {}
        for a in ASPECTS:
            v = (r[col[a]] or "").strip().lower() if isinstance(r[col[a]], str) else r[col[a]]
            if v in LABELS:
                labels[a] = v
            else:
                problems.append((r[rid_col], a, v))
        out[r[rid_col]] = labels
    return {"labels": out, "problems": problems}


def build_gold(manifest: dict, filled: dict) -> tuple[dict, dict]:
    """gold[review][aspect] = nhãn người (đa số với review chung)."""
    gold = defaultdict(dict)
    agreement = {}
    shared = manifest["plan"]["annotator_1"]["shared"]
    names = sorted(filled)
    for a in ASPECTS:
        items = [i for i in shared if all(a in filled[n]["labels"].get(i, {}) for n in names)]
        if items and len(names) >= 2:
            table = np.array(
                [
                    [sum(filled[n]["labels"][i][a] == lab for n in names) for lab in LABELS]
                    for i in items
                ]
            )
            pairs = [
                cohen_kappa(
                    [filled[x]["labels"][i][a] for i in items],
                    [filled[y]["labels"][i][a] for i in items],
                )
                for k, x in enumerate(names)
                for y in names[k + 1 :]
            ]
            agreement[a] = {
                "items": len(items),
                "fleiss_kappa": fleiss_kappa(table),
                "mean_pairwise_cohen_kappa": float(np.mean(pairs)),
                "unanimous": float(np.mean([t.max() == len(names) for t in table])),
            }
        for i in shared:
            votes = [
                filled[n]["labels"][i][a] for n in names if a in filled[n]["labels"].get(i, {})
            ]
            if votes and (m := majority(votes)) is not None:
                gold[i][a] = m
    for n in names:
        for i in manifest["plan"][n]["own"]:
            for a, lab in filled[n]["labels"].get(i, {}).items():
                gold[i][a] = lab
    return dict(gold), agreement


def score_against(gold: dict, pred: dict) -> dict:
    per = {}
    for a in ASPECTS:
        ids = [i for i in gold if a in gold[i] and i in pred]
        y = [gold[i][a] for i in ids]
        p = [pred[i][a] for i in ids]
        per[a] = {
            "n": len(ids),
            "macro_f1": macro_f1(y, p),
            "cohen_kappa": cohen_kappa(y, p),
            "accuracy": float(np.mean([u == v for u, v in zip(y, p)])),
        }
    return {
        "per_aspect": per,
        "macro_f1_mean": float(np.mean([v["macro_f1"] for v in per.values()])),
    }


def cmd_score(args) -> int:
    out = ROOT / args.out_dir
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    filled, problems = {}, {}
    for name in manifest["plan"]:
        path = out / f"{name}.xlsx"
        if not path.exists():
            print(f"Thiếu {path}, bỏ qua")
            continue
        filled[name] = read_workbook(path)
        problems[name] = len(filled[name]["problems"])
    if not filled:
        raise SystemExit("Chưa có file nào để chấm")
    gold, agreement = build_gold(manifest, filled)
    silver = {i: r["labels"] for i, r in read_jsonl(ROOT / SILVER).items() if i in gold}
    report = {
        "annotated_reviews": len(gold),
        "missing_or_invalid_cells": problems,
        "inter_annotator": agreement,
        "llm_silver_vs_human": score_against(gold, silver),
        "llm_changed_by_humans": {
            a: float(np.mean([silver[i][a] != gold[i][a] for i in gold if a in gold[i]]))
            for a in ASPECTS
        },
        "models_vs_human": {},
        "models_vs_silver_same_reviews": {},
    }
    for name, d in SYSTEMS.items():
        runs_h, runs_s = [], []
        for s in SEEDS:
            pred = {
                i: r["prediction"]
                for i, r in read_jsonl(ROOT / d / f"seed{s}" / "test_predictions.jsonl").items()
                if i in gold
            }
            runs_h.append(score_against(gold, pred))
            runs_s.append(score_against({i: silver[i] for i in gold}, pred))
        for key, runs in (("models_vs_human", runs_h), ("models_vs_silver_same_reviews", runs_s)):
            vals = [r["macro_f1_mean"] for r in runs]
            report[key][name] = {
                "macro_f1_mean": float(np.mean(vals)),
                "macro_f1_std": float(np.std(vals, ddof=1)),
                "per_aspect_macro_f1": {
                    a: float(np.mean([r["per_aspect"][a]["macro_f1"] for r in runs]))
                    for a in ASPECTS
                },
            }
    (out / "human_eval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: report[k] for k in ("annotated_reviews", "missing_or_invalid_cells")},
            ensure_ascii=False,
        )
    )
    print("LLM so với người, Macro-F1:", round(report["llm_silver_vs_human"]["macro_f1_mean"], 4))
    for name, v in report["models_vs_human"].items():
        print(f"{name} so với người: {v['macro_f1_mean']:.4f} ± {v['macro_f1_std']:.4f}")
    for a, v in agreement.items():
        print(f"{a}: Fleiss κ = {v['fleiss_kappa']:.3f} trên {v['items']} review chung")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--shared", type=int, default=100)
    p.add_argument("--per-annotator", type=int, default=100)
    p.add_argument("--annotators", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261107)
    p.add_argument("--blind", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--out-dir", default=OUT_DIR)
    s = sub.add_parser("score")
    s.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    return cmd_prepare(args) if args.cmd == "prepare" else cmd_score(args)


if __name__ == "__main__":
    sys.exit(main())
