"""So sánh hai bộ nhãn ABSA trên cùng các review (ví dụ nhãn LLM với bộ tham chiếu).

Mỗi file là JSONL có `review_id` và `labels` {khía cạnh: nhãn}. Script in ra, cho từng khía
cạnh: tỉ lệ khớp, Cohen's kappa, Macro-F1 (coi file thứ nhất là nhãn đúng) và ma trận nhầm lẫn.
Nếu có `--dataset`, script đối chiếu thêm overall với nhãn sao gốc của IMDb.

Ví dụ:
    python scripts/absa_compare_labels.py \
        data/processed/imdb_absa_v1/reference/validation_300_claude.jsonl \
        data/processed/imdb_absa_v1/silver_qwen3_8b/validation.jsonl \
        --dataset data/processed/imdb_full_v1/validation.jsonl --out reports/absa_v1/agreement.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

LABELS = ["positive", "negative", "neutral", "none"]


def read_labels(path: Path) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["review_id"]] = rec["labels"]
    return out


def compare(ref: dict, other: dict, aspects: list[str]) -> dict:
    ids = sorted(set(ref) & set(other))
    report = {"common_reviews": len(ids), "per_aspect": {}}
    for a in aspects:
        y = [ref[i][a] for i in ids]
        p = [other[i][a] for i in ids]
        report["per_aspect"][a] = {
            "agreement": sum(u == v for u, v in zip(y, p)) / len(ids) if ids else None,
            "cohen_kappa": float(cohen_kappa_score(y, p, labels=LABELS)) if ids else None,
            "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)) if ids else None,
            "confusion_matrix": {
                "labels": LABELS,
                "rows_reference_cols_other": confusion_matrix(y, p, labels=LABELS).tolist(),
            },
        }
    vals = list(report["per_aspect"].values())
    report["mean"] = {
        k: sum(v[k] for v in vals) / len(vals) for k in ("agreement", "cohen_kappa", "macro_f1")
    }
    return report


def overall_vs_rating(labels: dict, dataset: Path) -> dict:
    gold = {}
    with open(dataset, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                gold[r["review_id"]] = r["label"]
    ids = [i for i in labels if i in gold]
    decided = [i for i in ids if labels[i]["overall"] in ("positive", "negative")]
    correct = sum((labels[i]["overall"] == "positive") == (gold[i] == 1) for i in decided)
    return {
        "reviews": len(ids),
        "decided_fraction": len(decided) / len(ids) if ids else None,
        "accuracy_on_decided": correct / len(decided) if decided else None,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("reference")
    ap.add_argument("other", nargs="?")
    ap.add_argument("--dataset", help="file JSONL gốc có `label` 0/1 để đối chiếu overall")
    ap.add_argument(
        "--aspects",
        nargs="+",
        default=["acting", "story", "visual", "sound", "direction", "pacing", "overall"],
    )
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ref = read_labels(Path(args.reference))
    result = {"reference": args.reference}
    if args.dataset:
        result["reference_overall_vs_imdb_rating"] = overall_vs_rating(ref, Path(args.dataset))
    if args.other:
        other = read_labels(Path(args.other))
        result["other"] = args.other
        result["comparison"] = compare(ref, other, args.aspects)
        if args.dataset:
            result["other_overall_vs_imdb_rating"] = overall_vs_rating(other, Path(args.dataset))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
