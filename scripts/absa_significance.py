"""Kiểm định ghép cặp cho 3 mô hình ABSA trên 25.000 review test (chỉ đọc, chạy CPU).

    python scripts/absa_significance.py [--resamples 10000]

Thống kê: Macro-F1 trung bình 6 khía cạnh (và Macro-F1 từng khía cạnh), lấy trung bình trên 3 seed
cùng số. Bootstrap ghép cặp theo phim: mỗi lần lấy mẫu lại cả phim, vì review của cùng một phim không
độc lập. Hiệu chỉnh Holm riêng cho nhóm so sánh chính (3 cặp, điểm trung bình) và nhóm theo khía cạnh
(3 cặp × 6 khía cạnh). Nhãn so sánh là nhãn silver do LLM gán.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from absa_core import LABELS  # noqa: E402

SEEDS = [42, 43, 44]
SYSTEMS = {
    "roberta_large": "work_dirs/absa_roberta_large_v1/seed{s}",
    "bert_base": "work_dirs/absa_bert_base_v1/seed{s}",
    "bilstm": "work_dirs/absa_bilstm_v1/seed{s}",
}
COMPARISONS = [("roberta_large", "bert_base"), ("roberta_large", "bilstm"), ("bert_base", "bilstm")]
K = len(LABELS)


def load(path: Path) -> dict:
    with open(path / "test_predictions.jsonl", encoding="utf-8") as f:
        return {r["review_id"]: r for r in map(json.loads, f) if r}


def macro_f1_from_confusion(c: np.ndarray) -> np.ndarray:
    """c[..., K, K] (hàng = nhãn đúng, cột = dự đoán) -> Macro-F1 trên các lớp có mặt.

    Lớp "có mặt" khi xuất hiện ở nhãn đúng hoặc dự đoán (2tp + fp + fn > 0), giống
    `absa_core.aspect_metrics`.
    """
    tp = np.diagonal(c, axis1=-2, axis2=-1)
    denom = c.sum(-1) + c.sum(-2)  # = 2tp + fn + fp
    present = denom > 0
    f1 = np.where(present, 2 * tp / np.where(present, denom, 1), 0.0)
    return f1.sum(-1) / present.sum(-1)


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adj = np.empty(len(pvalues))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adj[i] = running
    return adj.tolist()


def per_movie_confusion(runs, ids, aspects, cluster, n_movies):
    """(seed, movie, aspect, K, K) — số review theo (nhãn silver, dự đoán) trong từng phim."""
    idx = {lab: i for i, lab in enumerate(LABELS)}
    out = np.zeros((len(runs), n_movies, len(aspects), K, K))
    for s, run in enumerate(runs):
        for a, asp in enumerate(aspects):
            y = np.array([idx[run[i]["silver"][asp]] for i in ids])
            p = np.array([idx[run[i]["prediction"][asp]] for i in ids])
            flat = np.bincount(cluster * K * K + y * K + p, minlength=n_movies * K * K)
            out[s, :, a] = flat.reshape(n_movies, K, K)
    return out


def p_two_sided(d: np.ndarray) -> float:
    b = len(d)
    return float(min(1.0, 2 * min((np.sum(d <= 0) + 1) / (b + 1), (np.sum(d >= 0) + 1) / (b + 1))))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--out", default="reports/absa_v1/significance.json")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    preds = {n: [load(ROOT / p.format(s=s)) for s in SEEDS] for n, p in SYSTEMS.items()}
    ref = preds["roberta_large"][0]
    ids = sorted(ref)
    aspects = list(ref[ids[0]]["silver"])
    for name, runs in preds.items():
        for run in runs:
            if sorted(run) != ids:
                raise ValueError(f"{name}: tập review khác nhau")
            if any(run[i]["silver"] != ref[i]["silver"] for i in ids):
                raise ValueError(f"{name}: nhãn silver khác nhau")
    movies, cluster = np.unique([ref[i]["movie_id"] for i in ids], return_inverse=True)
    n = len(movies)

    C = {name: per_movie_confusion(runs, ids, aspects, cluster, n) for name, runs in preds.items()}
    # Điểm gốc: (seed, aspect)
    point = {name: macro_f1_from_confusion(C[name].sum(1)) for name in C}

    rng = np.random.default_rng(20261107)
    B = args.resamples
    boot = {name: [] for name in C}  # mỗi phần tử: (b, aspect), đã lấy trung bình theo seed
    for start in range(0, B, args.chunk):
        b = min(args.chunk, B - start)
        draw = rng.integers(0, n, (b, n))
        W = np.stack([np.bincount(row, minlength=n) for row in draw]).astype(float)
        for name in C:
            cm = np.einsum("bm,smakl->bsakl", W, C[name])
            boot[name].append(macro_f1_from_confusion(cm).mean(1))
    boot = {name: np.concatenate(v) for name, v in boot.items()}

    systems = {}
    for name in C:
        mean_b = boot[name].mean(1)
        systems[name] = {
            "macro_f1_mean_per_seed": point[name].mean(1).tolist(),
            "macro_f1_mean": float(point[name].mean()),
            "macro_f1_mean_ci95": [float(np.percentile(mean_b, q)) for q in (2.5, 97.5)],
            "per_aspect_macro_f1": {
                a: float(point[name][:, k].mean()) for k, a in enumerate(aspects)
            },
            "per_aspect_ci95": {
                a: [float(np.percentile(boot[name][:, k], q)) for q in (2.5, 97.5)]
                for k, a in enumerate(aspects)
            },
        }

    def compare(a, c, k=None):
        if k is None:
            d = boot[a].mean(1) - boot[c].mean(1)
            obs = float(point[a].mean() - point[c].mean())
        else:
            d = boot[a][:, k] - boot[c][:, k]
            obs = float(point[a][:, k].mean() - point[c][:, k].mean())
        return {
            "first": a,
            "second": c,
            "difference": obs,
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_value": p_two_sided(d),
        }

    primary = [compare(a, c) for a, c in COMPARISONS]
    for r, adj in zip(primary, holm([r["p_value"] for r in primary])):
        r["p_value_holm"] = adj
    per_aspect = []
    for k, asp in enumerate(aspects):
        for a, c in COMPARISONS:
            per_aspect.append(dict(compare(a, c, k), aspect=asp))
    for r, adj in zip(per_aspect, holm([r["p_value"] for r in per_aspect])):
        r["p_value_holm"] = adj

    report = {
        "test_reviews": len(ids),
        "movies": int(n),
        "seeds": SEEDS,
        "aspects": aspects,
        "resamples": B,
        "label_source": "silver (Qwen3-8B, prompt v3) — chưa phải nhãn người",
        "bootstrap": "ghép cặp, lấy mẫu lại theo movie_id; thống kê = Macro-F1 (lớp có mặt) "
        "trung bình trên 3 seed cùng số",
        "p_value_floor": 2 / (B + 1),
        "systems": systems,
        "comparisons_mean": primary,
        "comparisons_per_aspect": per_aspect,
        "sources": SYSTEMS,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for name, s in systems.items():
        lo, hi = s["macro_f1_mean_ci95"]
        print(f"{name:14s} Macro-F1 {s['macro_f1_mean']*100:.2f} CI95 [{lo*100:.2f}, {hi*100:.2f}]")
    for r in primary + per_aspect:
        tag = r.get("aspect", "MEAN")
        lo, hi = r["ci95"]
        print(
            f"{tag:9s} {r['first']} vs {r['second']}: {r['difference']*100:+.2f} điểm "
            f"CI95 [{lo*100:+.2f}, {hi*100:+.2f}] p_holm={r['p_value_holm']:.2g}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
