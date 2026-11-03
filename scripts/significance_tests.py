"""Paired significance tests on the official 25k test predictions (CPU only, read-only on work_dirs).

  python scripts/significance_tests.py [--resamples 10000]

Primary: movie-clustered paired bootstrap of the Macro-F1 difference, averaged over the 3 matched seeds
(the split is grouped by movie, so reviews of one movie are not independent). Holm correction over comparisons.
Secondary: exact McNemar test per matched seed.
"""

import argparse
import json
import numpy as np
from scipy.stats import binomtest
from train_baselines import ROOT
from runtime_io import atomic_json

SEEDS = [42, 43, 44]
SYSTEMS = {
    "finetune_roberta_large": "work_dirs/finetune_roberta_large_v1/seed{s}",
    "finetune_headtail512": "work_dirs/finetune_headtail_v1/seed{s}",
    "frozen_bilstm_headtail": "work_dirs/imdb_full_v1/bilstm_headtail_seed{s}",
    "frozen_bilstm_head": "work_dirs/imdb_full_v1/bilstm_seed{s}",
    "reproduction_code": "work_dirs/reproduction_v1/code_seed{s}",
    "reproduction_paper": "work_dirs/reproduction_v1/paper_seed{s}",
}
COMPARISONS = [
    ("finetune_roberta_large", "finetune_headtail512"),
    ("finetune_roberta_large", "frozen_bilstm_headtail"),
    ("finetune_roberta_large", "reproduction_code"),
    ("finetune_headtail512", "frozen_bilstm_headtail"),
    ("finetune_headtail512", "reproduction_code"),
    ("finetune_headtail512", "reproduction_paper"),
    ("frozen_bilstm_headtail", "reproduction_code"),
    ("frozen_bilstm_head", "reproduction_code"),
]


def load(path):
    rows = [
        json.loads(l)
        for l in (ROOT / path / "test_predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return {r["review_id"]: r for r in rows}


def macro_f1(c):
    """c[...,4] = tp, fp, fn, tn with positive label 1."""
    tp, fp, fn, tn = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    return (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fn + fp)) / 2


def holm(pvalues):
    order = np.argsort(pvalues)
    adj = np.empty(len(pvalues))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adj[i] = running
    return adj.tolist()


def main(args):
    preds = {name: [load(p.format(s=s)) for s in SEEDS] for name, p in SYSTEMS.items()}
    ids = sorted(preds["finetune_headtail512"][0])
    for name, runs in preds.items():
        for run in runs:
            assert sorted(run) == ids, f"{name}: review ids differ"
    ref = preds["finetune_headtail512"][0]
    labels = np.array([ref[i]["label"] for i in ids])
    for name, runs in preds.items():
        for run in runs:
            assert all(run[i]["label"] == ref[i]["label"] for i in ids), f"{name}: labels differ"
    movies, cluster = np.unique([ref[i]["movie_id"] for i in ids], return_inverse=True)
    n = len(movies)

    def counts(name):
        """(seeds, movies, 4) confusion counts per movie."""
        out = np.zeros((len(SEEDS), n, 4))
        for k, run in enumerate(preds[name]):
            p = np.array([run[i]["prediction"] for i in ids])
            for j, mask in enumerate(
                [
                    (p == 1) & (labels == 1),
                    (p == 1) & (labels == 0),
                    (p == 0) & (labels == 1),
                    (p == 0) & (labels == 0),
                ]
            ):
                out[k, :, j] = np.bincount(cluster[mask], minlength=n)
        return out

    C = {name: counts(name) for name in SYSTEMS}
    point = {
        name: [float(macro_f1(C[name][k].sum(0))) for k in range(len(SEEDS))] for name in SYSTEMS
    }
    rng = np.random.default_rng(20261031)
    B = args.resamples
    diffs = {c: [] for c in COMPARISONS}
    for start in range(0, B, 500):
        b = min(500, B - start)
        draw = rng.integers(0, n, (b, n))
        W = np.stack([np.bincount(row, minlength=n) for row in draw]).astype(
            float
        )  # movie multiplicities
        f1 = {name: macro_f1(np.einsum("bm,smk->bsk", W, C[name])).mean(1) for name in SYSTEMS}
        for a, c in COMPARISONS:
            diffs[(a, c)].append(f1[a] - f1[c])
    results = []
    for a, c in COMPARISONS:
        d = np.concatenate(diffs[(a, c)])
        obs = float(np.mean(point[a]) - np.mean(point[c]))
        p = min(1.0, 2 * min((np.sum(d <= 0) + 1) / (B + 1), (np.sum(d >= 0) + 1) / (B + 1)))
        mcnemar = []
        for k, s in enumerate(SEEDS):
            ra = np.array([preds[a][k][i]["prediction"] for i in ids]) == labels
            rc = np.array([preds[c][k][i]["prediction"] for i in ids]) == labels
            only_a = int(np.sum(ra & ~rc))
            only_c = int(np.sum(~ra & rc))
            mcnemar.append(
                {
                    "seed": s,
                    "only_first_correct": only_a,
                    "only_second_correct": only_c,
                    "p_value": (
                        float(binomtest(only_a, only_a + only_c, 0.5).pvalue)
                        if only_a + only_c
                        else 1.0
                    ),
                }
            )
        results.append(
            {
                "first": a,
                "second": c,
                "macro_f1_first_mean": float(np.mean(point[a])),
                "macro_f1_second_mean": float(np.mean(point[c])),
                "difference": obs,
                "bootstrap_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "bootstrap_p_value": float(p),
                "mcnemar_exact_per_seed": mcnemar,
            }
        )
    for r, adj in zip(results, holm([r["bootstrap_p_value"] for r in results])):
        r["bootstrap_p_value_holm"] = adj
    report = {
        "test_reviews": len(ids),
        "movies": int(n),
        "seeds": SEEDS,
        "resamples": B,
        "bootstrap": "paired, clustered by movie_id, statistic = Macro-F1 averaged over matched seeds",
        "p_value_floor": 2 / (B + 1),
        "macro_f1_per_seed": point,
        "comparisons": results,
        "sources": SYSTEMS,
    }
    atomic_json(ROOT / "reports/significance_tests.json", report)
    for r in results:
        mp = ["%.1e" % m["p_value"] for m in r["mcnemar_exact_per_seed"]]
        print(
            f"{r['first']} vs {r['second']}: diff={r['difference']*100:+.2f} pts CI95=[{r['bootstrap_ci95'][0]*100:+.2f},{r['bootstrap_ci95'][1]*100:+.2f}] "
            f"p_holm={r['bootstrap_p_value_holm']:.2g} mcnemar_p={mp}"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resamples", type=int, default=10000)
    main(p.parse_args())
