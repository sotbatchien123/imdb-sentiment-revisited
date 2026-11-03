"""Use all labeled IMDb reviews; preserve official test and audit training exclusions."""

import collections
import hashlib
import html
import json
import re
import tarfile
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from runtime_io import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/imdb_full_v1"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "manifest.json").exists():
        print("Full dataset already prepared; preserving existing split.")
        return
    archive = ROOT / "data/raw/aclImdb_v1.tar.gz"
    urls = {}
    rows = {"train": [], "test": []}
    with tarfile.open(archive, "r:gz") as tf:
        for m in tf:
            match = re.fullmatch(r"aclImdb/(train|test)/urls_(pos|neg)\.txt", m.name)
            if match:
                urls[match.groups()] = tf.extractfile(m).read().decode().splitlines()
    with tarfile.open(archive, "r:gz") as tf:
        for m in tf:
            match = re.fullmatch(r"aclImdb/(train|test)/(pos|neg)/(\d+)_(\d+)\.txt", m.name)
            if not match:
                continue
            split, label, idx, rating = match.groups()
            mid = re.search(r"tt\d+", urls[(split, label)][int(idx)]).group()
            text = " ".join(
                html.unescape(
                    re.sub(r"<[^>]+>", " ", tf.extractfile(m).read().decode("utf-8"))
                ).split()
            )
            assert text
            rows[split].append(
                {
                    "review_id": f"{split}/{label}/{idx}_{rating}",
                    "original_split": split,
                    "movie_id": mid,
                    "label": int(label == "pos"),
                    "rating": int(rating),
                    "text": text,
                    "text_hash": hashlib.sha256(text.casefold().encode()).hexdigest(),
                }
            )
    assert len(rows["train"]) == 25000 and len(rows["test"]) == 25000
    for rs in rows.values():
        rs.sort(key=lambda r: r["review_id"])
    test_movies = {r["movie_id"] for r in rows["test"]}
    test_hashes = {r["text_hash"] for r in rows["test"]}
    label_sets = collections.defaultdict(set)
    for r in rows["train"]:
        label_sets[r["text_hash"]].add(r["label"])
    conflicts = {h for h, ls in label_sets.items() if len(ls) > 1}
    seen = set()
    clean = []
    excluded = []
    for r in rows["train"]:
        reason = (
            "movie_in_official_test"
            if r["movie_id"] in test_movies
            else (
                "text_in_official_test"
                if r["text_hash"] in test_hashes
                else (
                    "conflicting_training_labels"
                    if r["text_hash"] in conflicts
                    else "duplicate_training_text" if r["text_hash"] in seen else None
                )
            )
        )
        if reason:
            excluded.append(
                {"review_id": r["review_id"], "movie_id": r["movie_id"], "reason": reason}
            )
        else:
            clean.append(r)
            seen.add(r["text_hash"])
    y = np.array([r["label"] for r in clean])
    groups = np.array([r["movie_id"] for r in clean])
    best = None
    for ti, vi in GroupShuffleSplit(n_splits=200, test_size=0.15, random_state=42).split(
        clean, y, groups
    ):
        score = abs(len(vi) / len(clean) - 0.15) + abs(y[vi].mean() - y.mean())
        if best is None or score < best[0]:
            best = (score, ti, vi)
    splits = {
        "train": [clean[i] for i in best[1]],
        "validation": [clean[i] for i in best[2]],
        "test": rows["test"],
    }
    stats = {}
    hashes = {}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        for field in ["movie_id", "text_hash"]:
            assert not ({r[field] for r in splits[a]} & {r[field] for r in splits[b]})
    for split, rs in splits.items():
        p = OUT / f"{split}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        hashes[split] = digest(p)
        stats[split] = {
            "reviews": len(rs),
            "movies": len({r["movie_id"] for r in rs}),
            "labels": dict(collections.Counter(r["label"] for r in rs)),
            "duplicate_text_rows": len(rs) - len({r["text_hash"] for r in rs}),
        }
    report = {
        "scope": "all_50000_labeled_IMDb_reviews_no_genre_or_title_type_filter",
        "unlabeled_reviews_used": False,
        "source_archive_sha256": digest(archive),
        "source_train_reviews": 25000,
        "official_test_preserved": True,
        "excluded_training_rows": excluded,
        "exclusion_counts": dict(collections.Counter(r["reason"] for r in excluded)),
        "splits": stats,
        "split_sha256": hashes,
        "cross_split_movie_overlap": 0,
        "cross_split_text_overlap": 0,
        "validation_selection": "GroupShuffleSplit seed42, 200 candidates, 15% target; balance using training labels only",
        "test_disclosure": "Horror subset of official test was previously evaluated. This is not a never-seen independent holdout.",
    }
    atomic_json(OUT / "manifest.json", report)
    atomic_json(ROOT / "reports/full_imdb_data.json", report)
    print(
        json.dumps({"splits": stats, "excluded": report["exclusion_counts"]}, indent=2), flush=True
    )


if __name__ == "__main__":
    main()
