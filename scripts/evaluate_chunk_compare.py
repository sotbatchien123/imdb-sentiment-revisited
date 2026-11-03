"""Plan C comparison and locked evaluation for chunk_compare_v1.

python scripts/evaluate_chunk_compare.py compare --seed 42
    Pilot gate C1 vs B2 from finished validation results -> reports/chunk_attention_v1/pilot_comparison.json
python scripts/evaluate_chunk_compare.py evaluate --run attention_seed42 --split validation
python scripts/evaluate_chunk_compare.py evaluate --run attention_seed42 --split test --allow-test
    Test only after method/checkpoint selection is locked on validation; every compared config should be reported.
"""

import argparse
import json
import time
import torch
from phase2_core import ROOT
from chunk_attention_model import ChunkCompareHead
from runtime_io import atomic_json
from train_chunk_compare import CONFIG, evaluate, load_encoder, load_split, read_json, signature

REPORT = ROOT / "reports/chunk_attention_v1"


def compare(cfg, seed):
    out = ROOT / cfg["output"]
    runs = {}
    for agg in ["mean", "attention"]:
        path = out / f"{agg}_seed{seed}/result.json"
        if not path.exists():
            raise RuntimeError(f"Missing finished run: {path.relative_to(ROOT)}")
        runs[agg] = read_json(path)
    if (
        runs["mean"]["signature"] != runs["attention"]["signature"]
        or runs["mean"]["shared_init_hash"] != runs["attention"]["shared_init_hash"]
    ):
        raise RuntimeError("B2/C1 were not run under the same signature and shared init")
    m = runs["mean"]["validation"]
    a = runs["attention"]["validation"]
    rule = cfg["gate"]
    delta = a["macro_f1"] - m["macro_f1"]
    long_delta = a["long"]["macro_f1"] - m["long"]["macro_f1"]
    full = all(r["epochs_completed"] == r["max_epochs"] == cfg["max_epochs"] for r in runs.values())
    checks = {
        "delta_f1": delta >= rule["min_delta_f1_attention_vs_mean"],
        "long_not_lower": long_delta >= 0 if rule["long_f1_not_lower_than_mean"] else True,
        "full_epochs": full if rule["require_full_epochs"] else True,
    }
    ref = cfg["replace_A_reference"]
    report = {
        "seed": seed,
        "mean": runs["mean"],
        "attention": runs["attention"],
        "delta_f1": delta,
        "delta_long_f1": long_delta,
        "gate": {"passed": all(checks.values()), "checks": checks, "rule": rule},
        "replace_A": {
            "attention_meets_overall": a["macro_f1"] >= ref["overall_f1_min"],
            "attention_meets_long": a["long"]["macro_f1"] >= ref["long_f1_min"],
            "reference": ref,
        },
        "test_evaluated": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    atomic_json(REPORT / "pilot_comparison.json", report)
    print(
        json.dumps(
            {
                "gate_passed": report["gate"]["passed"],
                "checks": checks,
                "delta_f1": delta,
                "delta_long_f1": long_delta,
            }
        )
    )


def evaluate_run(cfg, run_name, split, allow_test, device_name):
    if split == "test" and not allow_test:
        raise RuntimeError(
            "Test evaluation needs --allow-test after the protocol is locked on validation"
        )
    if split == "test" and cfg["mode"] != "frozen":
        raise RuntimeError("Config must be frozen before any test evaluation")
    run = ROOT / cfg["output"] / run_name
    best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    if best["signature"] != signature(cfg):
        raise RuntimeError("Checkpoint signature differs from current code/config/data")
    device = torch.device(device_name)
    tok, encoder = load_encoder(cfg, device)
    head = ChunkCompareHead(cfg, best["aggregation"]).to(device)
    head.load_state_dict(best["head"])
    metrics, predictions = evaluate(
        encoder, head, load_split(cfg, tok, split, allow_test=split == "test"), device, cfg
    )
    folder = REPORT / "evaluations" / f"{run_name}_{split}"
    atomic_json(
        folder / "metrics.json",
        {
            "run": run_name,
            "split": split,
            "best_epoch": best["epoch"],
            "metrics": metrics,
            "signature": best["signature"],
            "device": device_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    atomic_json(folder / "predictions.json", predictions)
    print(
        json.dumps(
            {
                "run": run_name,
                "split": split,
                "macro_f1": metrics["macro_f1"],
                "long_f1": metrics["long"]["macro_f1"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--seed", type=int, default=42)
    e = sub.add_parser("evaluate")
    e.add_argument("--run", required=True)
    e.add_argument("--split", required=True, choices=["validation", "test"])
    e.add_argument("--allow-test", action="store_true")
    e.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    cfg = read_json(CONFIG)
    if args.command == "compare":
        compare(cfg, args.seed)
    else:
        evaluate_run(cfg, args.run, args.split, args.allow_test, args.device)
