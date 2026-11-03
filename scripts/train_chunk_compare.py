"""Plan C runner: B2 (mean pooling) vs C1 (attention pooling) on one identical pipeline. Never reads the test split.

  python scripts/train_chunk_compare.py --smoke --aggregation attention [--device cpu]
  python scripts/train_chunk_compare.py --aggregation mean --seed 42        (requires config mode "frozen")

No gate/deploy side effects: comparison and test evaluation live in evaluate_chunk_compare.py.
"""

import argparse
import hashlib
import json
import msvcrt
import os
import random
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import AutoModel, AutoTokenizer
from phase2_core import ROOT, chunks_from_ids
from chunk_attention_model import (
    accumulation_windows,
    attention_stats,
    build_head,
    forward_reviews,
    shared_init_state,
    state_hash,
)
from train_baselines import save_checkpoint
from runtime_io import atomic_json, publish_status

CONFIG = ROOT / "configs/chunk_compare_v1.json"
SIGNED_SCRIPTS = [
    "chunk_attention_model.py",
    "train_chunk_compare.py",
    "phase2_core.py",
    "train_baselines.py",
    "runtime_io.py",
]
BUCKETS = [("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), (">=5", 5, 10**9)]
STATE = {"stage": "starting", "last_completed_epoch": None, "run_dir": None}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def scores(y, p):
    if not y:
        return None
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "f1_per_class": [
            float(v) for v in f1_score(y, p, average=None, labels=[0, 1], zero_division=0)
        ],
        "confusion_matrix": confusion_matrix(y, p, labels=[0, 1]).tolist(),
        "count": len(y),
    }


def load_split(cfg, tok, split, allow_test=False):
    if split == "test" and not allow_test:
        raise RuntimeError("Test split is locked for this runner")
    rows = [
        json.loads(l)
        for l in (ROOT / cfg["dataset"] / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    ids = tok([r["text"] for r in rows], truncation=False, verbose=False)["input_ids"]
    items = []
    for row, seq in zip(rows, ids):
        chunks = chunks_from_ids(
            seq, tok.cls_token_id, tok.sep_token_id, cfg["content_tokens_per_chunk"]
        )
        assert [t for c in chunks for t in c[1:-1]] == seq[
            1:-1
        ], f"token coverage failed for {row['review_id']}"
        items.append(
            {
                "chunks": chunks,
                "label": row["label"],
                "review_id": row["review_id"],
                "movie_id": row.get("movie_id"),
                "long": len(seq) > 256,
            }
        )
    return items


def signature(cfg):
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode())
    for name in SIGNED_SCRIPTS:
        digest.update((ROOT / "scripts" / name).read_bytes())
    for split in ["train", "validation"]:
        digest.update((ROOT / cfg["dataset"] / f"{split}.jsonl").read_bytes())
    return digest.hexdigest()


def rng_state(order_rng):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "order": order_rng.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state, order_rng):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    order_rng.setstate(state["order"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def seed_run(seed, cfg):
    """Same dropout RNG start for B2/C1; batch order has its own generator so model init never shifts it."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return random.Random(seed + cfg["order_seed_offset"])


def evaluate(encoder, head, items, device, cfg):
    head.eval()
    truth = []
    pred = []
    prob = []
    weights = []
    loss_sum = 0.0
    with torch.no_grad():
        for i in range(0, len(items), cfg["batch_size"]):
            batch = items[i : i + cfg["batch_size"]]
            logits, w = forward_reviews(
                encoder,
                head,
                [x["chunks"] for x in batch],
                device,
                cfg["chunk_microbatch"],
                return_weights=True,
            )
            labels = torch.tensor([x["label"] for x in batch], device=device)
            loss_sum += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            truth.extend(labels.tolist())
            pred.extend(logits.argmax(1).tolist())
            prob.extend(logits.softmax(-1)[:, 1].tolist())
            w = w.cpu().tolist()
            pos = 0
            for x in batch:
                weights.append(w[pos : pos + len(x["chunks"])])
                pos += len(x["chunks"])

    def subset(keep):
        idx = [i for i, x in enumerate(items) if keep(x)]
        return scores([truth[i] for i in idx], [pred[i] for i in idx])

    result = scores(truth, pred)
    result["loss"] = loss_sum / len(items)
    result["long"] = subset(lambda x: x["long"])
    result["short"] = subset(lambda x: not x["long"])
    result["by_chunk_count"] = {
        name: subset(lambda x, a=a, b=b: a <= len(x["chunks"]) <= b) for name, a, b in BUCKETS
    }
    if head.aggregation == "attention":
        flat = torch.tensor([v for w in weights for v in w])
        owners = torch.tensor([r for r, w in enumerate(weights) for _ in w])
        ent, peak = attention_stats(flat, owners, len(weights))
        result["attention"] = {
            "multi_chunk_reviews": len(ent),
            "mean_normalized_entropy": float(np.mean(ent)) if ent else None,
            "mean_max_weight": float(np.mean(peak)) if peak else None,
        }
    predictions = [
        {
            "review_id": x["review_id"],
            "label": x["label"],
            "prediction": p,
            "positive_probability": q,
            "long": x["long"],
            "chunks": len(x["chunks"]),
            "chunk_weights": [round(v, 6) for v in w],
        }
        for x, p, q, w in zip(items, pred, prob, weights)
    ]
    return result, predictions


def load_shared_init(cfg, seed, out):
    """Recompute the seed's shared init and pin it to a file; a later run must match the pinned hash."""
    state = shared_init_state(cfg, seed)
    digest = state_hash(state)
    path = out / f"shared_init_seed{seed}.pt"
    if path.exists():
        pinned = torch.load(path, map_location="cpu", weights_only=False)
        if pinned["hash"] != digest or state_hash(pinned["state"]) != digest:
            raise RuntimeError(f"Shared init hash mismatch for seed {seed}")
    else:
        save_checkpoint(path, {"state": state, "hash": digest, "seed": seed})
    return state, digest


def train_run(cfg, aggregation, seed, device, encoder, data, sig, out):
    run = out / f"{aggregation}_seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    STATE["run_dir"] = run

    def status(**kw):
        publish_status(
            run / "status.json",
            {
                "pid": os.getpid(),
                "updated_at": now(),
                "aggregation": aggregation,
                "seed": seed,
                **kw,
            },
        )

    shared, shared_digest = load_shared_init(cfg, seed, out)
    order_rng = seed_run(seed, cfg)
    head = build_head(cfg, aggregation, seed, shared).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    history = []
    best = -1.0
    first = 1
    if (run / "latest.pt").exists():
        state = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
        if state["signature"] != sig:
            raise RuntimeError(
                "Resume signature mismatch; keep this run unchanged or start a new version"
            )
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"], order_rng)
        history = state["history"]
        best = state["best"]
        first = state["epoch"] + 1
        STATE["last_completed_epoch"] = state["epoch"]
    provenance = {
        "signature": sig,
        "shared_init_hash": shared_digest,
        "aggregation": aggregation,
        "seed": seed,
    }
    train = data["train"]
    n = len(train)
    for epoch in range(first, cfg["max_epochs"] + 1):
        if (out / "PAUSE").exists() or (run / "PAUSE").exists():
            status(stage="paused", completed_epoch=epoch - 1)
            return "paused"
        STATE["stage"] = f"train_epoch_{epoch}"
        head.train()
        order = list(range(n))
        order_rng.shuffle(order)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.time()
        loss_sum = 0.0
        done = 0
        for k, (size, batches) in enumerate(
            accumulation_windows(n, cfg["batch_size"], cfg["gradient_accumulation"])
        ):
            optimizer.zero_grad(set_to_none=True)
            for s, e in batches:
                batch = [train[i] for i in order[s:e]]
                logits = forward_reviews(
                    encoder,
                    head,
                    [x["chunks"] for x in batch],
                    device,
                    cfg["chunk_microbatch"],
                    cfg["activation_checkpointing_head"],
                )
                loss = torch.nn.functional.cross_entropy(
                    logits,
                    torch.tensor([x["label"] for x in batch], device=device),
                    reduction="sum",
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at epoch {epoch}, reviews {s}-{e}")
                loss_sum += loss.item()
                done = e
                (loss / size).backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), cfg["clip_grad_norm"])
            if not torch.isfinite(norm):
                raise RuntimeError(f"Nonfinite gradient norm at epoch {epoch}")
            optimizer.step()
            if k % 10 == 0:
                status(
                    stage="training",
                    epoch=epoch,
                    max_epochs=cfg["max_epochs"],
                    completed_reviews=done,
                    total_reviews=n,
                    train_loss=loss_sum / done,
                    epoch_elapsed_seconds=time.time() - started,
                    best_validation_f1=best,
                )
        train_seconds = time.time() - started
        STATE["stage"] = f"validation_epoch_{epoch}"
        status(stage="validation", epoch=epoch, max_epochs=cfg["max_epochs"])
        metrics, predictions = evaluate(encoder, head, data["validation"], device, cfg)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / n,
            "validation": metrics,
            "train_seconds": train_seconds,
            "seconds": time.time() - started,
        }
        if device.type == "cuda":
            row["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
            row["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        history.append(row)
        if metrics["macro_f1"] > best:  # strict: a tie keeps the earlier epoch
            best = metrics["macro_f1"]
            save_checkpoint(
                run / "best.pt",
                {"head": head.state_dict(), "epoch": epoch, "validation": metrics, **provenance},
            )
            atomic_json(run / "best_validation_predictions.json", predictions)
        save_checkpoint(
            run / "latest.pt",
            {
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best": best,
                "history": history,
                "rng": rng_state(order_rng),
                **provenance,
            },
        )
        atomic_json(run / "history.json", history)
        STATE["last_completed_epoch"] = epoch
        print(
            json.dumps(
                {
                    "aggregation": aggregation,
                    "seed": seed,
                    "epoch": epoch,
                    "train_loss": row["train_loss"],
                    "val_f1": metrics["macro_f1"],
                    "val_long_f1": metrics["long"]["macro_f1"],
                    "seconds": row["seconds"],
                }
            ),
            flush=True,
        )
    chosen = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    result = {
        "aggregation": aggregation,
        "seed": seed,
        "epochs_completed": len(history),
        "max_epochs": cfg["max_epochs"],
        "best_epoch": chosen["epoch"],
        "validation": chosen["validation"],
        "test_evaluated": False,
        **provenance,
    }
    atomic_json(run / "result.json", result)
    status(stage="completed", result=result)
    STATE["stage"] = "completed"
    return "completed"


def load_encoder(cfg, device):
    tok = AutoTokenizer.from_pretrained(ROOT / cfg["encoder"], local_files_only=True)
    encoder = (
        AutoModel.from_pretrained(
            ROOT / cfg["encoder"], local_files_only=True, attn_implementation="sdpa"
        )
        .to(device)
        .eval()
    )
    encoder.requires_grad_(False)
    return tok, encoder


def smoke(cfg, aggregation, device, out):
    """Real-BERT checks on a separate folder; never counted toward the 50 epochs."""
    folder = out / "smoke" / f"{aggregation}_{device.type}"
    folder.mkdir(parents=True, exist_ok=True)
    STATE["run_dir"] = folder
    tok, encoder = load_encoder(cfg, device)
    data = {s: load_split(cfg, tok, s) for s in ["train", "validation"]}
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    shared = shared_init_state(cfg, 42)
    heads = {a: build_head(cfg, a, 42, shared).to(device).eval() for a in ["mean", "attention"]}
    one = [data["train"][0]["chunks"][:1]]
    with torch.no_grad():
        a = forward_reviews(encoder, heads["mean"], one, device)
        b = forward_reviews(encoder, heads["attention"], one, device)
    torch.testing.assert_close(a, b)
    head = heads[aggregation]
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    longest = max(data["train"], key=lambda x: len(x["chunks"]))
    batch = [longest, data["train"][0]]
    labels = torch.tensor([x["label"] for x in batch], device=device)
    order_rng = seed_run(42, cfg)

    def step(model, opt):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = forward_reviews(
            encoder,
            model,
            [x["chunks"] for x in batch],
            device,
            cfg["chunk_microbatch"],
            cfg["activation_checkpointing_head"],
        )
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum") / len(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad_norm"])
        opt.step()
        return loss

    loss = step(head, optimizer)
    assert torch.isfinite(loss)
    grads = {name: p.grad for name, p in head.named_parameters()}
    assert all(
        g is not None and torch.isfinite(g).all() for g in grads.values()
    ), "every head parameter must receive a finite gradient"
    assert all(p.grad is None for p in encoder.parameters()), "encoder must stay frozen"
    save_checkpoint(
        folder / "latest.pt",
        {
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": rng_state(order_rng),
        },
    )
    state = torch.load(folder / "latest.pt", map_location="cpu", weights_only=False)
    restored = build_head(cfg, aggregation, 42, shared).to(device)
    restored.load_state_dict(state["head"])
    opt2 = torch.optim.AdamW(
        restored.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    opt2.load_state_dict(state["optimizer"])
    rng = rng_state(order_rng)
    for model, opt in [(head, optimizer), (restored, opt2)]:
        restore_rng(rng, order_rng)
        step(model, opt)
    for p, q in zip(head.parameters(), restored.parameters()):
        torch.testing.assert_close(p, q)
    longest_val = max(data["validation"], key=lambda x: len(x["chunks"]))
    head.eval()
    with torch.no_grad():
        logits, w = forward_reviews(
            encoder,
            head,
            [longest_val["chunks"]],
            device,
            cfg["chunk_microbatch"],
            return_weights=True,
        )
    assert torch.isfinite(logits).all() and (w >= 0).all() and abs(float(w.sum()) - 1) < 1e-5
    report = {
        "passed": True,
        "aggregation": aggregation,
        "device": device.type,
        "checks": [
            "token coverage train/validation",
            "single-chunk mean==attention (real BERT, eval)",
            "longest train review backward with head checkpointing",
            "finite gradient for every head parameter",
            "frozen encoder",
            "checkpoint+optimizer+RNG restore gives identical next update",
            "longest validation review eval, weights sum to 1",
        ],
        "longest_train_chunks": len(longest["chunks"]),
        "longest_validation_chunks": len(longest_val["chunks"]),
        "validation_weights_longest": [round(float(v), 4) for v in w],
        "seconds": time.time() - started,
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
        ),
        "updated_at": now(),
    }
    atomic_json(folder / "smoke_results.json", report)
    print(json.dumps(report), flush=True)


def main(args):
    cfg = read_json(CONFIG)
    out = ROOT / cfg["output"]
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device(args.device)
    lock = (out / ("smoke.lock" if args.smoke else "runner.lock")).open("a+b")
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise RuntimeError(
            "Another chunk_compare runner holds the lock; run B2/C1 sequentially"
        ) from None
    if args.smoke:
        return smoke(cfg, args.aggregation, device, out)
    if cfg["mode"] != "frozen":
        raise RuntimeError('configs/chunk_compare_v1.json mode must be "frozen" before a main run')
    seed = args.seed if args.seed is not None else cfg["pilot_seed"]
    if seed != cfg["pilot_seed"]:
        if seed not in cfg["extension_seeds"]:
            raise RuntimeError(f"Seed {seed} is not in the protocol")
        gate = ROOT / "reports/chunk_attention_v1/pilot_comparison.json"
        if not gate.exists() or not read_json(gate)["gate"]["passed"]:
            raise RuntimeError("Extension seeds require a passed pilot gate")
    STATE["stage"] = "loading"
    tok, encoder = load_encoder(cfg, device)
    data = {s: load_split(cfg, tok, s) for s in ["train", "validation"]}
    sig = signature(cfg)
    atomic_json(
        out / "tokenization.json",
        {
            s: {
                "reviews": len(v),
                "chunks": sum(len(x["chunks"]) for x in v),
                "max_chunks": max(len(x["chunks"]) for x in v),
                "long_reviews": sum(x["long"] for x in v),
                "discarded_content_tokens": 0,
            }
            for s, v in data.items()
        },
    )
    return train_run(cfg, args.aggregation, seed, device, encoder, data, sig, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregation", required=True, choices=["mean", "attention"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    try:
        main(args)
    except Exception:
        # One record per failure; no automatic restart. Resume later from latest.pt in a fresh process.
        cfg = read_json(CONFIG)
        target = STATE["run_dir"] or ROOT / cfg["output"]
        error = {
            "stage": "failed",
            "failed_stage": STATE["stage"],
            "last_completed_epoch": STATE["last_completed_epoch"],
            "pid": os.getpid(),
            "error": traceback.format_exc(),
            "updated_at": now(),
        }
        atomic_json(Path(target) / "error.json", error)
        publish_status(Path(target) / "status.json", error)
        raise
