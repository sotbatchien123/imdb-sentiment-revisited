"""Direction B: full fine-tuning of BERT-base on head+tail 512-token reviews (clean IMDb split).

  python scripts/train_finetune_headtail.py --smoke        (64 train reviews, 1 epoch, separate smoke/ dir)
  python scripts/train_finetune_headtail.py                (all seeds; resumable at epoch boundaries)
  python scripts/train_finetune_headtail.py --seed 42
  python scripts/train_finetune_headtail.py --config configs/finetune_roberta_large_v1.json   (direction D)

Best epoch is chosen on validation Macro-F1; test is evaluated once on that checkpoint.
Pause: create work_dirs/finetune_headtail_v1/PAUSE (honoured at epoch boundaries).
"""

import argparse
import hashlib
import json
import msvcrt
import os
import time
import traceback
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from train_baselines import ROOT, save_checkpoint, seed_all
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from runtime_io import atomic_json, publish_status

DEFAULT_CONFIG = "configs/finetune_headtail_v1.json"


def head_tail(ids, head, tail, cls_id, sep_id):
    content = ids[1:-1]
    if len(content) <= head + tail:
        return ids
    return [cls_id] + content[:head] + content[-tail:] + [sep_id]


def load_split(cfg, tokenizer, split, limit=None):
    rows = [
        json.loads(l)
        for l in (ROOT / cfg["dataset"] / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if limit:
        rows = rows[:limit]
    all_ids = tokenizer(
        [r["text"] for r in rows], add_special_tokens=True, truncation=False, verbose=False
    )["input_ids"]
    seqs = []
    for r, ids in zip(rows, all_ids):
        r["exceeds_max_length"] = len(ids) > cfg["max_length"]
        ids = head_tail(
            ids,
            cfg["head_tokens"],
            cfg["tail_tokens"],
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
        )
        assert len(ids) <= cfg["max_length"]
        seqs.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
    batch = tokenizer.pad(
        seqs, padding="max_length", max_length=cfg["max_length"], return_tensors="pt"
    )
    stats = {"reviews": len(rows), "truncated_reviews": sum(r["exceeds_max_length"] for r in rows)}
    return (
        TensorDataset(
            batch["input_ids"], batch["attention_mask"], torch.tensor([r["label"] for r in rows])
        ),
        rows,
        stats,
    )


def evaluate(model, dataset, batch_size):
    model.eval()
    y = []
    pred = []
    loss = 0.0
    started = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ids, mask, labels in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits.float()
            loss += nn.functional.cross_entropy(logits, labels.cuda(), reduction="sum").item()
            y.extend(labels.tolist())
            pred.extend(logits.argmax(1).cpu().tolist())
    return {
        "loss": loss / len(y),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "examples": len(y),
        "seconds": time.time() - started,
    }, pred


def scores(y, p):
    return {
        "examples": len(y),
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
    }


def run_seed(cfg, seed, data, out, sig, smoke):
    run = out / ("smoke" if smoke else "") / f"seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    if (run / "result.json").exists() and not smoke:
        return json.loads((run / "result.json").read_text(encoding="utf-8"))

    def status(**kw):
        publish_status(
            run / "status.json",
            {
                "pid": os.getpid(),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "seed": seed,
                **kw,
            },
        )

    train, val = data["train"][0], data["validation"][0]
    seed_all(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        ROOT / cfg["encoder"], num_labels=2, local_files_only=True, attn_implementation="sdpa"
    ).cuda()
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    decay = [
        p
        for n, p in model.named_parameters()
        if not any(k in n for k in ("bias", "LayerNorm.weight"))
    ]
    no_decay = [
        p for n, p in model.named_parameters() if any(k in n for k in ("bias", "LayerNorm.weight"))
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg["learning_rate"],
        eps=cfg["adam_epsilon"],
        foreach=cfg.get("optimizer_foreach"),
    )  # False: per-tensor update, lower peak VRAM
    accum = cfg["gradient_accumulation"]
    micro = cfg["micro_batch_size"]
    epochs = 1 if smoke else cfg["epochs"]
    updates_per_epoch = -(-len(train) // (micro * accum))
    total = updates_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg["warmup_fraction"] * total), total
    )
    order = torch.Generator().manual_seed(seed)
    history = []
    first = 0
    best = -1.0
    if (run / "latest.pt").exists():
        st = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
        if st["signature"] != sig:
            raise RuntimeError(f"Signature mismatch for {run.name}; start a new version instead")
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        order.set_state(st["order"])
        torch.set_rng_state(st["torch"])
        torch.cuda.set_rng_state(st["cuda"])
        history = st["history"]
        first = st["epoch"]
        best = st["best"]
        del st
    for epoch in range(first, epochs):
        if (out / "PAUSE").exists():
            status(stage="paused", completed_epoch=epoch)
            return None
        model.train()
        started = time.time()
        loss_sum = 0.0
        seen = 0
        last = 0.0
        loader = DataLoader(train, batch_size=micro, shuffle=True, generator=order)
        steps = len(loader)
        optimizer.zero_grad(set_to_none=True)
        for step, (ids, mask, labels) in enumerate(loader):
            # Normalise by the real size of each accumulation window so the short last window is not down-weighted.
            window = min(micro * accum, len(train) - (step // accum) * micro * accum)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits.float()
            loss = nn.functional.cross_entropy(logits, labels.cuda(), reduction="sum")
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at epoch {epoch+1} step {step}")
            (loss / window).backward()
            if (step + 1) % accum == 0 or step + 1 == steps:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += loss.item()
            seen += len(labels)
            if time.time() - last >= 10 or step + 1 == steps:
                el = time.time() - started
                status(
                    stage="training",
                    epoch=epoch + 1,
                    epochs=epochs,
                    step=step + 1,
                    steps=steps,
                    train_loss=loss_sum / seen,
                    lr=scheduler.get_last_lr()[0],
                    epoch_eta_seconds=el / (step + 1) * (steps - step - 1),
                    gpu_peak_mib=torch.cuda.max_memory_allocated() / 2**20,
                    best_validation_macro_f1=best,
                )
                last = time.time()
        status(stage="validation", epoch=epoch + 1)
        v, _ = evaluate(model, val, cfg["eval_batch_size"])
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": loss_sum / seen,
                "validation": v,
                "seconds": time.time() - started,
            }
        )
        if v["macro_f1"] > best:
            best = v["macro_f1"]
            save_checkpoint(
                run / "best.pt",
                {
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "signature": sig,
                    "validation": v,
                },
            )
        save_checkpoint(
            run / "latest.pt",
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "order": order.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(),
                "history": history,
                "epoch": epoch + 1,
                "best": best,
                "signature": sig,
            },
        )
        atomic_json(run / "history.json", history)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "epoch": epoch + 1,
                    "train_loss": loss_sum / seen,
                    "val_f1": v["macro_f1"],
                    "val_acc": v["accuracy"],
                    "seconds": history[-1]["seconds"],
                }
            ),
            flush=True,
        )
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    if smoke:
        report = {
            "passed": True,
            "history": history,
            "train_examples": len(train),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "seconds_per_train_example": history[-1]["seconds"] / len(train),
        }
        atomic_json(run / "smoke_results.json", report)
        return report
    status(stage="final_test", best_epoch=ck["epoch"])
    test, rows = data["test"][0], data["test"][1]
    metrics, pred = evaluate(model, test, cfg["eval_batch_size"])
    labels = [r["label"] for r in rows]
    groups = {
        g: scores(
            [labels[i] for i, r in enumerate(rows) if r["exceeds_max_length"] == f],
            [pred[i] for i, r in enumerate(rows) if r["exceeds_max_length"] == f],
        )
        for g, f in [("short", False), ("long", True)]
    }
    with (run / "test_predictions.jsonl").open("w", encoding="utf-8") as fh:
        for r, y in zip(rows, pred):
            fh.write(
                json.dumps(
                    {
                        "review_id": r["review_id"],
                        "movie_id": r["movie_id"],
                        "label": r["label"],
                        "prediction": y,
                    }
                )
                + "\n"
            )
    result = {
        "seed": seed,
        "best_epoch": ck["epoch"],
        "best_validation": ck["validation"],
        "history": history,
        "test": {
            **{k: metrics[k] for k in ("accuracy", "macro_f1", "confusion_matrix", "loss")},
            "length_groups": groups,
        },
        "training_seconds": sum(h["seconds"] for h in history),
        "signature": sig,
    }
    atomic_json(run / "result.json", result)
    status(stage="complete", test=result["test"], best_epoch=ck["epoch"])
    return result


def aggregate(cfg):
    results = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((ROOT / cfg["output"]).glob("seed*/result.json"))
    ]
    summary = (
        {
            m: {
                "mean": float(np.mean([r["test"][m] for r in results])),
                "sample_std": (
                    float(np.std([r["test"][m] for r in results], ddof=1))
                    if len(results) > 1
                    else None
                ),
            }
            for m in ["accuracy", "macro_f1"]
        }
        if results
        else {}
    )
    summary["seeds"] = [r["seed"] for r in results]
    atomic_json(
        ROOT / cfg["report_dir"] / "results.json",
        {
            "runs": [{k: v for k, v in r.items() if k != "history"} for r in results],
            "aggregate": summary,
            "references_macro_f1": {
                "reproduction_code_variant": 0.8941,
                "frozen_bilstm_headtail": 0.9187,
                "frozen_bilstm_head": 0.9014,
                "finetune_bert_headtail512": 0.9438,
            },
        },
    )
    return summary


def main(args):
    cfg = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    out = ROOT / cfg["output"]
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ("smoke.lock" if args.smoke else "runner.lock")).open("a+b")
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise RuntimeError("Another fine-tune runner holds the lock") from None
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    tokenizer = AutoTokenizer.from_pretrained(ROOT / cfg["encoder"], local_files_only=True)
    sig = hashlib.sha256(
        (
            json.dumps(cfg, sort_keys=True)
            + (ROOT / "scripts/train_finetune_headtail.py").read_text(encoding="utf-8")
            + (ROOT / cfg["dataset"] / "manifest.json").read_text(encoding="utf-8")
        ).encode()
    ).hexdigest()
    limits = {"train": 64, "validation": 64} if args.smoke else {}
    data = {
        s: load_split(cfg, tokenizer, s, limits.get(s))
        for s in (["train", "validation"] if args.smoke else ["train", "validation", "test"])
    }
    atomic_json(
        out / ("smoke" if args.smoke else "") / "tokenization.json",
        {s: d[2] for s, d in data.items()},
    )
    for seed in ([args.seed] if args.seed else cfg["seeds"]):
        torch.cuda.reset_peak_memory_stats()
        r = run_seed(cfg, seed, data, out, sig, args.smoke)
        if r is None:
            return
        torch.cuda.empty_cache()
        if args.smoke:
            print(json.dumps(r), flush=True)
            return
        aggregate(cfg)
    if not args.seed:
        publish_status(
            out / "status.json",
            {"stage": "complete", "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    args = p.parse_args()
    try:
        main(args)
    except Exception:
        cfg = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
        publish_status(
            ROOT / cfg["output"] / ("smoke_status.json" if args.smoke else "status.json"),
            {
                "stage": "failed",
                "pid": os.getpid(),
                "error": traceback.format_exc(),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
        raise
