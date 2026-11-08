"""Huấn luyện mô hình ABSA 7 khía cạnh × 4 nhãn trên nhãn silver của IMDb.

  python scripts/train_absa.py --config configs/absa_bert_base_v1.json --smoke
  python scripts/train_absa.py --config configs/absa_bert_base_v1.json            (cả 3 seed)
  python scripts/train_absa.py --config configs/absa_roberta_large_v1.json --seed 42

`model_type` trong config: `finetune` (BERT-base, RoBERTa-large) hoặc `frozen_bilstm`.
Chọn epoch theo Macro-F1 trung bình 7 khía cạnh trên validation; test chỉ chạy một lần trên
checkpoint tốt nhất. Có thể dừng bằng cách tạo file PAUSE trong thư mục output (kiểm tra ở
ranh giới epoch) và chạy lại để tiếp tục.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from absa_core import (
    LABELS,
    FinetuneABSA,
    FrozenBiLSTMABSA,
    aspect_metrics,
    class_weights,
    head_tail,
    load_silver,
    multi_aspect_loss,
)
from runtime_io import atomic_json, publish_status

ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def save_checkpoint(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.25)


def seed_all(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast(device: str):
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


# ---------------------------------------------------------------- dữ liệu


def load_split(cfg: dict, tokenizer, split: str, limit: int | None = None):
    rows = [
        json.loads(line)
        for line in (resolve(cfg["dataset"]) / f"{split}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    silver = load_silver(resolve(cfg["labels_dir"]) / f"{split}.jsonl", cfg["aspects"])
    missing = [r["review_id"] for r in rows if r["review_id"] not in silver]
    rows = [r for r in rows if r["review_id"] in silver]
    if limit:
        rows = rows[:limit]
    if missing and not limit and len(missing) > cfg.get("max_missing_labels", 0):
        raise RuntimeError(f"{split}: thiếu nhãn silver cho {len(missing)} review")
    texts = [r["text"].replace("<br />", " ") for r in rows]
    encoded = tokenizer(texts, add_special_tokens=True, truncation=False, verbose=False)[
        "input_ids"
    ]
    seqs = []
    for r, ids in zip(rows, encoded):
        r["exceeds_max_length"] = len(ids) > cfg["max_length"]
        ids = head_tail(ids, cfg["head_tokens"], cfg["tail_tokens"])
        assert len(ids) <= cfg["max_length"]
        seqs.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
    batch = tokenizer.pad(
        seqs, padding="max_length", max_length=cfg["max_length"], return_tensors="pt"
    )
    labels = torch.tensor([silver[r["review_id"]] for r in rows], dtype=torch.long)
    stats = {
        "reviews": len(rows),
        "missing_labels": len(missing),
        "truncated_reviews": int(sum(r["exceeds_max_length"] for r in rows)),
    }
    return TensorDataset(batch["input_ids"], batch["attention_mask"], labels), rows, stats


# ---------------------------------------------------------------- mô hình


def build_model(cfg: dict) -> nn.Module:
    encoder = AutoModel.from_pretrained(
        resolve(cfg["encoder"]), local_files_only=True, attn_implementation="sdpa"
    )
    if cfg["model_type"] == "finetune":
        if cfg.get("gradient_checkpointing"):
            encoder.gradient_checkpointing_enable()
        return FinetuneABSA(encoder, len(cfg["aspects"]), cfg.get("dropout", 0.1))
    if cfg["model_type"] == "frozen_bilstm":
        return FrozenBiLSTMABSA(
            encoder, len(cfg["aspects"]), cfg.get("lstm_hidden_size", 768), cfg.get("dropout", 0.5)
        )
    raise ValueError(f"model_type không hỗ trợ: {cfg['model_type']}")


def build_optimizer(cfg: dict, model: nn.Module):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {
            "params": [p for n, p in params if not any(k in n for k in no_decay)],
            "weight_decay": cfg["weight_decay"],
        },
        {"params": [p for n, p in params if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=cfg["learning_rate"],
        eps=cfg.get("adam_epsilon", 1e-8),
        foreach=cfg.get("optimizer_foreach"),
    )


def predict(model, dataset, batch_size, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad(), autocast(device):
        for ids, mask, labels in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            logits = model(ids.to(device), mask.to(device)).float()
            ys.append(labels.numpy())
            ps.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


# ---------------------------------------------------------------- huấn luyện


def run_seed(cfg, seed, data, sig, smoke, device):
    tag = "smoke" if smoke else ""
    run = resolve(cfg["output"]) / tag / f"seed{seed}"
    ckpt = resolve(cfg.get("checkpoint_dir", cfg["output"])) / tag / f"seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
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

    aspects = cfg["aspects"]
    train, val = data["train"][0], data["validation"][0]
    seed_all(seed)
    model = build_model(cfg).to(device)
    optimizer = build_optimizer(cfg, model)
    weights = class_weights(train.tensors[2].numpy(), cfg.get("class_weighting", "none"))
    accum, micro = cfg["gradient_accumulation"], cfg["micro_batch_size"]
    epochs = 1 if smoke else cfg["epochs"]
    total = -(-len(train) // (micro * accum)) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg["warmup_fraction"] * total), total
    )
    order = torch.Generator().manual_seed(seed)
    history, first, best = [], 0, -1.0

    if (ckpt / "latest.pt").exists():
        st = torch.load(ckpt / "latest.pt", map_location="cpu", weights_only=False)
        if st["signature"] != sig:
            raise RuntimeError(f"Signature không khớp cho {run}; hãy tạo version mới")
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        order.set_state(st["order"])
        torch.set_rng_state(st["torch"])
        if device == "cuda":
            torch.cuda.set_rng_state(st["cuda"])
        history, first, best = st["history"], st["epoch"], st["best"]
        del st

    for epoch in range(first, epochs):
        if (resolve(cfg["output"]) / "PAUSE").exists():
            status(stage="paused", completed_epoch=epoch)
            return None
        model.train()
        started, loss_sum, seen, last = time.time(), 0.0, 0, 0.0
        loader = DataLoader(train, batch_size=micro, shuffle=True, generator=order)
        steps = len(loader)
        optimizer.zero_grad(set_to_none=True)
        for step, (ids, mask, labels) in enumerate(loader):
            window = min(micro * accum, len(train) - (step // accum) * micro * accum)
            with autocast(device):
                logits = model(ids.to(device), mask.to(device))
            loss = multi_aspect_loss(logits, labels.to(device), weights)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss không hữu hạn ở epoch {epoch + 1} bước {step}")
            (loss / (window * len(aspects))).backward()
            if (step + 1) % accum == 0 or step + 1 == steps:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg["max_grad_norm"]
                )
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
                    train_loss=loss_sum / (seen * len(aspects)),
                    epoch_eta_seconds=el / (step + 1) * (steps - step - 1),
                    gpu_peak_mib=(
                        (torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else None
                    ),
                    best_validation_macro_f1=best,
                )
                last = time.time()

        status(stage="validation", epoch=epoch + 1)
        y, p = predict(model, val, cfg["eval_batch_size"], device)
        v = aspect_metrics(y, p, aspects)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": loss_sum / (seen * len(aspects)),
                "validation": v["mean"],
                "validation_per_aspect_macro_f1": {
                    a: v["per_aspect"][a]["macro_f1"] for a in aspects
                },
                "seconds": time.time() - started,
            }
        )
        if v["mean"]["macro_f1"] > best:
            best = v["mean"]["macro_f1"]
            save_checkpoint(
                ckpt / "best.pt",
                {
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "signature": sig,
                    "validation": v,
                },
            )
        save_checkpoint(
            ckpt / "latest.pt",
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "order": order.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state() if device == "cuda" else None,
                "history": history,
                "epoch": epoch + 1,
                "best": best,
                "signature": sig,
            },
        )
        atomic_json(run / "history.json", history)
        print(
            json.dumps({"seed": seed, **history[-1]["validation"], "epoch": epoch + 1}), flush=True
        )

    ck = torch.load(ckpt / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    if smoke:
        report = {
            "passed": True,
            "history": history,
            "train_examples": len(train),
            "peak_allocated_mib": (
                (torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else None
            ),
            "seconds_per_train_example": history[-1]["seconds"] / len(train),
        }
        atomic_json(run / "smoke_results.json", report)
        return report

    status(stage="final_test", best_epoch=ck["epoch"])
    test_set, rows = data["test"][0], data["test"][1]
    y, p = predict(model, test_set, cfg["eval_batch_size"], device)
    metrics = aspect_metrics(y, p, aspects)
    long_mask = np.array([r["exceeds_max_length"] for r in rows])
    groups = {
        g: aspect_metrics(y[m], p[m], aspects)["mean"]
        for g, m in (("short", ~long_mask), ("long", long_mask))
        if m.any()
    }
    # Đối chiếu overall dự đoán với nhãn sao gốc của IMDb (nhãn người thật, độc lập với LLM).
    overall_vs_rating = None
    if "overall" in aspects:
        k = aspects.index("overall")
        gold = np.array([r["label"] for r in rows])
        pred_bin = np.where(
            p[:, k] == LABELS.index("positive"),
            1,
            np.where(p[:, k] == LABELS.index("negative"), 0, -1),
        )
        decided = pred_bin >= 0
        overall_vs_rating = {
            "decided_fraction": float(decided.mean()),
            "accuracy_on_decided": (
                float((pred_bin[decided] == gold[decided]).mean()) if decided.any() else None
            ),
        }
    with (run / "test_predictions.jsonl").open("w", encoding="utf-8") as fh:
        for r, yy, pp in zip(rows, y, p):
            fh.write(
                json.dumps(
                    {
                        "review_id": r["review_id"],
                        "movie_id": r["movie_id"],
                        "silver": {a: LABELS[i] for a, i in zip(aspects, yy)},
                        "prediction": {a: LABELS[i] for a, i in zip(aspects, pp)},
                    }
                )
                + "\n"
            )
    result = {
        "seed": seed,
        "best_epoch": ck["epoch"],
        "best_validation": ck["validation"]["mean"],
        "history": history,
        "test": metrics,
        "test_length_groups": groups,
        "overall_vs_imdb_rating": overall_vs_rating,
        "label_source": "silver (LLM) — chưa phải nhãn người",
        "training_seconds": sum(h["seconds"] for h in history),
        "signature": sig,
    }
    atomic_json(run / "result.json", result)
    status(stage="complete", best_epoch=ck["epoch"], test_mean=metrics["mean"])
    return result


def aggregate(cfg: dict) -> dict:
    results = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(resolve(cfg["output"]).glob("seed*/result.json"))
    ]
    summary = {"seeds": [r["seed"] for r in results]}
    for key in ("macro_f1", "detection_f1", "sentiment_macro_f1"):
        vals = [r["test"]["mean"][key] for r in results]
        summary[key] = {
            "mean": float(np.mean(vals)) if vals else None,
            "sample_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
        }
    summary["per_aspect_macro_f1"] = (
        {
            a: float(np.mean([r["test"]["per_aspect"][a]["macro_f1"] for r in results]))
            for a in cfg["aspects"]
        }
        if results
        else {}
    )
    atomic_json(
        resolve(cfg["report_dir"]) / "results.json",
        {
            "runs": [{k: v for k, v in r.items() if k != "history"} for r in results],
            "aggregate": summary,
        },
    )
    return summary


def signature(cfg: dict) -> str:
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode())
    for name in ("train_absa.py", "absa_core.py"):
        h.update((ROOT / "scripts" / name).read_bytes())
    for split in ("train", "validation", "test"):
        f = resolve(cfg["labels_dir"]) / f"{split}.jsonl"
        if f.exists():
            h.update(f.read_bytes())
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--smoke", action="store_true", help="64 review, 1 epoch, thư mục smoke/")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cfg = json.loads(resolve(args.config).read_text(encoding="utf-8"))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Cần CUDA (hoặc dùng --device cpu cho smoke)")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    tokenizer = AutoTokenizer.from_pretrained(resolve(cfg["encoder"]), local_files_only=True)
    sig = signature(cfg)
    splits = ["train", "validation"] if args.smoke else ["train", "validation", "test"]
    limit = cfg.get("smoke_examples", 64) if args.smoke else None
    data = {s: load_split(cfg, tokenizer, s, limit) for s in splits}
    out = resolve(cfg["output"]) / ("smoke" if args.smoke else "")
    atomic_json(out / "tokenization.json", {s: d[2] for s, d in data.items()})

    for seed in [args.seed] if args.seed else cfg["seeds"]:
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        result = run_seed(cfg, seed, data, sig, args.smoke, args.device)
        if result is None:
            return 0
        # Model/optimizer của seed trước có thể còn kẹt trong tham chiếu vòng (gradient
        # checkpointing): phải gc trước empty_cache, nếu không RoBERTa-large tràn VRAM ở seed sau.
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
        if args.smoke:
            print(json.dumps({k: v for k, v in result.items() if k != "history"}), flush=True)
            return 0
        aggregate(cfg)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        raise
