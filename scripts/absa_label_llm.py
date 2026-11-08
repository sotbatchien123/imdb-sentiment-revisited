"""Gán nhãn cảm xúc theo khía cạnh (silver labels) cho IMDb bằng một LLM chạy local.

Script gửi từng review tới llama.cpp `llama-server` (API tương thích OpenAI) và nhận về
một nhãn trong {positive, negative, neutral, none} cho mỗi khía cạnh khai báo trong config.

- Đầu ra được ghi nối tiếp vào `<output_dir>/<split>.jsonl`; chạy lại sẽ bỏ qua review đã có
  nhãn, nên có thể dừng và chạy tiếp bất cứ lúc nào.
- Review lỗi (server lỗi, JSON hỏng) được ghi vào `<output_dir>/errors.jsonl` và sẽ được thử
  lại ở lần chạy sau.

Ví dụ:
    python scripts/absa_label_llm.py --splits validation --limit 50
    python scripts/absa_label_llm.py --splits train validation test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def clip_review(text: str, max_chars: int) -> str:
    """Giữ phần đầu và phần cuối khi review quá dài (nhận xét về nhạc, hình ảnh hay ở cuối)."""
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + " [...] " + text[-half:]


def response_schema(aspects: list[str], labels: list[str], evidence: bool = False) -> dict:
    """Schema JSON ép đầu ra. Chế độ evidence: mỗi khía cạnh là {evidence, label}, evidence viết trước."""
    if evidence:
        item = {
            "type": "object",
            "properties": {
                "evidence": {"type": "string", "maxLength": 160},
                "label": {"type": "string", "enum": labels},
            },
            "required": ["evidence", "label"],
            "additionalProperties": False,
        }
    else:
        item = {"type": "string", "enum": labels}
    return {
        "type": "object",
        "properties": {a: item for a in aspects},
        "required": aspects,
        "additionalProperties": False,
    }


def is_grounded(evidence: str, review: str, min_overlap: float = 0.7) -> bool:
    """Câu bằng chứng có thật trong review không: khớp nguyên văn, hoặc >= 70% từ có mặt."""
    ev = re.findall(r"[a-z0-9']+", evidence.lower())
    if not ev:
        return False
    words = re.findall(r"[a-z0-9']+", review.lower())
    if " ".join(ev) in " ".join(words):
        return True
    vocab = set(words)
    return sum(w in vocab for w in ev) / len(ev) >= min_overlap


def parse_evidence_labels(
    content: str, aspects: list[str], labels: list[str], review: str, exempt: list[str]
) -> tuple[dict, dict, list[str]]:
    """Đọc đầu ra chế độ evidence. Nhãn khác none mà không có bằng chứng thật bị ép về none.

    Trả về (nhãn, bằng chứng, danh sách khía cạnh bị ép về none).
    """
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    match = re.search(r"\{.*\}", content, flags=re.S)
    if not match:
        raise ValueError(f"không tìm thấy JSON: {content[:200]!r}")
    data = json.loads(match.group(0))
    out_labels, out_evidence, overridden = {}, {}, []
    for aspect in aspects:
        item = data.get(aspect)
        if not isinstance(item, dict):
            raise ValueError(f"thiếu object cho {aspect}: {item!r}")
        label = str(item.get("label", "")).strip().lower()
        if label not in labels:
            raise ValueError(f"nhãn không hợp lệ cho {aspect}: {label!r}")
        ev = str(item.get("evidence", "")).strip()
        if label != "none" and aspect not in exempt and not is_grounded(ev, review):
            overridden.append(aspect)
            label = "none"
        out_labels[aspect] = label
        out_evidence[aspect] = ev if label != "none" else ""
    return out_labels, out_evidence, overridden


def build_payload(review: str, system_prompt: str, cfg: dict) -> dict:
    gen = cfg["generation"]
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Review:\n" + review},
        ],
        "temperature": gen["temperature"],
        "max_tokens": gen["max_tokens"],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "aspect_labels",
                "schema": response_schema(
                    cfg["aspects"], cfg["labels"], cfg.get("evidence", False)
                ),
            },
        },
        "chat_template_kwargs": {"enable_thinking": gen["enable_thinking"]},
    }


def parse_labels(content: str, aspects: list[str], labels: list[str]) -> dict:
    """Đọc JSON trả về và kiểm tra đủ khía cạnh, đúng tập nhãn. Sai thì báo ValueError."""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    match = re.search(r"\{.*\}", content, flags=re.S)
    if not match:
        raise ValueError(f"không tìm thấy JSON: {content[:200]!r}")
    data = json.loads(match.group(0))
    result = {}
    for aspect in aspects:
        value = str(data.get(aspect, "")).strip().lower()
        if value not in labels:
            raise ValueError(f"nhãn không hợp lệ cho {aspect}: {value!r}")
        result[aspect] = value
    return result


def call_server(url: str, payload: dict, timeout: float = 300.0) -> str:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def read_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["review_id"])
    return done


def label_one(row: dict, system_prompt: str, cfg: dict) -> dict:
    review = clip_review(row["text"], cfg["max_review_chars"])
    payload = build_payload(review, system_prompt, cfg)
    start = time.perf_counter()
    last_error = None
    for _ in range(3):
        try:
            content = call_server(cfg["server"]["url"], payload)
            res = {"ok": True, "seconds": None}
            if cfg.get("evidence"):
                labels, evidence, overridden = parse_evidence_labels(
                    content, cfg["aspects"], cfg["labels"], review, cfg.get("evidence_exempt", [])
                )
                res.update(evidence=evidence, grounding_overrides=overridden)
            else:
                labels = parse_labels(content, cfg["aspects"], cfg["labels"])
            res.update(labels=labels, seconds=time.perf_counter() - start)
            return res
        except (urllib.error.URLError, ValueError, KeyError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return {"ok": False, "error": last_error, "seconds": time.perf_counter() - start}


def run_split(split: str, cfg: dict, system_prompt: str, prompt_sha: str, args) -> None:
    src = ROOT / cfg["source_dataset"] / f"{split}.jsonl"
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.jsonl"
    err_path = out_dir / "errors.jsonl"

    done = read_done_ids(out_path)
    with open(src, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    todo = [r for r in rows if r["review_id"] not in done]
    if args.ids_file:
        wanted = set(json.loads((ROOT / args.ids_file).read_text(encoding="utf-8")))
        todo = [r for r in todo if r["review_id"] in wanted]
    if args.limit:
        todo = todo[: max(0, args.limit - len(done))]
    print(f"[{split}] tổng {len(rows)}, đã có {len(done)}, cần gán {len(todo)}", flush=True)
    if not todo:
        return

    workers = args.workers or cfg["parallel_requests"]
    started = time.perf_counter()
    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=workers) as pool, open(
        out_path, "a", encoding="utf-8"
    ) as out, open(err_path, "a", encoding="utf-8") as err:
        futures = {pool.submit(label_one, r, system_prompt, cfg): r for r in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            row = futures[fut]
            res = fut.result()
            record = {"review_id": row["review_id"], "split": split}
            if res["ok"]:
                record.update(
                    labels=res["labels"],
                    model=cfg["server"]["model_file"],
                    prompt_version=cfg["prompt_version"],
                    prompt_sha256=prompt_sha,
                    seconds=round(res["seconds"], 3),
                )
                for key in ("evidence", "grounding_overrides"):
                    if key in res:
                        record[key] = res[key]
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_ok += 1
            else:
                record.update(error=res["error"])
                err.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_err += 1
            if i % 50 == 0 or i == len(todo):
                out.flush()
                err.flush()
                rate = i / (time.perf_counter() - started)
                eta_h = (len(todo) - i) / rate / 3600 if rate else float("inf")
                print(
                    f"[{split}] {i}/{len(todo)} | lỗi {n_err} | {rate:.2f} review/s"
                    f" | còn khoảng {eta_h:.1f} giờ",
                    flush=True,
                )
    print(f"[{split}] xong: {n_ok} thành công, {n_err} lỗi", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/absa_labeling_v1.json")
    parser.add_argument("--splits", nargs="+", default=["validation"])
    parser.add_argument("--limit", type=int, default=0, help="tối đa số review mỗi split")
    parser.add_argument("--workers", type=int, default=0, help="ghi đè parallel_requests")
    parser.add_argument("--ids-file", help="file JSON danh sách review_id, chỉ gán các review này")
    parser.add_argument("--dry-run", action="store_true", help="in payload mẫu rồi thoát")
    args = parser.parse_args(argv)
    # Console Windows mặc định cp1252 không in được tiếng Việt.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cfg = load_config(ROOT / args.config)
    system_prompt = (ROOT / cfg["prompt_file"]).read_text(encoding="utf-8").strip()
    prompt_sha = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]

    if args.dry_run:
        sample = build_payload("An example review.", system_prompt, cfg)
        print(json.dumps(sample, indent=2, ensure_ascii=False))
        return 0

    for split in args.splits:
        run_split(split, cfg, system_prompt, prompt_sha, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
