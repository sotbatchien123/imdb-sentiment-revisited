"""Demo web ABSA chạy cục bộ: nhập một review tiếng Anh, xem cảm xúc theo 6 khía cạnh.

Nạp checkpoint best.pt của RoBERTa-large và BERT-base fine-tune. Với mỗi mô hình, chọn seed có
Macro-F1 validation cao nhất (không nhìn tập test). Chỉ nghe ở 127.0.0.1, không ghi log nội dung review.

    .venv-train\\Scripts\\python.exe scripts\\absa_demo_server.py            (CPU, cổng 8766)
    .venv-train\\Scripts\\python.exe scripts\\absa_demo_server.py --device cuda
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from absa_core import LABELS, FinetuneABSA, head_tail

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "demo_absa"
SYSTEMS = [
    ("roberta_large", "RoBERTa-large", "configs/absa_roberta_large_v1.json"),
    ("bert_base", "BERT-base", "configs/absa_bert_base_v1.json"),
]
MAX_CHARS = 20000


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def normalize(text: str) -> str:
    """Bỏ thẻ HTML (IMDb dùng <br />), giải mã thực thể, gộp khoảng trắng."""
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", text)).split())


def pick_seed(cfg: dict) -> tuple[int, dict]:
    """Seed có Macro-F1 validation tốt nhất trong các seed đã chạy xong."""
    best = None
    for seed in cfg["seeds"]:
        path = resolve(cfg["output"]) / f"seed{seed}" / "result.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        score = r["best_validation"]["macro_f1"]
        if best is None or score > best[1]["validation_macro_f1"]:
            best = (seed, {"seed": seed, "epoch": r["best_epoch"], "validation_macro_f1": score})
    if best is None:
        raise FileNotFoundError(f"Chưa có result.json nào trong {cfg['output']}")
    return best


class ABSAModel:
    def __init__(self, key: str, name: str, config_path: str, device: str):
        cfg = json.loads(resolve(config_path).read_text(encoding="utf-8"))
        if cfg["model_type"] != "finetune":
            raise ValueError("Demo chỉ hỗ trợ mô hình fine-tune")
        self.key, self.name, self.cfg, self.device = key, name, cfg, device
        self.aspects = cfg["aspects"]
        seed, self.selection = pick_seed(cfg)
        encoder_dir = resolve(cfg["encoder"])
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_dir, local_files_only=True)
        encoder = AutoModel.from_config(
            AutoConfig.from_pretrained(encoder_dir, local_files_only=True),
            attn_implementation="sdpa",
        )
        self.model = FinetuneABSA(encoder, len(self.aspects), cfg.get("dropout", 0.1))
        ckpt = resolve(cfg["checkpoint_dir"]) / f"seed{seed}" / "best.pt"
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        self.model.to(device).eval()

    def encode(self, text: str) -> tuple[list[int], int]:
        ids = self.tokenizer(text, add_special_tokens=True, truncation=False, verbose=False)[
            "input_ids"
        ]
        return head_tail(ids, self.cfg["head_tokens"], self.cfg["tail_tokens"]), len(ids)

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        ids, full_len = self.encode(text)
        x = torch.tensor([ids], device=self.device)
        probs = torch.softmax(self.model(x, torch.ones_like(x)).float()[0], dim=-1).cpu()
        return {
            "aspects": {
                a: {
                    "label": LABELS[int(probs[i].argmax())],
                    "probabilities": {
                        lab: round(float(probs[i, k]), 4) for k, lab in enumerate(LABELS)
                    },
                }
                for i, a in enumerate(self.aspects)
            },
            "token_count": full_len,
            "truncated": full_len > len(ids),
            "kept_tokens": len(ids),
        }


class Handler(BaseHTTPRequestHandler):
    FILES = {
        "/": ("index.html", "text/html; charset=utf-8"),
        "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        "/style.css": ("style.css", "text/css; charset=utf-8"),
    }

    def log_message(self, *args):
        pass  # không ghi review của người dùng vào log

    def allowed(self) -> bool:
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        origin = self.headers.get("Origin")
        return self.headers.get("Host") in hosts and (
            not origin or origin in {"http://" + h for h in hosts}
        )

    def send(self, code: int, data, mime: str = "application/json; charset=utf-8"):
        body = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; "
            "frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.allowed():
            return self.send(403, {"error": "Chỉ nhận yêu cầu từ máy này."})
        if self.path == "/health":
            return self.send(
                200,
                {
                    "ready": True,
                    "device": self.server.device,
                    "aspects": self.server.models[0].aspects,
                    "models": [
                        {"key": m.key, "name": m.name, **m.selection} for m in self.server.models
                    ],
                },
            )
        if self.path not in self.FILES:
            return self.send(404, {"error": "Không tìm thấy."})
        name, mime = self.FILES[self.path]
        self.send(200, (STATIC / name).read_bytes(), mime)

    def do_POST(self):
        if not self.allowed():
            return self.send(403, {"error": "Chỉ nhận yêu cầu từ máy này."})
        if self.path != "/api/predict":
            return self.send(404, {"error": "Không tìm thấy."})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 1 or size > 100000:
                return self.send(413, {"error": "Nội dung quá dài hoặc thiếu dữ liệu."})
            if self.headers.get_content_type() != "application/json":
                return self.send(415, {"error": "Yêu cầu phải là JSON."})
            self.connection.settimeout(30)
            payload = json.loads(self.rfile.read(size))
            text = payload.get("text") if isinstance(payload, dict) else None
            if not isinstance(text, str):
                raise ValueError("Review phải là văn bản.")
            if len(text) > MAX_CHARS:
                return self.send(413, {"error": "Giới hạn 20.000 ký tự; hãy rút ngắn review."})
            text = normalize(text)
            if not text:
                raise ValueError("Hãy nhập nội dung review bằng tiếng Anh.")
        except (ValueError, UnicodeError) as exc:
            return self.send(400, {"error": str(exc)})
        if not self.server.lock.acquire(blocking=False):
            return self.send(429, {"error": "Đang xử lý một review khác; thử lại sau giây lát."})
        try:
            started = time.time()
            results = {}
            for m in self.server.models:
                t = time.time()
                results[m.key] = {**m.predict(text), "seconds": round(time.time() - t, 3)}
            self.send(
                200,
                {
                    "aspects": self.server.models[0].aspects,
                    "models": [{"key": m.key, "name": m.name} for m in self.server.models],
                    "results": results,
                    "seconds": round(time.time() - started, 3),
                    "device": self.server.device,
                },
            )
        except Exception as exc:  # noqa: BLE001 - trả lỗi chung, chi tiết in ra console
            print(f"Lỗi dự đoán: {type(exc).__name__}: {exc}", flush=True)
            self.send(500, {"error": "Không dự đoán được; xem cửa sổ chạy demo."})
        finally:
            self.server.lock.release()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--threads", type=int, default=0, help="số luồng CPU cho PyTorch (0 = mặc định)"
    )
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # console Windows mặc định cp1252
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("Không thấy GPU CUDA")
    started = time.time()
    models = []
    for key, name, cfg in SYSTEMS:
        print(f"Đang nạp {name}…", flush=True)
        models.append(ABSAModel(key, name, cfg, args.device))
        print(
            f"  seed {models[-1].selection['seed']}, epoch {models[-1].selection['epoch']}",
            flush=True,
        )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.models, server.device, server.lock = models, args.device, threading.Lock()
    print(
        f"Sẵn sàng sau {time.time() - started:.0f} giây: http://127.0.0.1:{args.port}", flush=True
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
