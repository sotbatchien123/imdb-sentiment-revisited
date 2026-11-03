"""Offline, loopback-only sentiment comparison demo; CPU by default."""

import argparse
import html
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from phase2_core import ROOT, Predictor


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # Do not log entered reviews.

    def allowed(self):
        hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        origin = self.headers.get("Origin")
        return self.headers.get("Host") in hosts and (
            not origin or origin in {"http://" + h for h in hosts}
        )

    def send(self, code, data, mime="application/json; charset=utf-8"):
        body = (
            data
            if isinstance(data, bytes)
            else json.dumps(data, ensure_ascii=False).encode("utf-8")
        )
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.allowed():
            return self.send(403, {"error": "Only local requests are allowed."})
        if self.path == "/health":
            return self.send(
                200,
                {
                    "ready": True,
                    "device": str(self.server.predictor.device),
                    "models": self.server.predictor.selection,
                },
            )
        if self.path == "/api/status":
            path = ROOT / "work_dirs/multichunk_v1/status_live.json"
            try:
                state = (
                    json.loads(path.read_text(encoding="utf-8"))
                    if path.exists()
                    else {"stage": "not_started"}
                )
            except (OSError, json.JSONDecodeError):
                state = {"stage": "status_temporarily_unavailable"}
            return self.send(200, state)
        files = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/style.css": ("style.css", "text/css; charset=utf-8"),
        }
        if self.path not in files:
            return self.send(404, {"error": "Not found"})
        name, mime = files[self.path]
        self.send(200, (ROOT / "demo" / name).read_bytes(), mime)

    def do_POST(self):
        if not self.allowed():
            return self.send(403, {"error": "Only local requests are allowed."})
        if self.path != "/api/predict":
            return self.send(404, {"error": "Not found"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 1 or size > 100000:
                return self.send(413, {"error": "Nội dung quá dài hoặc thiếu dữ liệu."})
            if self.headers.get_content_type() != "application/json":
                return self.send(415, {"error": "Yêu cầu JSON."})
            self.connection.settimeout(30)
            payload = json.loads(self.rfile.read(size))
            text = payload.get("text") if isinstance(payload, dict) else None
            if not isinstance(text, str):
                raise ValueError("Review phải là văn bản.")
            if len(text) > 20000:
                return self.send(413, {"error": "Giới hạn 20.000 ký tự; hãy rút ngắn review."})
            text = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", text)).split())
            if not text:
                raise ValueError("Hãy nhập nội dung review bằng tiếng Anh.")
        except (ValueError, UnicodeError) as exc:
            return self.send(400, {"error": str(exc)})
        if not self.server.inference_lock.acquire(blocking=False):
            return self.send(429, {"error": "Đang xử lý một review khác; vui lòng thử lại."})
        try:
            started = time.time()
            result = self.server.predictor.demo(text)
            result.update(
                seconds=time.time() - started,
                device=str(self.server.predictor.device),
                normalized_text=text,
            )
            self.send(200, result)
        except Exception as exc:
            print(f"Prediction failed: {type(exc).__name__}: {exc}", flush=True)
            self.send(500, {"error": "Không thể dự đoán; xem log của demo."})
        finally:
            self.server.inference_lock.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    predictor = Predictor("cpu")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.predictor = predictor
    server.inference_lock = threading.Lock()
    print(f"Ready: http://127.0.0.1:{args.port} (CPU)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
