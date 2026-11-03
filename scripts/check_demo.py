"""Integration checks against a running local demo (no test-set access)."""

import json
import urllib.request
import urllib.error
from phase2_core import ROOT
from runtime_io import atomic_json

BASE = "http://127.0.0.1:8765"


def request(path, payload=None, headers=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json", **(headers or {})}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def main():
    checks = []
    code, state = request("/health")
    assert code == 200 and state["ready"] and state["device"] == "cpu"
    checks.append("CPU health and real checkpoint metadata")
    for text in ["", "  <br>  "]:
        code, _ = request("/api/predict", {"text": text})
        assert code == 400
    checks.append("empty and HTML-only rejected")
    code, _ = request("/api/predict", {"text": "x" * 20001})
    assert code == 413
    checks.append("oversized input rejected without truncation")
    code, _ = request("/api/predict", {"text": None})
    assert code == 400
    checks.append("invalid payload rejected")
    code, _ = request("/api/predict", {"text": "A great movie."}, {"Origin": "https://example.com"})
    assert code == 403
    checks.append("foreign origin rejected")
    code, _ = request("/../configs/full_imdb.json")
    assert code == 404
    checks.append("arbitrary file path rejected")
    records = []
    for name, text in [
        ("positive", "A wonderful film with brilliant performances. I loved every minute."),
        (
            "negative",
            "A terrible film. The acting is awful and the plot is boring. A waste of time.",
        ),
        (
            "long",
            "The scenery is beautiful and the music sounds nice. " * 100
            + "However, the story is incoherent and I hated the ending.",
        ),
        ("html", "<b>A great movie</b> &amp; a moving story."),
    ]:
        code, result = request("/api/predict", {"text": text})
        assert code == 200
        assert set(result["predictions"]) >= {"bilstm", "bilstm_headtail"}
        for kind in ["bilstm", "bilstm_headtail"]:
            p = result["predictions"][kind]
            visible = result["visible_input"][kind]
            assert 0 <= p["positive_probability"] <= 1
            assert (
                visible["retained_content_tokens"] + visible["discarded_content_tokens"]
                == result["token_count"] - 2
            )
            assert visible["retained_content_tokens"] <= 254
        if name == "long":
            assert (
                result["long"] and len(result["visible_input"]["bilstm_headtail"]["segments"]) == 2
            )
        if name == "html":
            assert result["normalized_text"] == "A great movie & a moving story."
        records.append(
            {
                "example": name,
                "predictions": result["predictions"],
                "tokens": result["token_count"],
                "seconds": result["seconds"],
            }
        )
    checks.append(
        "real inference: positive, negative, long, HTML normalization; token accounting and tail segments"
    )
    report = {
        "passed": True,
        "checks": checks,
        "examples": records,
        "note": "Functional integration, not benchmark accuracy; browser UI checked separately.",
    }
    atomic_json(ROOT / "reports/demo_checks.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
