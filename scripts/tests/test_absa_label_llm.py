"""Kiểm tra trên CPU cho script gán nhãn ABSA (không cần llama-server).

.venv-train\Scripts\python.exe -m unittest scripts.tests.test_absa_label_llm -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from absa_label_llm import (
    build_payload,
    clip_review,
    is_grounded,
    parse_evidence_labels,
    parse_labels,
    read_done_ids,
)

ASPECTS = ["acting", "story", "visual", "sound", "direction", "pacing", "overall"]
LABELS = ["positive", "negative", "neutral", "none"]
CFG = {
    "aspects": ASPECTS,
    "labels": LABELS,
    "generation": {"temperature": 0.0, "max_tokens": 120, "enable_thinking": False},
}


class ClipReviewTest(unittest.TestCase):
    def test_short_review_only_cleans_html(self):
        self.assertEqual(clip_review("Great<br /><br />film", 100), "Great film")

    def test_long_review_keeps_head_and_tail(self):
        text = "A" * 50 + "B" * 50
        clipped = clip_review(text, 20)
        self.assertTrue(clipped.startswith("A" * 10))
        self.assertTrue(clipped.endswith("B" * 10))
        self.assertIn("[...]", clipped)


class ParseLabelsTest(unittest.TestCase):
    def full(self, **override):
        data = {a: "none" for a in ASPECTS}
        data.update(override)
        return data

    def test_valid_json(self):
        content = json.dumps(self.full(acting="positive"))
        self.assertEqual(parse_labels(content, ASPECTS, LABELS)["acting"], "positive")

    def test_strips_think_block_and_fence(self):
        content = "<think>hmm</think>\n```json\n" + json.dumps(self.full()) + "\n```"
        self.assertEqual(parse_labels(content, ASPECTS, LABELS)["overall"], "none")

    def test_case_is_normalised(self):
        content = json.dumps(self.full(sound="Negative"))
        self.assertEqual(parse_labels(content, ASPECTS, LABELS)["sound"], "negative")

    def test_missing_aspect_fails(self):
        data = self.full()
        del data["pacing"]
        with self.assertRaises(ValueError):
            parse_labels(json.dumps(data), ASPECTS, LABELS)

    def test_unknown_label_fails(self):
        with self.assertRaises(ValueError):
            parse_labels(json.dumps(self.full(story="mixed")), ASPECTS, LABELS)


class PayloadTest(unittest.TestCase):
    def test_schema_restricts_labels_and_disables_thinking(self):
        payload = build_payload("text", "system", CFG)
        schema = payload["response_format"]["json_schema"]["schema"]
        self.assertEqual(schema["required"], ASPECTS)
        self.assertEqual(schema["properties"]["visual"]["enum"], LABELS)
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(payload["messages"][0]["content"], "system")


class EvidenceTest(unittest.TestCase):
    REVIEW = "Tom Hanks is brilliant here, but the plot makes no sense at all. 3/10."

    def item(self, evidence, label):
        return {"evidence": evidence, "label": label}

    def full(self, **override):
        data = {a: self.item("", "none") for a in ASPECTS}
        data.update(override)
        return json.dumps(data)

    def test_grounding(self):
        self.assertTrue(is_grounded("the plot makes no sense", self.REVIEW))
        self.assertTrue(is_grounded("Hanks is brilliant", self.REVIEW))
        self.assertFalse(is_grounded("", self.REVIEW))
        self.assertFalse(is_grounded("gorgeous cinematography and lighting", self.REVIEW))

    def test_ungrounded_label_forced_to_none(self):
        content = self.full(
            acting=self.item("Tom Hanks is brilliant", "positive"),
            sound=self.item("the music was terrible", "negative"),
            pacing=self.item("", "neutral"),
            overall=self.item("", "negative"),
        )
        labels, evidence, overridden = parse_evidence_labels(
            content, ASPECTS, LABELS, self.REVIEW, exempt=["overall"]
        )
        self.assertEqual(labels["acting"], "positive")
        self.assertEqual(evidence["acting"], "Tom Hanks is brilliant")
        self.assertEqual(labels["sound"], "none")
        self.assertEqual(labels["pacing"], "none")
        self.assertEqual(labels["overall"], "negative")  # overall được miễn kiểm tra bằng chứng
        self.assertEqual(sorted(overridden), ["pacing", "sound"])

    def test_evidence_schema_puts_evidence_first(self):
        cfg = dict(CFG, evidence=True)
        schema = build_payload("t", "s", cfg)["response_format"]["json_schema"]["schema"]
        item = schema["properties"]["story"]
        self.assertEqual(list(item["properties"]), ["evidence", "label"])
        self.assertEqual(item["properties"]["label"]["enum"], LABELS)


class ResumeTest(unittest.TestCase):
    def test_read_done_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            self.assertEqual(read_done_ids(path), set())
            path.write_text('{"review_id": "a"}\n\n{"review_id": "b"}\n', encoding="utf-8")
            self.assertEqual(read_done_ids(path), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
