"""Kiểm tra trên CPU cho phần lõi và runner ABSA, dùng một BERT tí hon khởi tạo ngẫu nhiên.

.venv-train\Scripts\python.exe -m unittest scripts.tests.test_absa_train -v
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from absa_core import (
    LABELS,
    NONE_INDEX,
    FinetuneABSA,
    FrozenBiLSTMABSA,
    aspect_metrics,
    class_weights,
    head_tail,
    multi_aspect_loss,
)

ROOT = Path(__file__).resolve().parents[2]
ASPECTS = ["acting", "story", "visual", "sound", "direction", "pacing", "overall"]


def tiny_encoder():
    from transformers import BertConfig, BertModel

    cfg = BertConfig(
        vocab_size=30522,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=512,
    )
    torch.manual_seed(0)
    return BertModel(cfg)


class CoreTest(unittest.TestCase):
    def test_head_tail_keeps_special_tokens(self):
        ids = [101] + list(range(1, 21)) + [102]
        out = head_tail(ids, 3, 2)
        self.assertEqual(out, [101, 1, 2, 3, 19, 20, 102])
        self.assertEqual(head_tail([101, 5, 102], 3, 2), [101, 5, 102])

    def test_class_weights_downweight_majority(self):
        labels = np.array([[NONE_INDEX]] * 90 + [[0]] * 10)
        w = class_weights(labels, "sqrt_inverse")
        self.assertEqual(tuple(w.shape), (1, 4))
        self.assertLess(w[0, NONE_INDEX], w[0, 0])
        self.assertEqual(float(w[0, 2]), 1.0)  # lớp không xuất hiện giữ trọng số 1
        self.assertTrue(torch.equal(class_weights(labels, "none"), torch.ones(1, 4)))

    def test_metrics_perfect_and_detection(self):
        y = np.array([[0, NONE_INDEX], [1, 2], [2, NONE_INDEX], [NONE_INDEX, 0]])
        m = aspect_metrics(y, y.copy(), ["a", "b"])
        self.assertAlmostEqual(m["mean"]["macro_f1"], 1.0)
        self.assertAlmostEqual(m["per_aspect"]["a"]["detection_f1"], 1.0)
        p = y.copy()
        p[0, 0] = NONE_INDEX  # bỏ sót khía cạnh có nhắc tới
        m2 = aspect_metrics(y, p, ["a", "b"])
        self.assertLess(m2["per_aspect"]["a"]["detection_f1"], 1.0)
        self.assertLess(m2["per_aspect"]["a"]["sentiment_macro_f1"], 1.0)

    def test_models_output_shape_and_frozen_encoder(self):
        ids = torch.randint(1000, 2000, (3, 12))
        mask = torch.ones_like(ids)
        mask[1, 8:] = 0
        ft = FinetuneABSA(tiny_encoder(), len(ASPECTS))
        self.assertEqual(tuple(ft(ids, mask).shape), (3, len(ASPECTS), len(LABELS)))
        fb = FrozenBiLSTMABSA(tiny_encoder(), len(ASPECTS), hidden=16)
        out = fb(ids, mask)
        self.assertEqual(tuple(out.shape), (3, len(ASPECTS), len(LABELS)))
        loss = multi_aspect_loss(
            out, torch.zeros(3, len(ASPECTS), dtype=torch.long), torch.ones(len(ASPECTS), 4)
        )
        loss.backward()
        self.assertTrue(all(p.grad is None for p in fb.encoder.parameters()))
        self.assertIsNotNone(fb.lstm.weight_ih_l0.grad)
        fb.train()
        self.assertFalse(fb.encoder.training)


class RunnerSmokeTest(unittest.TestCase):
    def test_cpu_smoke_and_full_run(self):
        import train_absa
        from transformers import AutoTokenizer

        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            enc_dir, data_dir, lab_dir = tmp / "enc", tmp / "data", tmp / "labels"
            tiny_encoder().save_pretrained(enc_dir)
            AutoTokenizer.from_pretrained(
                ROOT / "checkpoints/bert-base-uncased", local_files_only=True
            ).save_pretrained(enc_dir)
            data_dir.mkdir()
            lab_dir.mkdir()
            words = "great acting terrible plot lovely music slow boring film".split()
            for split, n in (("train", 24), ("validation", 12), ("test", 12)):
                with open(data_dir / f"{split}.jsonl", "w", encoding="utf-8") as d, open(
                    lab_dir / f"{split}.jsonl", "w", encoding="utf-8"
                ) as lab:
                    for i in range(n):
                        rid = f"{split}/{i}"
                        text = " ".join(rng.choice(words, size=int(rng.integers(5, 60))))
                        d.write(
                            json.dumps(
                                {
                                    "review_id": rid,
                                    "movie_id": f"tt{i % 4}",
                                    "label": i % 2,
                                    "text": text,
                                }
                            )
                            + "\n"
                        )
                        labels = {a: LABELS[int(rng.integers(0, 4))] for a in ASPECTS}
                        lab.write(json.dumps({"review_id": rid, "labels": labels}) + "\n")
            for model_type in ("finetune", "frozen_bilstm"):
                cfg = {
                    "version": "t",
                    "model_type": model_type,
                    "encoder": str(enc_dir),
                    "dataset": str(data_dir),
                    "labels_dir": str(lab_dir),
                    "aspects": ASPECTS,
                    "seeds": [42],
                    "max_length": 32,
                    "head_tokens": 10,
                    "tail_tokens": 20,
                    "class_weighting": "sqrt_inverse",
                    "learning_rate": 1e-3,
                    "weight_decay": 0.01,
                    "warmup_fraction": 0.1,
                    "max_grad_norm": 1.0,
                    "epochs": 2,
                    "micro_batch_size": 4,
                    "gradient_accumulation": 2,
                    "eval_batch_size": 8,
                    "lstm_hidden_size": 16,
                    "smoke_examples": 8,
                    "output": str(tmp / model_type / "out"),
                    "checkpoint_dir": str(tmp / model_type / "ckpt"),
                    "report_dir": str(tmp / model_type / "report"),
                }
                cfg_path = tmp / f"{model_type}.json"
                cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        train_absa.main(["--config", str(cfg_path), "--smoke", "--device", "cpu"]),
                        0,
                    )
                    self.assertEqual(
                        train_absa.main(["--config", str(cfg_path), "--device", "cpu"]), 0
                    )
                result = json.loads(
                    (tmp / model_type / "out/seed42/result.json").read_text("utf-8")
                )
                self.assertEqual(len(result["history"]), 2)
                self.assertEqual(set(result["test"]["per_aspect"]), set(ASPECTS))
                self.assertIsNotNone(result["overall_vs_imdb_rating"])
                preds = (tmp / model_type / "out/seed42/test_predictions.jsonl").read_text("utf-8")
                self.assertEqual(len(preds.splitlines()), 12)
                agg = json.loads((tmp / model_type / "report/results.json").read_text("utf-8"))
                self.assertEqual(agg["aggregate"]["seeds"], [42])
                self.assertTrue((tmp / model_type / "ckpt/seed42/best.pt").exists())


if __name__ == "__main__":
    unittest.main()
