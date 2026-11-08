"""Kiểm tra phần không cần nạp mô hình của demo ABSA (chuẩn hoá văn bản, chọn seed).

.venv-train\\Scripts\\python.exe -m unittest scripts.tests.test_absa_demo -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import absa_demo_server as demo  # noqa: E402


class NormalizeTest(unittest.TestCase):
    def test_strips_html_and_entities(self):
        self.assertEqual(demo.normalize("Great<br /><br />film &amp; cast"), "Great film & cast")

    def test_html_only_becomes_empty(self):
        self.assertEqual(demo.normalize("  <br/> <p></p> "), "")


class PickSeedTest(unittest.TestCase):
    def write(self, root: Path, seed: int, score: float, epoch: int = 2):
        d = root / f"seed{seed}"
        d.mkdir(parents=True)
        (d / "result.json").write_text(
            json.dumps({"best_epoch": epoch, "best_validation": {"macro_f1": score}}),
            encoding="utf-8",
        )

    def test_picks_best_validation_and_skips_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write(root, 42, 0.70)
            self.write(root, 44, 0.73, epoch=3)
            seed, sel = demo.pick_seed({"seeds": [42, 43, 44], "output": str(root)})
            self.assertEqual(seed, 44)
            self.assertEqual(sel, {"seed": 44, "epoch": 3, "validation_macro_f1": 0.73})

    def test_no_results_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                demo.pick_seed({"seeds": [42], "output": tmp})


if __name__ == "__main__":
    unittest.main()
