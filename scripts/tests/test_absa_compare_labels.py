"""Unit test cho scripts/absa_compare_labels.py (chỉ cần scikit-learn, không cần GPU).

python -m unittest scripts.tests.test_absa_compare_labels -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from absa_compare_labels import LABELS, compare, overall_vs_rating


class CompareTest(unittest.TestCase):
    def test_identical_labels_agree_perfectly(self):
        ref = {
            "r1": {"acting": "positive"},
            "r2": {"acting": "negative"},
            "r3": {"acting": "none"},
        }
        out = compare(ref, ref, ["acting"])
        acting = out["per_aspect"]["acting"]
        self.assertEqual(out["common_reviews"], 3)
        self.assertAlmostEqual(acting["agreement"], 1.0)
        self.assertAlmostEqual(acting["cohen_kappa"], 1.0)
        self.assertAlmostEqual(acting["macro_f1"], 1.0)

    def test_one_disagreement_is_counted_in_confusion_matrix(self):
        ref = {
            "r1": {"acting": "positive"},
            "r2": {"acting": "negative"},
            "r3": {"acting": "neutral"},
        }
        other = {
            "r1": {"acting": "positive"},
            "r2": {"acting": "negative"},
            "r3": {"acting": "positive"},  # tham chiếu neutral, bên kia positive
        }
        acting = compare(ref, other, ["acting"])["per_aspect"]["acting"]
        self.assertAlmostEqual(acting["agreement"], 2 / 3)
        matrix = acting["confusion_matrix"]["rows_reference_cols_other"]
        self.assertEqual(matrix[LABELS.index("neutral")][LABELS.index("positive")], 1)
        self.assertEqual(sum(map(sum, matrix)), 3)

    def test_only_common_reviews_are_compared(self):
        ref = {"r1": {"acting": "positive"}, "r2": {"acting": "negative"}, "only_ref": {"acting": "none"}}
        other = {"r1": {"acting": "positive"}, "r2": {"acting": "negative"}, "only_other": {"acting": "none"}}
        out = compare(ref, other, ["acting"])
        self.assertEqual(out["common_reviews"], 2)
        self.assertAlmostEqual(out["per_aspect"]["acting"]["agreement"], 1.0)

    def test_mean_averages_aspects(self):
        ref = {
            "r1": {"acting": "positive", "story": "positive"},
            "r2": {"acting": "negative", "story": "negative"},
        }
        other = {
            "r1": {"acting": "positive", "story": "positive"},
            "r2": {"acting": "negative", "story": "positive"},  # story lệch 1/2
        }
        out = compare(ref, other, ["acting", "story"])
        self.assertAlmostEqual(out["per_aspect"]["story"]["agreement"], 0.5)
        self.assertAlmostEqual(out["mean"]["agreement"], 0.75)

    def test_no_common_reviews_does_not_crash(self):
        out = compare({"a": {"acting": "none"}}, {"b": {"acting": "none"}}, ["acting"])
        self.assertEqual(out["common_reviews"], 0)
        self.assertIsNone(out["per_aspect"]["acting"]["agreement"])
        self.assertIsNone(out["mean"]["agreement"])


class OverallVsRatingTest(unittest.TestCase):
    def test_accuracy_on_decided_reviews(self):
        labels = {
            "r1": {"overall": "positive"},  # sao khen -> đúng
            "r2": {"overall": "negative"},  # sao chê -> đúng
            "r3": {"overall": "positive"},  # sao chê -> sai
            "r4": {"overall": "neutral"},   # không quyết định
        }
        gold = {"r1": 1, "r2": 0, "r3": 0, "r4": 1}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.jsonl"
            path.write_text(
                "\n".join(json.dumps({"review_id": k, "label": v}) for k, v in gold.items()),
                encoding="utf-8",
            )
            out = overall_vs_rating(labels, path)
        self.assertEqual(out["reviews"], 4)
        self.assertAlmostEqual(out["decided_fraction"], 3 / 4)
        self.assertAlmostEqual(out["accuracy_on_decided"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
