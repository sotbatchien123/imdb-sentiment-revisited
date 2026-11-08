"""Kiểm tra các hàm tính toán của kiểm định, phân tích lỗi và chấm điểm kiểm tay (CPU, không cần file).

.venv-train\Scripts\python.exe -m unittest scripts.tests.test_absa_analysis -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from absa_error_analysis import class_f1  # noqa: E402
from absa_human_check import (  # noqa: E402
    LABELS,
    assign,
    build_gold,
    cohen_kappa,
    fleiss_kappa,
    macro_f1,
    majority,
    sample_reviews,
)
from absa_significance import holm, macro_f1_from_confusion  # noqa: E402


class SignificanceTest(unittest.TestCase):
    def test_confusion_macro_f1_matches_sklearn_on_present_classes(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, 500)  # lớp 3 vắng mặt hoàn toàn
        p = np.where(rng.random(500) < 0.7, y, rng.integers(0, 3, 500))
        cm = np.bincount(y * 4 + p, minlength=16).reshape(4, 4)
        expect = f1_score(y, p, labels=[0, 1, 2], average="macro")
        self.assertAlmostEqual(float(macro_f1_from_confusion(cm)), expect, places=10)

    def test_batched_confusion(self):
        cm = np.zeros((2, 3, 4, 4))
        cm[..., 0, 0] = 5
        np.testing.assert_allclose(macro_f1_from_confusion(cm), np.ones((2, 3)))

    def test_holm(self):
        self.assertEqual(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


class ErrorAnalysisTest(unittest.TestCase):
    def test_class_f1_marks_absent_class(self):
        cm = np.array([[3, 1, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        f1 = class_f1(cm)
        self.assertAlmostEqual(f1[0], 6 / 7)
        self.assertIsNone(f1[2])


class HumanCheckTest(unittest.TestCase):
    def test_kappas_match_reference(self):
        a = ["positive", "none", "neutral", "none", "negative", "none"]
        b = ["positive", "none", "none", "none", "negative", "neutral"]
        self.assertAlmostEqual(cohen_kappa(a, b), cohen_kappa_score(a, b), places=10)
        self.assertAlmostEqual(fleiss_kappa(np.array([[3, 0], [0, 3], [3, 0]])), 1.0)

    def test_macro_f1_and_majority(self):
        y = ["positive", "negative", "none", "none"]
        self.assertAlmostEqual(macro_f1(y, y), 1.0)
        self.assertAlmostEqual(
            macro_f1(y, ["positive", "none", "none", "none"]),
            f1_score(y, ["positive", "none", "none", "none"], average="macro"),
        )
        self.assertEqual(majority(["none", "none", "positive"]), "none")
        self.assertIsNone(majority(["none", "neutral", "positive"]))

    def test_sampling_is_stratified_and_one_per_movie(self):
        data = {f"r{i}": {"rating": [1, 10][i % 2], "movie_id": f"m{i // 3}"} for i in range(300)}
        ids = sample_reviews(data, 40, seed=1)
        self.assertEqual(len(ids), 40)
        self.assertEqual(len({data[i]["movie_id"] for i in ids}), 40)
        self.assertEqual(sum(data[i]["rating"] == 1 for i in ids), 20)
        self.assertEqual(ids, sample_reviews(data, 40, seed=1))

    def test_assignment_and_gold(self):
        ids = [f"r{i}" for i in range(7)]
        plan = assign(ids, shared=1, per=2, annotators=3, seed=0)
        shared = plan["annotator_1"]["shared"]
        owns = [set(p["own"]) for p in plan.values()]
        self.assertEqual(len(set().union(*owns)), 6)
        self.assertTrue(all(set(p["shared"]) == set(shared) for p in plan.values()))
        lab = {a: "none" for a in ["acting", "story", "visual", "sound", "direction", "overall"]}
        filled = {}
        for k, (name, p) in enumerate(plan.items()):
            labels = {i: dict(lab) for i in p["order"]}
            labels[shared[0]]["acting"] = LABELS[k]  # 3 người 3 nhãn khác nhau → không có đa số
            filled[name] = {"labels": labels}
        gold, agreement = build_gold({"plan": plan}, filled)
        self.assertNotIn("acting", gold[shared[0]])
        self.assertEqual(gold[shared[0]]["story"], "none")
        self.assertEqual(len(gold), 7)
        self.assertEqual(agreement["story"]["items"], 1)


if __name__ == "__main__":
    unittest.main()
