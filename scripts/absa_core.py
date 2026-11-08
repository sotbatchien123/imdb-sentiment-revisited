"""Phần lõi dùng chung cho bài toán ABSA: mã hoá nhãn, cắt đầu–cuối, mô hình nhiều đầu, độ đo.

Mỗi review có A khía cạnh, mỗi khía cạnh nhận một trong C = 4 nhãn
(positive, negative, neutral, none). Mô hình trả về logits dạng [batch, A, C].
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

LABELS = ["positive", "negative", "neutral", "none"]
NONE_INDEX = LABELS.index("none")
SENTIMENT_INDICES = [LABELS.index(x) for x in ("positive", "negative", "neutral")]


# ---------------------------------------------------------------- dữ liệu


def encode_labels(record: dict, aspects: list[str]) -> list[int]:
    return [LABELS.index(record[a]) for a in aspects]


def load_silver(path: Path, aspects: list[str]) -> dict[str, list[int]]:
    """Đọc file nhãn do `absa_label_llm.py` ghi ra: review_id -> danh sách chỉ số nhãn."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["review_id"]] = encode_labels(rec["labels"], aspects)
    return out


def head_tail(ids: list[int], head: int, tail: int) -> list[int]:
    """Giữ token đặc biệt đầu/cuối, `head` token đầu và `tail` token cuối của nội dung."""
    content = ids[1:-1]
    if len(content) <= head + tail:
        return ids
    return [ids[0]] + content[:head] + content[-tail:] + [ids[-1]]


def class_weights(labels: np.ndarray, mode: str, clip: float = 5.0) -> torch.Tensor:
    """Trọng số lớp theo từng khía cạnh, dạng [A, C]. `sqrt_inverse` giảm ưu thế của lớp none."""
    a_count = labels.shape[1]
    if mode == "none":
        return torch.ones(a_count, len(LABELS))
    if mode != "sqrt_inverse":
        raise ValueError(f"class_weighting không hỗ trợ: {mode}")
    weights = np.ones((a_count, len(LABELS)), dtype=np.float32)
    for a in range(a_count):
        counts = np.bincount(labels[:, a], minlength=len(LABELS)).astype(np.float64)
        present = counts > 0
        w = np.sqrt(counts.sum() / (present.sum() * np.maximum(counts, 1)))
        w[~present] = 1.0
        weights[a] = np.clip(w, 1.0 / clip, clip)
    return torch.tensor(weights)


def multi_aspect_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor):
    """Tổng cross-entropy của các khía cạnh, cộng theo batch (reduction='sum')."""
    total = logits.new_zeros(())
    for a in range(logits.shape[1]):
        total = total + nn.functional.cross_entropy(
            logits[:, a].float(), labels[:, a], weight=weights[a].to(logits.device), reduction="sum"
        )
    return total


# ---------------------------------------------------------------- mô hình


class FinetuneABSA(nn.Module):
    """Fine-tune toàn bộ encoder; vector token đầu -> dropout -> Linear(A*C)."""

    def __init__(self, encoder: nn.Module, n_aspects: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.n_aspects = n_aspects
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.config.hidden_size, n_aspects * len(LABELS))

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(self.dropout(h[:, 0]))
        return logits.view(-1, self.n_aspects, len(LABELS))


class FrozenBiLSTMABSA(nn.Module):
    """Encoder đóng băng; BiLSTM chạy trên toàn bộ token -> trạng thái cuối hai chiều -> Linear(A*C)."""

    def __init__(self, encoder: nn.Module, n_aspects: int, hidden: int = 768, dropout: float = 0.5):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.n_aspects = n_aspects
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            encoder.config.hidden_size, hidden, batch_first=True, bidirectional=True
        )
        self.classifier = nn.Linear(2 * hidden, n_aspects * len(LABELS))

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()  # encoder luôn ở chế độ eval (không dropout) vì không được huấn luyện
        return self

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            x = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = self.dropout(x.float())
        packed = nn.utils.rnn.pack_padded_sequence(
            x, attention_mask.sum(1).cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        logits = self.classifier(self.dropout(torch.cat((h[-2], h[-1]), dim=1)))
        return logits.view(-1, self.n_aspects, len(LABELS))


# ---------------------------------------------------------------- độ đo


def _present(y: np.ndarray, p: np.ndarray) -> list[int]:
    """Các lớp cảm xúc xuất hiện trong nhãn thật hoặc dự đoán (lớp vắng mặt không bị tính 0)."""
    seen = set(np.unique(y)) | set(np.unique(p))
    return [k for k in SENTIMENT_INDICES if k in seen]


def aspect_metrics(y: np.ndarray, p: np.ndarray, aspects: list[str]) -> dict:
    """Độ đo cho từng khía cạnh và trung bình.

    - macro_f1: Macro-F1 trên các lớp xuất hiện trong nhãn thật hoặc dự đoán (độ đo chính).
    - detection_f1: F1 của việc nhận ra khía cạnh có được nhắc tới (khác none).
    - sentiment_macro_f1: Macro-F1 trên 3 lớp cảm xúc, chỉ tính các review mà nhãn thật khác none;
      dự đoán none ở đây bị tính là sai.
    """
    per = {}
    for i, a in enumerate(aspects):
        ya, pa = y[:, i], p[:, i]
        mentioned = ya != NONE_INDEX
        per[a] = {
            "macro_f1": float(f1_score(ya, pa, average="macro", zero_division=0)),
            "detection_f1": float(
                f1_score(mentioned, pa != NONE_INDEX, average="binary", zero_division=0)
            ),
            "sentiment_macro_f1": (
                float(
                    f1_score(
                        ya[mentioned],
                        pa[mentioned],
                        labels=_present(ya[mentioned], pa[mentioned]),
                        average="macro",
                        zero_division=0,
                    )
                )
                if mentioned.any()
                else None
            ),
            "accuracy": float((ya == pa).mean()),
            "label_counts": {
                LABELS[k]: int(v) for k, v in enumerate(np.bincount(ya, minlength=len(LABELS)))
            },
        }
    mean = {
        k: float(np.mean([per[a][k] for a in aspects if per[a][k] is not None]))
        for k in ("macro_f1", "detection_f1", "sentiment_macro_f1", "accuracy")
    }
    return {"mean": mean, "per_aspect": per, "examples": int(len(y))}
