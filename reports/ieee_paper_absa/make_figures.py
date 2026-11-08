"""Vẽ hình cho bài báo IEEE về ABSA (nền trắng, cỡ cột IEEE).

Chạy từ thư mục gốc dự án, sau absa_significance.py và absa_error_analysis.py:
    python reports/ieee_paper_absa/make_figures.py
Đầu ra: reports/ieee_paper_absa/figures/*.pdf (dùng trong LaTeX) và *.png (xem nhanh).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
sig = json.loads((ROOT / "reports/absa_v1/significance.json").read_text(encoding="utf-8"))
ana = json.loads(
    (ROOT / "reports/absa_v1/analysis/error_analysis.json").read_text(encoding="utf-8")
)

ACCENT, MID, LIGHT, DARK = "#c8102e", "#e8838f", "#bdbdbd", "#222222"
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.edgecolor": "#555555",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)
COL = 3.5  # độ rộng một cột IEEE (inch)
SYSTEMS = [
    ("bilstm", "Frozen BERT + BiLSTM", LIGHT),
    ("bert_base", "BERT-base FT", MID),
    ("roberta_large", "RoBERTa-large FT", ACCENT),
]
ASPECTS = sig["aspects"]


def clean(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color="#e3e3e3", linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


# Hình 1: Macro-F1 từng khía cạnh, có khoảng tin cậy bootstrap 95%
fig, ax = plt.subplots(figsize=(COL, 1.9))
x = np.arange(len(ASPECTS))
w = 0.26
for k, (key, label, color) in enumerate(SYSTEMS):
    s = sig["systems"][key]
    vals = np.array([s["per_aspect_macro_f1"][a] for a in ASPECTS]) * 100
    ci = np.array([s["per_aspect_ci95"][a] for a in ASPECTS]) * 100
    err = np.vstack([vals - ci[:, 0], ci[:, 1] - vals])
    ax.bar(x + (k - 1) * w, vals, w, color=color, label=label, edgecolor="white", linewidth=0.3)
    ax.errorbar(x + (k - 1) * w, vals, yerr=err, fmt="none", ecolor=DARK, elinewidth=0.6, capsize=1.2)
ax.set_xticks(x, ASPECTS)
ax.set_ylim(50, 82)
ax.set_ylabel("Test Macro-F1 (%)")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=7)
clean(ax)
save(fig, "fig_aspects")

# Hình 2: F1 từng lớp của RoBERTa-large
rob = ana["systems"]["roberta_large"]["per_aspect"]
labels = ana["labels"]
mat = np.array([[rob[a]["class_f1"][lab] * 100 for lab in labels] for a in ASPECTS])
fig, ax = plt.subplots(figsize=(COL, 1.75))
im = ax.imshow(mat.T, cmap="Reds", vmin=30, vmax=100, aspect="auto")
ax.set_xticks(range(len(ASPECTS)), ASPECTS)
ax.set_yticks(range(len(labels)), labels)
for i in range(len(ASPECTS)):
    for j in range(len(labels)):
        v = mat[i, j]
        ax.text(i, j, f"{v:.1f}", ha="center", va="center", fontsize=7,
                color="white" if v > 75 else DARK)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
save(fig, "fig_class_f1")

# Hình 3: overall dự đoán theo số sao IMDb
by = ana["roberta_overall_by_star_rating"]
stars = list(by)
fig, ax = plt.subplots(figsize=(COL, 1.6))
bottom = np.zeros(len(stars))
for lab, color in (("negative", "#555555"), ("neutral", LIGHT), ("positive", ACCENT),
                   ("none", "#eeeeee")):
    v = np.array([by[s][lab] for s in stars]) * 100
    ax.bar(stars, v, 0.7, bottom=bottom, color=color, label=lab, edgecolor="white", linewidth=0.3)
    bottom += v
ax.set_xlabel("IMDb star rating given by the reviewer")
ax.set_ylabel("Predicted overall (%)")
ax.set_ylim(0, 100)
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22), fontsize=7)
clean(ax)
save(fig, "fig_stars")

print("Đã vẽ:", sorted(p.name for p in OUT.glob("*.pdf")))
