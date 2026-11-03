"""Vẽ hình cho bài báo IEEE (nền trắng, cỡ cột IEEE) từ reports/improvement_report/report_data.json.

Chạy từ thư mục gốc dự án:  python reports/ieee_paper/make_figures.py
Đầu ra: reports/ieee_paper/figures/*.pdf (dùng trong LaTeX) và *.png (xem nhanh).
"""

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
d = json.loads((ROOT / "reports/improvement_report/report_data.json").read_text(encoding="utf-8"))

ACCENT, GREY, DARK = "#c8102e", "#8c8c8c", "#222222"
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


def clean(ax, grid="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color="#e3e3e3", linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


sig = d["sig"]["macro_f1_per_seed"]

# Hình 1: Macro-F1 của 6 cấu hình (TB ± SD qua 3 seed)
rows = [
    ("Reproduction (paper settings)", "reproduction_paper"),
    ("Reproduction (author code)", "reproduction_code"),
    ("Frozen BERT + BiLSTM, head", "frozen_bilstm_head"),
    ("Frozen BERT + BiLSTM, head+tail", "frozen_bilstm_headtail"),
    ("Fine-tuned BERT-base, head+tail 512", "finetune_headtail512"),
    ("Fine-tuned RoBERTa-large, head+tail 512", "finetune_roberta_large"),
]
fig, ax = plt.subplots(figsize=(COL, 1.9))
clean(ax)
for i, (name, key) in enumerate(rows):
    y = len(rows) - 1 - i
    m, s = st.mean(sig[key]) * 100, st.stdev(sig[key]) * 100
    c = ACCENT if key == "finetune_roberta_large" else GREY
    ax.errorbar(m, y, xerr=s, fmt="o", color=c, ms=4, capsize=2, elinewidth=0.8)
    ax.text(m + max(s, 0.1) + 0.35, y, f"{m:.2f}", va="center", fontsize=7.5, color=DARK)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows][::-1])
ax.set_xlim(83, 99)
ax.set_xlabel("Test Macro-F1 (%), mean ± SD over 3 seeds")
save(fig, "fig_progression")

# Hình 2: chênh lệch Macro-F1 với CI 95% (bootstrap theo phim) cho cả 8 phép so sánh
names = {
    "finetune_roberta_large": "RoBERTa-L FT",
    "finetune_headtail512": "BERT-B FT",
    "frozen_bilstm_headtail": "Frozen head+tail",
    "frozen_bilstm_head": "Frozen head",
    "reproduction_code": "Repro (code)",
    "reproduction_paper": "Repro (paper)",
}
comps = d["sig"]["comparisons"]
fig, ax = plt.subplots(figsize=(COL, 2.2))
clean(ax)
for i, c in enumerate(comps):
    y = len(comps) - 1 - i
    m = c["difference"] * 100
    lo, hi = (x * 100 for x in c["bootstrap_ci95"])
    ax.plot([lo, hi], [y, y], color=ACCENT, lw=2.2, solid_capstyle="butt")
    ax.plot(m, y, "o", color=DARK, ms=3)
    ax.text(hi + 0.25, y, f"+{m:.2f}", va="center", fontsize=7.5, color=DARK)
ax.axvline(0, color=DARK, lw=0.8)
ax.set_yticks(range(len(comps)))
ax.set_yticklabels([f"{names[c['first']]} vs {names[c['second']]}" for c in comps][::-1])
ax.set_xlim(-0.3, 11)
ax.set_xlabel("Difference in test Macro-F1 (points), 95% CI")
save(fig, "fig_significance")

# Hình 3: Macro-F1 theo nhóm độ dài
runs = d["full"]["runs"]


def frozen(model, g):
    return st.mean(r["length_groups"][g]["macro_f1"] for r in runs if r["model"] == model) * 100


def tuned(k, g):
    return st.mean(r["test"]["length_groups"][g]["macro_f1"] for r in d[k]["runs"]) * 100


fig, axes = plt.subplots(1, 2, figsize=(COL, 1.75))
panels = [
    (
        axes[0],
        "(a) Frozen BERT + BiLSTM, 256 tokens",
        ["Head", "Head+tail"],
        [
            (
                "Short",
                GREY,
                [frozen("bilstm", "short"), frozen("bilstm_headtail", "short")],
                0.5,
                0.06,
            ),
            (
                "Truncated",
                ACCENT,
                [frozen("bilstm", "long"), frozen("bilstm_headtail", "long")],
                0.72,
                -0.24,
            ),
        ],
        (85.5, 94),
    ),
    (
        axes[1],
        "(b) Fine-tuned, head+tail 512",
        ["Short", "Truncated"],
        [
            ("BERT-B", GREY, [tuned("B", "short"), tuned("B", "long")], 0.45, -0.12),
            ("RoBERTa-L", ACCENT, [tuned("D", "short"), tuned("D", "long")], 0.5, 0.06),
        ],
        (93, 97.5),
    ),
]
for ax, title, xt, series, ylim in panels:
    clean(ax, "y")
    for lab, c, (a, b), lx, dy in series:
        ax.plot([0, 1], [a, b], color=c, marker="o", ms=3.5, lw=1.4)
        ax.text(-0.08, a, f"{a:.1f}", ha="right", va="center", fontsize=7, color=c)
        ax.text(1.08, b, f"{b:.1f}", ha="left", va="center", fontsize=7, color=c)
        ax.text(
            lx, a + (b - a) * lx + (ylim[1] - ylim[0]) * dy, lab, ha="center", fontsize=7, color=c
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(xt)
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=7.5)
axes[0].set_ylabel("Test Macro-F1 (%)")
fig.tight_layout(w_pad=1.0)
save(fig, "fig_length")

# Hình 4: Macro-F1 validation theo epoch
fig, axes = plt.subplots(1, 2, figsize=(COL, 1.6))
for ax, k, title in ((axes[0], "B", "(a) BERT-base"), (axes[1], "D", "(b) RoBERTa-large")):
    clean(ax, "y")
    for s_, ls in (("42", "-"), ("43", "--"), ("44", ":")):
        h = d["hist"][k + s_]
        ax.plot(
            [e["epoch"] for e in h],
            [e["val_f1"] * 100 for e in h],
            ls,
            color=DARK,
            marker="o",
            ms=2.5,
            lw=1,
            label=f"seed {s_}",
        )
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Epoch")
    ax.set_title(title, fontsize=7.5)
axes[0].set_ylabel("Validation Macro-F1 (%)")
axes[1].legend(frameon=False, fontsize=6.5, loc="lower right")
fig.tight_layout(w_pad=1.0)
save(fig, "fig_validation")
print("saved to", OUT)
