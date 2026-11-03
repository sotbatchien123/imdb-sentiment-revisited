import json, sys, statistics as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root, out = sys.argv[1], sys.argv[2]
d = json.load(open(root + "/reports/improvement_report/report_data.json", encoding="utf-8"))
for k in ("B", "D"):
    print(k, "training_minutes", [round(r["training_seconds"] / 60, 1) for r in d[k]["runs"]])

BG, FG, MUTED, GRID, RED, GREY = "#1a1e22", "#ffffff", "#b8bcc2", "#3a3f45", "#e50f10", "#8a8f96"
plt.rcParams.update(
    {
        "font.family": "Segoe UI",
        "text.color": FG,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": GRID,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
    }
)


def fig(wpx, hpx):
    return plt.figure(figsize=(wpx / 200, hpx / 200), dpi=200)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def title(f, t, sub=None):
    f.text(0.06, 0.95, t, fontsize=17, fontweight="bold", color=FG, va="top")
    if sub:
        f.text(0.06, 0.905, sub, fontsize=11, color=MUTED, va="top")


sig = d["sig"]["macro_f1_per_seed"]
mean = lambda xs: st.mean(xs) * 100
sd = lambda xs: st.stdev(xs) * 100

# A: tiến trình 6 hệ thống (slide 19, khung 795x864)
rows = [
    ("Tái lập (mô tả)", "reproduction_paper"),
    ("Tái lập (mã)", "reproduction_code"),
    ("BiLSTM lấy đầu", "frozen_bilstm_head"),
    ("BiLSTM đầu–cuối", "frozen_bilstm_headtail"),
    ("BERT-base fine-tune", "finetune_headtail512"),
    ("RoBERTa-large fine-tune", "finetune_roberta_large"),
]
f = fig(1590, 1728)
ax = f.add_axes([0.40, 0.10, 0.52, 0.74])
style(ax)
for i, (name, key) in enumerate(rows):
    m, s = mean(sig[key]), sd(sig[key])
    c = RED if key == "finetune_roberta_large" else GREY
    y = len(rows) - 1 - i
    ax.errorbar(m, y, xerr=s, fmt="o", color=c, ms=13, capsize=6, elinewidth=2)
    ax.text(
        m + max(s, 0.15) + 0.5,
        y,
        f"{m:.2f}%".replace(".", ","),
        va="center",
        fontsize=12.5,
        color=FG,
        fontweight="bold" if c == RED else "normal",
    )
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=12.5, color=FG)
ax.set_xlim(82, 100)
ax.set_xticks([84, 88, 92, 96, 100])
ax.set_xlabel("Macro-F1 test (%), TB 3 seed ± SD", fontsize=11)
title(f, "Từ 84,97% lên 96,74%", "Mỗi bước cải tiến đều tăng Macro-F1")
f.savefig(out + "/A_tien_trinh.png")
plt.close(f)

# B: validation BERT-base (slide 13, khung 424x551)
f = fig(848, 1102)
ax = f.add_axes([0.20, 0.12, 0.72, 0.66])
style(ax)
ax.grid(axis="y", color=GRID)
for s_, ls in (("42", "-"), ("43", "--"), ("44", ":")):
    h = d["hist"]["B" + s_]
    ax.plot(
        [e["epoch"] for e in h],
        [e["val_f1"] * 100 for e in h],
        ls,
        color=FG,
        lw=2.2,
        marker="o",
        ms=7,
        label="seed " + s_,
    )
ax.axvline(2, color=RED, lw=2.5, alpha=0.9)
ax.text(2.07, 93.62, "đỉnh", color=RED, fontsize=13, fontweight="bold")
ax.set_xticks([1, 2, 3])
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Macro-F1 validation (%)", fontsize=11)
ax.legend(frameon=False, fontsize=10.5, labelcolor=MUTED, loc="lower right")
title(f, "Đỉnh ở epoch 2", "BERT-base, 3 seed")
f.savefig(out + "/B_validation.png")
plt.close(f)

# C: kiểm định (slide 17, khung 864x864)
comp = {(c["first"], c["second"]): c for c in d["sig"]["comparisons"]}
items = [
    ("RoBERTa vs BERT-base", ("finetune_roberta_large", "finetune_headtail512")),
    ("RoBERTa vs BiLSTM đầu–cuối", ("finetune_roberta_large", "frozen_bilstm_headtail")),
    ("RoBERTa vs tái lập (mã)", ("finetune_roberta_large", "reproduction_code")),
]
f = fig(1728, 1728)
ax = f.add_axes([0.38, 0.14, 0.54, 0.68])
style(ax)
for i, (name, key) in enumerate(items):
    c = comp[key]
    m = c["difference"] * 100
    lo, hi = (x * 100 for x in c["bootstrap_ci95"])
    y = len(items) - 1 - i
    ax.plot([lo, hi], [y, y], color=RED, lw=6, solid_capstyle="round")
    ax.plot(m, y, "o", color=FG, ms=14)
    ax.text(hi + 0.3, y, f"+{m:.2f}".replace(".", ","), va="center", fontsize=14, fontweight="bold")
    ax.text(
        hi + 0.3,
        y - 0.22,
        f"[{lo:.2f}; {hi:.2f}]".replace(".", ","),
        va="center",
        fontsize=10.5,
        color=MUTED,
    )
ax.axvline(0, color=FG, lw=1.5)
ax.text(0.15, len(items) - 0.45, "0 = không khác biệt", color=MUTED, fontsize=10.5)
ax.set_yticks(range(len(items)))
ax.set_yticklabels([n for n, _ in items][::-1], fontsize=13, color=FG)
ax.set_xlim(-0.5, 10)
ax.set_ylim(-0.6, len(items) - 0.3)
ax.set_xlabel("Chênh lệch Macro-F1 (điểm %), CI 95%", fontsize=11)
title(f, "Cả 3 mức tăng đều xa 0", "Bootstrap theo 3.581 phim, 10.000 lần · p sau Holm ≤ 0,0016")
f.savefig(out + "/C_kiem_dinh.png")
plt.close(f)

# D: đầu–cuối theo độ dài (slide 10, khung 724x864)
runs = d["full"]["runs"]


def grp(model, g):
    return st.mean(r["length_groups"][g]["macro_f1"] for r in runs if r["model"] == model) * 100


f = fig(1448, 1728)
ax = f.add_axes([0.16, 0.12, 0.70, 0.66])
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID)
ax.set_axisbelow(True)
for g, lab, c in (("short", "Review ngắn", GREY), ("long", "Review bị cắt", RED)):
    a, b = grp("bilstm", g), grp("bilstm_headtail", g)
    ax.plot([0, 1], [a, b], color=c, lw=3.5, marker="o", ms=12)
    ax.text(-0.06, a, f"{a:.1f}".replace(".", ","), ha="right", va="center", fontsize=13, color=c)
    ax.text(
        1.06,
        b,
        f"{b:.1f}".replace(".", ","),
        ha="left",
        va="center",
        fontsize=13,
        color=c,
        fontweight="bold",
    )
    ly = 0.85 * (b - a) + a - 1.5 if g == "long" else (a + b) / 2 + 0.35
    ax.text(
        0.85 if g == "long" else 0.5,
        ly,
        f"{lab}: +{b - a:.1f}".replace(".", ","),
        ha="center",
        fontsize=12.5,
        color=c,
        fontweight="bold",
    )
ax.set_xticks([0, 1])
ax.set_xticklabels(["Lấy đầu", "Đầu–cuối"], fontsize=13, color=FG)
ax.set_xlim(-0.35, 1.35)
ax.set_ylim(85, 94.5)
ax.set_ylabel("Macro-F1 test (%)", fontsize=11)
title(f, "Tăng chủ yếu ở review dài", "BiLSTM + BERT đóng băng, 256 token")
f.savefig(out + "/D_dau_cuoi.png")
plt.close(f)


# E: review dài, BERT-base vs RoBERTa (slide 15, khung 594x863)
def lg(k, g):
    return st.mean(r["test"]["length_groups"][g]["macro_f1"] for r in d[k]["runs"]) * 100


f = fig(1188, 1726)
ax = f.add_axes([0.20, 0.12, 0.66, 0.66])
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID)
ax.set_axisbelow(True)
for k, lab, c in (("B", "BERT-base", GREY), ("D", "RoBERTa", RED)):
    a, b = lg(k, "short"), lg(k, "long")
    ax.plot([0, 1], [a, b], color=c, lw=3.5, marker="o", ms=12)
    ax.text(-0.07, a, f"{a:.1f}".replace(".", ","), ha="right", va="center", fontsize=13, color=c)
    ax.text(
        1.07,
        b,
        f"{b:.1f}".replace(".", ","),
        ha="left",
        va="center",
        fontsize=13,
        color=c,
        fontweight="bold",
    )
    ax.text(0.5, (a + b) / 2 + 0.25, lab, ha="center", fontsize=13, color=c, fontweight="bold")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Ngắn", "Dài"], fontsize=13, color=FG)
ax.set_xlim(-0.4, 1.4)
ax.set_ylim(93, 97.6)
ax.set_ylabel("Macro-F1 test (%)", fontsize=11)
title(f, "Review dài\nkhông còn kém", None)
f.savefig(out + "/E_review_dai.png")
plt.close(f)
print("ok")
