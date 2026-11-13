"""Flow chart of the proposed approach and evaluation protocol (phase boxes, data cylinders). Usage: python -I fig_flow.py <out_stem>"""
import sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Rectangle, FancyArrowPatch
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(8.6, 12.2)); ax.set_xlim(0, 86); ax.set_ylim(0, 122); ax.axis("off")
BL = "#1F5FA8"
TH = {"grey": ("#D9D9D9", "#8C8C8C", "#555555"), "yellow": ("#FCE9A8", "#E3B23C", "#E8B931"), "green": ("#CFE8C4", "#6BAA55", "#5DA34A"), "red": ("#F6C3C0", "#D16A63", "#C9534B")}
def phase(y0, h, title, col):
    ax.add_patch(FancyBboxPatch((2, y0), 82, h, boxstyle="round,pad=0.2,rounding_size=3", fc="none", ec=TH[col][2], lw=2.6, ls=(0, (1, 2.2))))
    pass
def cyl(x, y, w, h, text, col):
    f, e, _ = TH[col]
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc=f, ec="none", zorder=2))
    ax.plot([x - w / 2] * 2, [y - h / 2, y + h / 2], color=e, lw=1, zorder=3); ax.plot([x + w / 2] * 2, [y - h / 2, y + h / 2], color=e, lw=1, zorder=3)
    ax.add_patch(Ellipse((x, y - h / 2), w, 2.4, fc=f, ec=e, lw=1, zorder=2)); ax.add_patch(Ellipse((x, y + h / 2), w, 2.4, fc=f, ec=e, lw=1, zorder=4))
    ax.text(x, y - 0.4, text, ha="center", va="center", fontsize=8.6, zorder=5)
def box(x, y, w, h, text, col, fs=8.6):
    f, e, _ = TH[col]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.3,rounding_size=1", fc=f, ec=e, lw=1, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=5)
def arr(p, q, conn="arc3"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.6, color=BL, connectionstyle=conn, zorder=1))
# Phase 1: data preparation
def title(y, t, right=False): ax.text(79 if right else 5, y, t, fontsize=12.5, fontweight="bold", color=BL, va="top", ha="right" if right else "left")
phase(97, 24, "", "grey"); title(119.6, "Data Preparation Phase")
cyl(30, 112, 16, 6, "Imbalanced\ndataset", "grey"); cyl(64, 112, 14, 6, "Test fold", "grey")
cyl(30, 101, 16, 5, "Train folds", "grey"); arr((30, 107.8), (30, 104.4))
ax.text(47, 113.4, "stratified 5-fold CV", fontsize=8, ha="center", color="#444444"); arr((38.2, 112), (56.8, 112))
box(55, 101, 22, 4.6, "standardise\n(fitted on train folds)", "grey", 8); arr((38.2, 101), (43.8, 101))
# Phase 2: portfolio oversampling
phase(52, 42, "", "yellow"); title(92.6, "Portfolio Oversampling Phase", right=True)
arr((30, 98.3), (30, 86.2)); ax.text(28.5, 88.5, "run each\noversampler once", fontsize=8, color="#444444", ha="right")
names = ["SMOTE", "ProWSyn", "PF-SMOTE", "SMOTE-IPF", "CCR", "LVQ-SMOTE", "SL-SMOTE", "MWMOTE"]
xs = [12 + i * 19 for i in range(4)]
for i, nm in enumerate(names):
    box(xs[i % 4], 82 if i < 4 else 74.5, 16, 4.6, nm, "yellow")
arr((40, 71.6), (40, 68.4))
box(40, 66, 54, 4.6, "keep $w_o\\cdot g$ synthetic examples from each  (PORT: $w_o = 1/8$)", "yellow")
arr((40, 63.5), (40, 60.6)); cyl(40, 57.3, 20, 5, "Balanced\ntraining data", "yellow")
# Phase 3: evaluation
phase(3, 47, "", "green"); title(48.6, "Evaluation Phase")
arr((40, 54.6), (40, 44.2)); ax.plot([12, 69], [44, 44], color=BL, lw=1.6)
cl = ["kNN", "SVM", "MLP", "RF"]; cx = [12, 31, 50, 69]
for x, c in zip(cx, cl):
    arr((x, 44), (x, 40.8)); box(x, 38.2, 15, 4.6, c, "green")
    arr((x, 35.7), (x, 32.6))
box(40, 30.2, 66, 4.6, "predict the held-out test fold:  AUC and G-mean", "green")
ax.plot([71.2, 81, 81], [112, 112, 30.2], color=BL, lw=1.6); arr((81, 30.2), (73.6, 30.2))
arr((40, 27.7), (40, 24.6)); box(40, 22.2, 66, 4.6, "average over folds; rank the 19 methods on each dataset", "green")
arr((40, 19.7), (40, 16.6)); box(40, 14.2, 66, 4.6, "Friedman / Iman-Davenport test over datasets", "green")
arr((40, 11.7), (40, 8.6)); box(40, 6.4, 66, 4.2, "Wilcoxon tests with Holm correction and effect sizes", "green")
plt.savefig(sys.argv[1] + ".png", dpi=300, bbox_inches="tight"); plt.savefig(sys.argv[1] + ".pdf", bbox_inches="tight"); print("ok")
