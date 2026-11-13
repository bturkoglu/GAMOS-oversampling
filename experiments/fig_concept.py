"""Conceptual figure: where four portfolio members place synthetic examples on the same 2-D toy data (real outputs),
and the PORT union. Style follows +/- marker illustrations common in oversampling papers. Usage: python -I fig_concept.py <out_stem>"""
import sys, logging, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import smote_variants as sv; logging.getLogger("smote_variants").setLevel(logging.ERROR)
RED, BLUE, GREEN = "#D62828", "#1D4ED8", "#2E9E44"
plt.rcParams.update({"font.family": "DejaVu Sans"})
rng = np.random.RandomState(7)
Xmaj = rng.normal([0.0, 0.0], [1.0, 0.85], (40, 2))
Xmin = np.r_[rng.normal([2.55, 1.85], [0.32, 0.32], (6, 2)), [[1.25, 0.95], [1.55, 0.25], [0.55, 1.45], [-0.35, -0.45]]]
X = np.vstack([Xmaj, Xmin]); y = np.r_[np.zeros(len(Xmaj), int), np.ones(len(Xmin), int)]
def pm(ax, P, kind, s=230):
    c = RED if kind == "+" else BLUE
    ax.scatter(*P.T, s=s, c=c, edgecolors="white", linewidths=1.2, zorder=3)
    ax.scatter(*P.T, s=s * 0.32, c="white", marker="P" if kind == "+" else "_", linewidths=2.2 if kind == "-" else 0, zorder=4)
def syn(ax, P, color=GREEN, s=120):
    ax.scatter(*P.T, s=s, c=color, edgecolors="white", linewidths=1, zorder=5)
    ax.scatter(*P.T, s=s * 0.35, c="white", marker="x", linewidths=1.6, zorder=6)
def run(name, n=8):
    o = getattr(sv, name)(random_state=1, proportion=1.0)
    Xr, yr = o.sample(X, y); S = Xr[len(X):]
    idx = np.random.RandomState(3).choice(len(S), size=min(n, len(S)), replace=False); return S[idx]
panels = [("SMOTE", "SMOTE", "interpolates between\nminority neighbours"),
          ("ProWSyn", "ProWSyn", "favours minority examples\nnear the majority class"),
          ("Safe_Level_SMOTE", "Safe-Level-SMOTE", "pushes samples toward\nsafe minority regions"),
          ("CCR", "CCR", "samples inside spheres\naround minority examples")]
fig, axes = plt.subplots(1, 5, figsize=(17, 4.2))
outs = {}
for ax, (code, title, sub) in zip(axes[:4], panels):
    S = run(code); outs[title] = S
    pm(ax, Xmaj, "-", 150); pm(ax, Xmin, "+", 150); syn(ax, S)
    if code == "SMOTE":   # show one interpolation segment like classic SMOTE figures
        a, b = Xmin[6], Xmin[0]; ax.plot(*np.c_[a, b], ls="--", color="#3B82F6", lw=1.3, zorder=2)
        ax.text(*(a + [0.32, -0.30]), r"$\mathbf{x}_i$", fontsize=12, ha="center"); ax.text(*(b + [0.0, 0.33]), r"$\mathbf{x}_j$", fontsize=12, ha="center")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=4); ax.text(0.5, -0.06, sub, transform=ax.transAxes, ha="center", va="top", fontsize=10)
cols = {"SMOTE": "#2E9E44", "ProWSyn": "#F59E0B", "Safe-Level-SMOTE": "#8B5CF6", "CCR": "#0EA5E9"}
ax = axes[4]; pm(ax, Xmaj, "-", 150); pm(ax, Xmin, "+", 150)
for k, (t, S) in enumerate(outs.items()):
    syn(ax, S[:4], cols[t], 110); ax.text(0.97, 0.30 - k * 0.075, t + " \u25CF", color=cols[t], transform=ax.transAxes, fontsize=9.5, va="top", ha="right", fontweight="bold")
ax.set_title("PORT (union)", fontsize=13, fontweight="bold", pad=4); ax.text(0.5, -0.06, "equal shares from every\nmember (colour = source)", transform=ax.transAxes, ha="center", va="top", fontsize=10)
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="o", ls="", ms=10, mfc=RED, mec="white", label="minority"), Line2D([], [], marker="o", ls="", ms=10, mfc=BLUE, mec="white", label="majority"),
     Line2D([], [], marker="o", ls="", ms=9, mfc=GREEN, mec="white", label="synthetic")]
fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.04))
lim = (X[:, 0].min() - 0.6, X[:, 0].max() + 0.6), (X[:, 1].min() - 0.6, X[:, 1].max() + 0.6)
for a in axes:
    a.set_xlim(*lim[0]); a.set_ylim(*lim[1]); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_color("#BBBBBB")
plt.tight_layout(w_pad=0.6); plt.savefig(sys.argv[1] + ".png", dpi=300, bbox_inches="tight"); plt.savefig(sys.argv[1] + ".pdf", bbox_inches="tight"); print("ok")
