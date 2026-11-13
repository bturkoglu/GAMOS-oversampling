"""Figure: learned GAMOS v2 mixed strategy per Collection C dataset (oversampler weights and expected sampling ratio),
plus exploratory Spearman correlations. Usage: python -I figStratC.py <strategy_csv> <out_stem>"""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
D = pd.read_csv(sys.argv[1])
O = ["ProWSyn", "polynom_fit_SMOTE_star", "SMOTE_IPF", "CCR", "LVQ_SMOTE", "Safe_Level_SMOTE", "MWMOTE", "SMOTE"]
ON = ["ProWSyn", "PF-SMOTE", "SMOTE-IPF", "CCR", "LVQ-SMOTE", "SL-SMOTE", "MWMOTE", "SMOTE"]
R = [0.25, 0.5, 1.0]
for o in O: D[o] = sum(D[f"{o}|{r}"] for r in R)
D["E_r"] = sum(D[f"{o}|{r}"] * r for o in O for r in R)
D["max_w"] = D[O].max(1); D["top"] = D[O].idxmax(1)
D["boundary"] = D[["ProWSyn", "MWMOTE", "CCR"]].sum(1)
D = D.sort_values("E_r")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.5, 6.2), gridspec_kw={"width_ratios": [8, 1.3]}, sharey=True)
im = ax.imshow(D[O].values, aspect="auto", cmap="Blues", vmin=0, vmax=max(0.5, D[O].values.max()))
ax.set_xticks(range(len(O))); ax.set_xticklabels(ON, rotation=40, ha="right")
ax.set_yticks(range(len(D))); ax.set_yticklabels([f"{d[5:].replace('_', '-')} (IR {ir:.0f})" for d, ir in zip(D.dataset, D.IR)])
for i, row in enumerate(D[O].values):
    for j, v in enumerate(row):
        if v >= 0.15: ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v > 0.35 else "black")
ax.set_xlabel("oversampler (weight summed over sampling ratios)")
ax2.barh(range(len(D)), D.E_r, color="#E76F51"); ax2.set_xlim(0, 1); ax2.set_xlabel("expected\nratio r"); ax2.axvline(1, color="grey", lw=0.5)
plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="weight in GAMOS mixed strategy")
plt.tight_layout(); plt.savefig(sys.argv[2] + ".png", dpi=300); plt.savefig(sys.argv[2] + ".pdf")
print("datasets with one oversampler > 0.5:", int((D.max_w > 0.5).sum()), "of", len(D))
print("top oversampler counts:", D.top.value_counts().to_dict())
print("mean weight per oversampler:", D[O].mean().round(3).to_dict())
print("E_r: median %.2f range %.2f-%.2f; share with E_r<0.9: %d" % (D.E_r.median(), D.E_r.min(), D.E_r.max(), (D.E_r < 0.9).sum()))
for y in ["E_r", "boundary", "Safe_Level_SMOTE", "SMOTE_IPF"]:
    print(y, {k: (round(spearmanr(D[y], D[k]).correlation, 2), round(spearmanr(D[y], D[k]).pvalue, 3)) for k in ["IR", "n_min", "safe", "border", "rare", "outlier"]})
if "gap" in D:
    eps = np.sqrt(np.log(24) / 1000) + np.sqrt(np.log(6) / 1000)
    G = D.dropna(subset=["gap"])
    print("gap: n=%d mean %.4f max %.4f eps %.4f all<=eps %s; value inside %s" % (len(G), G.gap.mean(), G.gap.max(), eps, (G.gap <= eps).all(),
          ((G.value >= G.lower - 1e-9) & (G.value <= G.upper + 1e-9)).all()))
    print("worst-split rescaled payoff: minimax %.3f, best pure %.3f, uniform %.3f" % (G.lower.mean(), G.best_pure_worst.mean(), G.uniform_worst.mean()))
    print("full-data fit time: median %.1f max %.1f" % (D.time.median(), D.time.max()))
