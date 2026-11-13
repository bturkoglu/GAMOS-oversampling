"""Figures from real benchmark results. Usage: python -I figs.py <res_dir> <out_dir> <tag>"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

res, out, tag = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
S = json.load(open(os.path.join(res, "analysis", "summary.json")))
NICE = {"Borderline_SMOTE1": "B-SMOTE1", "Borderline_SMOTE2": "B-SMOTE2", "Safe_Level_SMOTE": "SL-SMOTE", "SMOTE_D": "SMOTE-D",
        "SMOTE_RSB": "SMOTE-RSB", "A_SUWO": "A-SUWO", "polynom_fit_SMOTE_star": "PF-SMOTE", "SMOTE_IPF": "SMOTE-IPF",
        "G_SMOTE": "G-SMOTE", "LVQ_SMOTE": "LVQ-SMOTE"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
CLF = {"kNN": "kNN", "LinearSVM": "Linear SVM", "MLP": "MLP"}; MET = {"auc": "AUC", "gmean": "G-mean"}


def cd_diagram(ax, ranks, cd, title):
    names = list(ranks.keys()); r = np.array([ranks[n] for n in names]); k = len(r)
    lo, hi = 1, k
    ax.set_xlim(lo - 0.5, hi + 0.5); ax.set_ylim(0, 1); ax.axis("off")
    ax.hlines(0.8, lo, hi, color="k", lw=0.8)
    for t in range(lo, hi + 1):
        ax.vlines(t, 0.8, 0.83, color="k", lw=0.6); ax.text(t, 0.86, str(t), ha="center", fontsize=6)
    order = np.argsort(r); half = (k + 1) // 2
    for i, idx in enumerate(order):
        left = i < half; y = 0.72 - 0.085 * (i if left else k - 1 - i)
        x = r[idx]; xt = lo - 0.4 if left else hi + 0.4
        c = "#E76F51" if names[idx] == "GAMOS" else "k"; fw = "bold" if names[idx] == "GAMOS" else "normal"
        ax.plot([x, x, xt], [0.8, y, y], color=c, lw=0.6)
        ax.text(xt + (-0.05 if left else 0.05), y, f"{NICE.get(names[idx], names[idx])} ({x:.2f})", ha="right" if left else "left",
                va="center", fontsize=6, color=c, fontweight=fw)
    ax.plot([lo, lo + cd], [0.95, 0.95], color="k", lw=1.2); ax.text(lo + cd / 2, 0.97, f"CD = {cd:.2f}", ha="center", fontsize=6)
    # cliques: groups whose rank range < cd
    rs = np.sort(r); ylev = 0.76; used = []
    for i in range(k):
        j = i
        while j + 1 < k and rs[j + 1] - rs[i] < cd: j += 1
        if j > i and not any(a <= i and j <= b for a, b in used):
            ax.plot([rs[i] - 0.03, rs[j] + 0.03], [ylev, ylev], color="#2A9D8F", lw=2, alpha=0.8); ylev -= 0.02; used.append((i, j))
    ax.set_title(title, fontsize=8)


fig, axes = plt.subplots(3, 2, figsize=(11, 10.5))
for i, c in enumerate(CLF):
    for j, m in enumerate(MET):
        s = S["summary"][f"{c}_{m}"]
        cd_diagram(axes[i, j], s["rank"], s["nemenyi_cd"], f"{CLF[c]} – {MET[m]}  (Friedman p = {s['friedman_p']:.1e})")
plt.tight_layout(); plt.savefig(os.path.join(out, f"fig_cd_{tag}.png"), dpi=300); plt.savefig(os.path.join(out, f"fig_cd_{tag}.pdf"))

# ablation chart
A = S["ablation"]; labels = {"GAMOS-final-uniform": "uniform mixture\n(no learning)", "GAMOS-final-mean": "best average\npure strategy",
                             "GAMOS-final-det": "+ detector\nterm", "GAMOS-final-filter": "+ filter", "GAMOS-final-sigma0": "no noise",
                             "GAMOS-final-iso": "isotropic\nnoise"}
fig, ax = plt.subplots(figsize=(7.5, 3))
keys = [k for k in labels if k in next(iter(A.values()))["tests"]]
w = 0.13
for n, (sk, a) in enumerate(A.items()):
    d = [a["tests"][k]["diff"] for k in keys]
    ax.bar(np.arange(len(keys)) + (n - 2.5) * w, d, w, label=sk.replace("_", " ").replace("gmean", "G-mean").replace("auc", "AUC").replace("LinearSVM", "Lin. SVM"))
ax.axhline(0, color="k", lw=0.6); ax.set_xticks(np.arange(len(keys))); ax.set_xticklabels([labels[k] for k in keys])
ax.set_ylabel("GAMOS minus variant\n(mean over datasets)"); ax.legend(fontsize=6, ncol=3, frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(out, f"fig_ablation_{tag}.png"), dpi=300); plt.savefig(os.path.join(out, f"fig_ablation_{tag}.pdf"))

# runtime vs average rank
rt = S["runtime_mean_s"]; avg = {}
for k, s in S["summary"].items():
    for m, r in s["rank"].items(): avg.setdefault(m, []).append(r)
fig, ax = plt.subplots(figsize=(5, 3.4))
for m, r in avg.items():
    if m not in rt or m == "Original": continue
    c = "#E76F51" if m == "GAMOS" else "#1F3A5F"
    ax.scatter(max(rt[m], 1e-4), np.mean(r), color=c, s=18); ax.annotate(NICE.get(m, m), (max(rt[m], 1e-4), np.mean(r)), fontsize=6, xytext=(3, 2), textcoords="offset points", color=c)
ax.set_xscale("log"); ax.invert_yaxis(); ax.set_xlabel("mean CPU time per oversampling call (s, log scale)"); ax.set_ylabel("average rank over 6 settings\n(lower is better)")
plt.tight_layout(); plt.savefig(os.path.join(out, f"fig_runtime_{tag}.png"), dpi=300); plt.savefig(os.path.join(out, f"fig_runtime_{tag}.pdf"))
print("figures written")
