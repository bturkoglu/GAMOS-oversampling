"""Portfolio framing: dataset-level statistics for one collection. Usage: python -I stats3.py <res_dir> <tag> <out_dir>
Unit of analysis = dataset. For each dataset, a method's score is its average rank over the 8 (classifier, metric) settings.
Friedman / Iman-Davenport on these, control-method z-test (Holm), Wilcoxon signed-rank on per-dataset average ranks (Holm),
and effect sizes (mean difference in AUC and G-mean averaged over classifiers, with win/loss counts at dataset level)."""
import os, sys, glob
import numpy as np, pandas as pd
from scipy import stats
res, tag, out = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(out, exist_ok=True)
df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(res, "parts", "*.csv"))]).drop_duplicates(["dataset", "rep", "fold", "method", "clf"], keep="last")
df["method"] = df.method.replace({"GAMOS-final": "GAMOS-PL", "GAMOS-final-uniform": "PL-uniform", "G6-svp": "GAMOS", "G6-sv-uniform": "PORT",
                                  "G6-svp-uniform": "PORT-24", "G6-svp-mean": "Best-single", "G6-sv": "GAMOS-full"})
NICE = {"Borderline_SMOTE1": "B-SMOTE1", "Borderline_SMOTE2": "B-SMOTE2", "Safe_Level_SMOTE": "SL-SMOTE", "SMOTE_D": "SMOTE-D",
        "SMOTE_RSB": "SMOTE-RSB", "A_SUWO": "A-SUWO", "polynom_fit_SMOTE_star": "PF-SMOTE", "SMOTE_IPF": "SMOTE-IPF",
        "G_SMOTE": "G-SMOTE", "LVQ_SMOTE": "LVQ-SMOTE", "Original": "No resampling", "PORT": "**PORT**", "GAMOS": "**GAMOS**",
        "T2-ProWSyn": "Tuned ProWSyn", "T2-polynom_fit_SMOTE_star": "Tuned PF-SMOTE", "T2-SMOTE_IPF": "Tuned SMOTE-IPF"}
nm = lambda x: NICE.get(x, x)
BASE = ["Original", "ROS", "SMOTE", "Borderline_SMOTE1", "Borderline_SMOTE2", "ADASYN", "Safe_Level_SMOTE", "SMOTE_D", "SMOTE_RSB",
        "A_SUWO", "ProWSyn", "polynom_fit_SMOTE_star", "SMOTE_IPF", "MWMOTE", "G_SMOTE", "LVQ_SMOTE", "CCR"]
MAIN = BASE + ["PORT", "GAMOS"]
ABL = ["PORT", "PORT-24", "GAMOS", "GAMOS-full", "Best-single", "GAMOS-PL", "PL-uniform"]
TUN = ["PORT", "GAMOS", "ProWSyn", "T2-ProWSyn", "polynom_fit_SMOTE_star", "T2-polynom_fit_SMOTE_star", "SMOTE_IPF", "T2-SMOTE_IPF"]
SET = [(c, m) for c in ["kNN", "LinearSVM", "MLP", "RF"] for m in ["auc", "gmean"]]
CN = {"kNN": "kNN", "LinearSVM": "SVM", "MLP": "MLP", "RF": "RF"}; MN = {"auc": "AUC", "gmean": "GM"}
fp = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); a = np.empty_like(p); run = 0
    for i, k in enumerate(o): run = max(run, min(1, (len(p) - i) * p[k])); a[k] = run
    return a
def mat(c, m, ms): return df[(df.clf == c) & df.method.isin(ms)].groupby(["dataset", "method"])[m].mean().unstack()[ms].dropna()
def analyse(ms, ctrl, fname, caption_cols=True):
    ms = [x for x in ms if x in set(df.method)]
    per = {s: mat(*s, ms).rank(axis=1, ascending=False) for s in SET}
    DR = sum(per.values()) / len(SET)        # dataset x method average rank over the 8 settings
    N, k = DR.shape; chi = stats.friedmanchisquare(*[DR[x] for x in ms]).statistic
    FF = (N - 1) * chi / (N * (k - 1) - chi); pF = 1 - stats.f.cdf(FF, k - 1, (k - 1) * (N - 1))
    R2 = DR.rank(axis=1)                      # re-rank the averaged ranks within each dataset for the Friedman-based z-test
    Rm = R2.mean(); se = np.sqrt(k * (k + 1) / (6 * N)); oth = [x for x in ms if x != ctrl]
    z = np.array([(Rm[x] - Rm[ctrl]) / se for x in oth]); pz = holm(2 * (1 - stats.norm.cdf(np.abs(z))))
    pw = holm([stats.wilcoxon(DR[ctrl] - DR[x]).pvalue if (DR[ctrl] != DR[x]).any() else 1 for x in oth])
    eff = {}
    for x in ms:
        e = []
        for m in ["auc", "gmean"]:
            V = df[df.method.isin([ctrl, x])].groupby(["dataset", "clf", "method"])[m].mean().unstack().groupby("dataset").mean()
            d = V[ctrl] - V[x]; e += [d.mean(), int((d > 1e-9).sum()), int((d < -1e-9).sum())]
        eff[x] = e
    order = sorted(ms, key=lambda x: Rm[x])
    L = [f"| Method | Avg. rank | z | Holm p (z-test) | Holm p (Wilcoxon) | {nm(ctrl).strip('*')} − method, AUC (W/L) | {nm(ctrl).strip('*')} − method, G-mean (W/L) |", "|---|---|---|---|---|---|---|"]
    for x in order:
        if x == ctrl: L.append(f"| {nm(x)} | {Rm[x]:.2f} | | | | | |"); continue
        i = oth.index(x); e = eff[x]
        L.append(f"| {nm(x)} | {Rm[x]:.2f} | {z[i]:.2f} | {fp(pz[i])} | {fp(pw[i])} | {e[0]:+.4f} ({e[1]}/{e[2]}) | {e[3]:+.4f} ({e[4]}/{e[5]}) |")
    open(os.path.join(out, fname), "w").write("\n".join(L) + "\n")
    pd.DataFrame({"rank": Rm, "pz": pd.Series(dict(zip(oth, pz))), "pw": pd.Series(dict(zip(oth, pw))), "dauc": pd.Series({x: eff[x][0] for x in ms}), "dgm": pd.Series({x: eff[x][3] for x in ms})}).to_csv(os.path.join(out, fname.replace(".md", ".csv")))
    print(f"\n### {tag} {fname}: N={N}, k={k}, Iman-Davenport p={pF:.2e}"); print("\n".join(L))
    return Rm, dict(zip(oth, pz)), dict(zip(oth, pw))
r_main = analyse(MAIN, "PORT", f"table_{tag}_main.md")
analyse(ABL, "PORT", f"table_{tag}_ablation.md")
if all(x in set(df.method) for x in TUN) and df[df.method.str.startswith("T2-")].dataset.nunique() == df.dataset.nunique(): analyse(TUN, "PORT", f"table_{tag}_tuned.md")
# per-setting rank table (for the clean test): 19 methods x 8 settings, rank (mean)
R, V = {}, {}
for s in SET:
    M = mat(*s, MAIN); R[s] = M.rank(axis=1, ascending=False).mean(); V[s] = M.mean()
order = sorted(MAIN, key=lambda x: np.mean([R[s][x] for s in SET]))
L = ["| Method | " + " | ".join(f"{CN[c]} {MN[m]}" for c, m in SET) + " |", "|---" * (len(SET) + 1) + "|"]
for x in order:
    L.append(f"| {nm(x)} | " + " | ".join((f"**{R[s][x]:.2f}** ({V[s][x]:.3f})" if R[s][x] == R[s].min() else f"{R[s][x]:.2f} ({V[s][x]:.3f})") for s in SET) + " |")
open(os.path.join(out, f"table_{tag}_settings.md"), "w").write("\n".join(L) + "\n")
print(f"\n### {tag} settings"); print("\n".join(L))
# worst-case loss to best baseline (descriptive)
best = {m: df[df.method.isin(BASE)].groupby(["dataset", "clf", "method"])[m].mean().unstack().max(1) for m in ["auc", "gmean"]}
print(f"\n### {tag} loss to best baseline per (dataset, clf): mean / 90th pct / max")
for x in ABL:
    s = []
    for m in ["auc", "gmean"]:
        d = -(df[df.method == x].groupby(["dataset", "clf"])[m].mean() - best[m]).dropna(); s.append(f"{m} {d.mean():.3f}/{d.quantile(.9):.3f}/{d.max():.3f}")
    print(x, " | ".join(s))
# runtime table
T = df[df.method.isin(MAIN)].groupby(["method", "dataset"]).t_os.mean().groupby("method").agg(["mean", "median", "max"]).sort_values("mean")
open(os.path.join(out, f"table_{tag}_runtime.md"), "w").write("\n".join(["| Method | Mean (s) | Median over datasets (s) | Max (s) |", "|---|---|---|---|"] + [f"| {nm(x)} | {r['mean']:.2f} | {r['median']:.2f} | {r['max']:.1f} |" for x, r in T.iterrows()]) + "\n")
