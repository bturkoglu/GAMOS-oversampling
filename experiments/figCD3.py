"""Dataset-level CD diagrams for Collections C, A, B (19 methods). Usage: python -I figCD3.py <resC> <resA> <resB> <out_stem>"""
import os, sys, glob
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs.py")).read()
exec(src[src.index("NICE ="):src.index("CLF =")]); exec(src[src.index("def cd_diagram"):src.index("fig, axes")].replace("0.085", "0.068"))
NICE.update({"Original": "No resampling", "PORT": "PORT", "GAMOS": "GAMOS"})
BASE = ["Original", "ROS", "SMOTE", "Borderline_SMOTE1", "Borderline_SMOTE2", "ADASYN", "Safe_Level_SMOTE", "SMOTE_D", "SMOTE_RSB",
        "A_SUWO", "ProWSyn", "polynom_fit_SMOTE_star", "SMOTE_IPF", "MWMOTE", "G_SMOTE", "LVQ_SMOTE", "CCR"]
MAIN = BASE + ["PORT", "GAMOS"]; SET = [(c, m) for c in ["kNN", "LinearSVM", "MLP", "RF"] for m in ["auc", "gmean"]]
_cd = cd_diagram
def cd2(ax, ranks, cd, title):  # highlight both portfolio methods
    _cd(ax, ranks, cd, title)
    for t in ax.texts:
        if t.get_text().startswith("PORT ") or t.get_text().startswith("GAMOS "): t.set_color("#E76F51"); t.set_fontweight("bold")
fig, axes = plt.subplots(3, 1, figsize=(7.5, 10.5))
for ax, (res, name) in zip(axes, [(sys.argv[1], "Collection C (clean test, 21 datasets)"), (sys.argv[2], "Collection A (32 datasets)"), (sys.argv[3], "Collection B (33 datasets)")]):
    df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(res, "parts", "*.csv"))]).drop_duplicates(["dataset", "rep", "fold", "method", "clf"], keep="last")
    df["method"] = df.method.replace({"G6-svp": "GAMOS", "G6-sv-uniform": "PORT"})
    DR = sum(df[(df.clf == c) & df.method.isin(MAIN)].groupby(["dataset", "method"])[m].mean().unstack()[MAIN].rank(axis=1, ascending=False) for c, m in SET) / len(SET)
    R = DR.rank(axis=1).mean(); N, k = DR.shape
    cd = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * N))
    p = stats.friedmanchisquare(*[DR[x] for x in MAIN]).pvalue
    cd2(ax, R.to_dict(), cd, f"{name}; Friedman p = {p:.1e}")
plt.tight_layout(); plt.savefig(sys.argv[4] + ".png", dpi=300); plt.savefig(sys.argv[4] + ".pdf"); print("ok")
