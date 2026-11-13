import sys, os, pandas as pd
d = sys.argv[1]; fp = lambda p: "" if pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")
NICE = {"Borderline_SMOTE1": "B-SMOTE1", "Borderline_SMOTE2": "B-SMOTE2", "Safe_Level_SMOTE": "SL-SMOTE", "SMOTE_D": "SMOTE-D",
        "SMOTE_RSB": "SMOTE-RSB", "A_SUWO": "A-SUWO", "polynom_fit_SMOTE_star": "PF-SMOTE", "SMOTE_IPF": "SMOTE-IPF",
        "G_SMOTE": "G-SMOTE", "LVQ_SMOTE": "LVQ-SMOTE", "Original": "No resampling", "PORT": "**PORT**", "GAMOS": "**GAMOS**"}
A = pd.read_csv(os.path.join(d, "table_a_main.csv"), index_col=0); B = pd.read_csv(os.path.join(d, "table_b_main.csv"), index_col=0)
order = (A["rank"] + B["rank"]).sort_values().index
L = ["| Method | A: rank | A: Holm p | A: ΔAUC | A: ΔG-mean | B: rank | B: Holm p | B: ΔAUC | B: ΔG-mean |", "|---|---|---|---|---|---|---|---|---|"]
for x in order:
    f = lambda T: f"{T.loc[x,'rank']:.2f} | {fp(T.loc[x,'pw'])} | " + ("" if x == "PORT" else f"{T.loc[x,'dauc']:+.4f}") + " | " + ("" if x == "PORT" else f"{T.loc[x,'dgm']:+.4f}")
    L.append(f"| {NICE.get(x, x)} | {f(A)} | {f(B)} |")
open(os.path.join(d, "table_ab_main.md"), "w").write("\n".join(L) + "\n")
T = {c: pd.read_csv(os.path.join(d, f"table_{c}_ablation.csv"), index_col=0) for c in "cab"}
L = ["| Variant | C: rank | C: Holm p | A: rank | A: Holm p | B: rank | B: Holm p |", "|---|---|---|---|---|---|---|"]
for x in ["PORT", "PORT-24", "GAMOS", "GAMOS-full", "Best-single", "GAMOS-PL", "PL-uniform"]:
    L.append(f"| {NICE.get(x, x)} | " + " | ".join(f"{T[c].loc[x,'rank']:.2f} | {fp(T[c].loc[x,'pw'])}" for c in "cab") + " |")
open(os.path.join(d, "table_abl3.md"), "w").write("\n".join(L) + "\n"); print(open(os.path.join(d, "table_abl3.md")).read())
