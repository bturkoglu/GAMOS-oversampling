"""GAMOS v2 on each full Collection C dataset: learned mixed strategy, payoff matrix, duality gap vs LP value,
minority example types. Usage: python -I stratC.py <exp_dir> <out_csv>"""
import sys, warnings, time; warnings.filterwarnings("ignore"); sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linprog
from sklearn.neighbors import NearestNeighbors
from run import load, IMBL
from gamos6 import GAMOS, SV_PORTFOLIO
from gamos4 import GAMOS as G4
def one(ds):
    X, y = load(ds); t = time.time()
    g = GAMOS(generators=SV_PORTFOLIO, proportions=(0.25, 0.5, 1.0), random_state=0); g.fit_resample(X, y); t = time.time() - t
    lab, cnt = np.unique(y, return_counts=True); pos = lab[np.argmin(cnt)]
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    idx = NearestNeighbors(n_neighbors=6).fit(Xs).kneighbors(Xs[y == pos], return_distance=False)[:, 1:]
    maj = (y[idx] != pos).sum(1)
    row = dict(dataset=ds, n=len(y), d=X.shape[1], n_min=int(cnt.min()), IR=cnt.max() / cnt.min(), time=t,
               safe=np.mean(maj <= 1), border=np.mean((maj >= 2) & (maj <= 3)), rare=np.mean(maj == 4), outlier=np.mean(maj == 5))
    for (o, r), w in zip(g.strategies_, g.mixed_strategy_): row[f"{o}|{r}"] = w
    M = g.M_
    if M is not None:
        lo, hi = M.min(), M.max(); R = (M - lo) / (hi - lo) if hi > lo else M * 0 + .5
        p, q = G4.solve(M, 500); K, J = R.shape
        res = linprog(np.r_[np.zeros(K), -1], A_ub=np.c_[-R.T, np.ones(J)], b_ub=np.zeros(J), A_eq=np.r_[np.ones(K), 0][None], b_eq=[1], bounds=[(0, None)] * K + [(None, None)])
        row.update(value=-res.fun, lower=(R.T @ p).min(), upper=(R @ q).max(), raw_range=hi - lo,
                   best_pure_worst=R.min(1).max(), uniform_worst=R.mean(0).min())
        row["gap"] = row["upper"] - row["lower"]
    print(ds, round(t, 1), flush=True)
    return row
D = pd.DataFrame(Parallel(n_jobs=3)(delayed(one)(ds) for ds in ["imbl:" + d if not d.startswith("imbl:") else d for d in IMBL])); D.to_csv(sys.argv[2], index=False)
eps = np.sqrt(np.log(24) / 1000) + np.sqrt(np.log(6) / 1000)
print(D[["gap", "raw_range", "time"]].describe().round(4).to_string()); print("eps", round(eps, 4), "share gap<=eps", (D.gap <= eps).mean(),
      "value in [lower,upper]", ((D.value >= D.lower - 1e-9) & (D.value <= D.upper + 1e-9)).mean())
print("worst-split payoff (rescaled): minimax", D.lower.mean().round(4), "best pure", D.best_pure_worst.mean().round(4), "uniform", D.uniform_worst.mean().round(4))
