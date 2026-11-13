"""Matched-budget control: a baseline oversampler picks 1 of 8 parameter settings using the same protocol GAMOS
uses for its payoff (6 stratified 70/30 inner splits; mean G-mean of LR and 5-NN proxies on val + 0.5*coverage
is NOT used here: plain proxy G-mean averaged over the 6 splits). The chosen setting is then fit on all training data."""
import numpy as np, logging
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import smote_variants as sv
logging.getLogger("smote_variants").setLevel(logging.ERROR)

GRIDS = {
    "ProWSyn": [dict(n_neighbors=k, theta=t) for k in (3, 5) for t in (0.1, 0.5, 1.0, 2.0)],
    "polynom_fit_SMOTE_star": [dict(topology="star"), dict(topology="bus"), dict(topology="mesh")] + [dict(topology="poly", order=o) for o in (1, 2, 3, 4, 5)],
    "SMOTE_IPF": [dict(n_neighbors=k, n_folds=f) for k in (3, 5, 7, 9) for f in (5, 9)],
}

def _g(y, p, pos):
    s = y == pos
    return np.sqrt(np.mean(p[s] == pos) * np.mean(p[~s] != pos)) if s.any() and (~s).any() else 0.0

def make(method, params, seed):
    if method == "polynom_fit_SMOTE_star":
        kw = {k: v for k, v in params.items() if k != "topology"}
        return getattr(sv, "polynom_fit_SMOTE_" + params["topology"])(random_state=seed, **kw)
    return getattr(sv, method)(random_state=seed, **params)

def tuned_resample(method, X, y, seed, n_val=6):
    lab, cnt = np.unique(y, return_counts=True); pos = lab[np.argmin(cnt)]
    grid = GRIDS[method]
    if cnt.min() < 4:
        return make(method, grid[0], seed).sample(X, y)
    rng = np.random.RandomState(seed)
    score = np.zeros(len(grid))
    for tr, va in StratifiedShuffleSplit(n_val, test_size=0.3, random_state=rng.randint(1 << 30)).split(X, y):
        for i, prm in enumerate(grid):
            try:
                Xr, yr = make(method, prm, seed).sample(X[tr], y[tr])
                g = [_g(y[va], c.fit(Xr, yr).predict(X[va]), pos) for c in (LogisticRegression(max_iter=300), KNeighborsClassifier(5))]
                score[i] += np.mean(g)
            except Exception:
                pass
    return make(method, grid[int(np.argmax(score))], seed).sample(X, y)
