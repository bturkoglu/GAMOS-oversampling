"""GAMOS v6: the generator's pure strategies are (generator, proportion) pairs.
generator: either a GAMOS-v4 interpolation rule (theta, partner) or an oversampler from smote-variants;
proportion r: the strategy generates r * (n_maj - n_min) synthetic minority examples.
The game (nature picks validation splits, Hedge vs Hedge) learns both HOW and HOW MUCH to oversample.
Final set: each strategy k contributes round(p_bar[k] * r_k * gap) synthetic examples.
Proxies: logistic regression + depth-5 decision tree (no k-NN)."""
import numpy as np, logging
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
import gamos2, gamos3, gamos4

SV_PORTFOLIO = ("ProWSyn", "polynom_fit_SMOTE_star", "SMOTE_IPF", "CCR", "LVQ_SMOTE", "Safe_Level_SMOTE", "MWMOTE", "SMOTE")


class GAMOS:
    def __init__(self, generators="v4", proportions=(1.0,), n_val=6, T_game=500, lam=0.5, mode="minimax", random_state=None):
        self.generators = generators; self.proportions = proportions; self.n_val = n_val; self.T_game = T_game
        self.lam = lam; self.mode = mode; self.random_state = random_state

    # --- generators -------------------------------------------------------
    def _gen(self, g, X, y, pos, n, rng):
        """Return n synthetic minority rows for training set (X, y) in standardised space."""
        if n <= 0:
            return np.empty((0, X.shape[1]))
        if isinstance(g, tuple):  # v4 interpolation rule
            v4 = self._v4; v4.strategies_ = [g]
            A = v4._anchor_weights(X, y, pos)
            return v4._generate(X[y == pos], A, n, np.zeros(n, int), rng)
        import smote_variants as sv
        logging.getLogger("smote_variants").setLevel(logging.ERROR)
        try:
            Xr, yr = getattr(sv, g)(random_state=int(rng.randint(1 << 30))).sample(X, y)
            S = np.asarray(Xr[len(X):], float)
        except Exception:
            S = np.empty((0, X.shape[1]))
        if len(S) == 0:
            return self._gen((0.0, "knn"), X, y, pos, n, rng)
        idx = rng.choice(len(S), size=n, replace=n > len(S))
        return S[idx]

    def _gmean_proxies(self, Xa, ya, Xv, yv, pos):
        g = []
        for clf in (LogisticRegression(max_iter=300), DecisionTreeClassifier(max_depth=5, random_state=0)):
            try: g.append(gamos2._gmean(yv, clf.fit(Xa, ya).predict(Xv), pos))
            except Exception: g.append(0.0)
        return float(np.mean(g))

    def fit_resample(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, float); y = np.asarray(y)
        lab, cnt = np.unique(y, return_counts=True)
        if len(lab) < 2: return X, y
        pos = lab[np.argmin(cnt)]; gap = int(cnt.max() - cnt.min())
        if gap <= 0: return X, y
        sc = StandardScaler().fit(X); Xs = sc.transform(X)
        self._v4 = gamos3.GAMOS(random_state=self.random_state)
        w = np.ones(X.shape[1])
        try:
            c = np.abs(LogisticRegression(penalty="l1", solver="liblinear").fit(Xs, y).coef_[0])
            if c.sum() > 0: w = c / c.mean()
        except Exception: pass
        self._v4.w_ = w
        gens = [(t, p) for t in (0.0, 0.5, 1.0, 2.0) for p in ("knn", "level")] if self.generators == "v4" else list(self.generators)
        self.strategies_ = [(g, r) for g in gens for r in self.proportions]
        K = len(self.strategies_); p_bar = np.full(K, 1.0 / K); self.M_ = None
        if self.mode != "uniform" and cnt.min() >= 4:
            splits = list(StratifiedShuffleSplit(self.n_val, test_size=0.3, random_state=rng.randint(1 << 30)).split(Xs, y))
            M = np.zeros((K, len(splits)))
            for j, (tr, va) in enumerate(splits):
                Xt, yt = Xs[tr], y[tr]; gap_t = int(np.sum(yt != pos) - np.sum(yt == pos)); Xmin_t = Xt[yt == pos]
                pool = {}
                for gi, g in enumerate(gens):
                    pool[gi] = self._gen(g, Xt, yt, pos, gap_t, rng)
                for k, (g, r) in enumerate(self.strategies_):
                    S = pool[gens.index(g)][: int(round(r * gap_t))]
                    Xa = np.vstack([Xt, S]); ya = np.r_[yt, np.full(len(S), pos)]
                    cov = gamos2.GAMOS._coverage(Xmin_t, S) if len(S) else 0.0
                    M[k, j] = self._gmean_proxies(Xa, ya, Xs[va], y[va], pos) + self.lam * cov
            self.M_ = M
            if self.mode == "minimax": p_bar, _ = gamos4.GAMOS.solve(M, self.T_game)
            else: p_bar = np.eye(K)[np.argmax(M.mean(1))]
        self.mixed_strategy_ = p_bar
        parts = []
        for gi, g in enumerate(gens):
            n_g = int(round(sum(p_bar[k] * r * gap for k, (gg, r) in enumerate(self.strategies_) if gg == g)))
            if n_g > 0: parts.append(self._gen(g, Xs, y, pos, n_g, rng))
        S = np.vstack(parts) if parts else np.empty((0, X.shape[1]))
        return np.vstack([X, sc.inverse_transform(S)]) if len(S) else X, np.r_[y, np.full(len(S), pos)]

    sample = fit_resample
