"""GAMOS v3: the generator's pure strategies are (theta, partner):
  anchor allocation over proximity levels l = 1..L of the minority class (level 1 = minority points closest
  to the majority; ProWSyn-style peeling), P(level l) ∝ exp(-theta (l-1)), theta in {0, 0.5, 1, 2};
  partner rule: 'knn' (one of the k nearest minority neighbours, SMOTE) or 'level' (random point of the same level).
theta=1 with 'level' is ProWSyn's allocation; theta=0 with 'knn' is SMOTE-like. Everything else as in gamos2,
plus the utility is averaged over n_splits fresh stratified inner splits per round."""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedShuffleSplit
import gamos2


class GAMOS(gamos2.GAMOS):
    def __init__(self, thetas=(0.0, 0.5, 1.0, 2.0), partners=("knn", "level"), L=5, n_splits=3, **kw):
        super().__init__(**kw)
        self.thetas = thetas; self.partners = partners; self.L = L; self.n_splits = n_splits

    def _levels(self, X, y, pos):
        Xmaj = X[y != pos]; P = np.where(y == pos)[0]; lev = np.zeros(len(y), int)
        for l in range(1, self.L):
            if len(P) == 0:
                break
            k = min(len(P), self.k_neighbors)
            idx = np.unique(NearestNeighbors(n_neighbors=k).fit(X[P]).kneighbors(Xmaj, return_distance=False))
            lev[P[idx]] = l; P = np.delete(P, idx)
        lev[P] = self.L if len(P) else lev[P]
        return lev[y == pos]                     # level of every minority point (1..L)

    def _anchor_weights(self, X, y, pos):
        return {"lev": self._levels(X, y, pos)}

    def _generate(self, Xmin, A, n, strat, rng):
        m = len(Xmin); lev = A["lev"]
        if m == 1:
            return Xmin[np.zeros(n, int)] + rng.normal(size=(n, Xmin.shape[1])) * self.sigma * self.w_
        k = min(self.k_neighbors, m - 1)
        nbr = NearestNeighbors(n_neighbors=k + 1).fit(Xmin).kneighbors(Xmin, return_distance=False)[:, 1:]
        levels = np.unique(lev)
        a = np.empty(n, int); b = np.empty(n, int)
        for s in np.unique(strat):
            sel = np.where(strat == s)[0]; theta, partner = self.strategies_[s]
            pl = np.exp(-theta * (levels - 1.0)); pl /= pl.sum()
            chosen = levels[rng.choice(len(levels), size=len(sel), p=pl)]
            for L_ in np.unique(chosen):
                ss = sel[chosen == L_]; members = np.where(lev == L_)[0]
                a[ss] = members[rng.randint(0, len(members), size=len(ss))]
                if partner == "level" and len(members) > 1:
                    bb = members[rng.randint(0, len(members), size=len(ss))]
                    same = bb == a[ss]
                    bb[same] = nbr[a[ss][same], 0]
                    b[ss] = bb
                else:
                    b[ss] = nbr[a[ss], rng.randint(0, k, size=len(ss))]
        alpha = rng.uniform(0, 1, size=n)
        Xn = Xmin[a] + alpha[:, None] * (Xmin[b] - Xmin[a])
        return Xn + rng.normal(size=Xn.shape) * self.sigma * self.w_[None, :]

    def _utility(self, Xtr, ytr, S, Xva, yva, pos):
        # Xtr/ytr here is the FULL scaled training set; average over n_splits inner splits
        return self._u

    def fit_resample(self, X, y):
        self.strategies_ = None
        self.rules = ("lev",); self.amax_grid = (1.0,)
        orig = gamos2.GAMOS.fit_resample
        # override strategy list after parent builds it: patch via property-like hook
        return self._fit(X, y)

    def _fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, float); y = np.asarray(y)
        labels, counts = np.unique(y, return_counts=True)
        if len(labels) < 2:
            return X, y
        pos = labels[np.argmin(counts)]; n_syn = int(counts.max() - counts.min())
        if n_syn <= 0:
            return X, y
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        sc = StandardScaler().fit(X); Xs = sc.transform(X); Xmin = Xs[y == pos]
        self.w_ = np.ones(X.shape[1])
        if self.feature_weighting:
            try:
                w = np.abs(LogisticRegression(penalty="l1", solver="liblinear").fit(Xs, y).coef_[0])
                if w.sum() > 0: self.w_ = w / w.mean()
            except Exception:
                pass
        self.strategies_ = [(t, p) for t in self.thetas for p in self.partners]
        K = len(self.strategies_); p_bar = np.full(K, 1.0 / K); self.payoff_history_ = []
        if self.use_game and len(Xmin) >= 2:
            eta = np.sqrt(8 * np.log(K) / self.n_rounds); logw = np.zeros(K); p_sum = np.zeros(K)
            den = (1.0 if self.use_detector else 0.0) + (self.beta if self.use_utility else 0.0) + self.lam
            can_split = counts.min() >= 4
            for t in range(self.n_rounds):
                p = np.exp(logw - logw.max()); p /= p.sum(); p_sum += p
                splits = list(StratifiedShuffleSplit(self.n_splits, test_size=0.3, random_state=rng.randint(1 << 30)).split(Xs, y)) if (self.use_utility and can_split) else [(np.arange(len(y)), None)]
                u = np.zeros(K)
                for tr, va in splits:
                    Xtr, ytr = Xs[tr], y[tr]; Xmin_tr = Xtr[ytr == pos]
                    if len(Xmin_tr) < 2:
                        Xmin_tr, Xtr, ytr, va = Xmin, Xs, y, None
                    A = self._anchor_weights(Xtr, ytr, pos)
                    n_tr = max(1, int(np.sum(ytr != pos) - np.sum(ytr == pos)))
                    for k in range(K):
                        S = self._generate(Xmin_tr, A, n_tr, np.full(n_tr, k), rng)
                        I = self._indistinguishability(Xmin_tr, S, rng) if self.use_detector else 0.0
                        U = gamos2.GAMOS._utility(self, Xtr, ytr, S, Xs[va], y[va], pos) if (self.use_utility and va is not None) else 0.0
                        u[k] += (I + self.beta * U + self.lam * self._coverage(Xmin_tr, S)) / den / len(splits)
                r = (u - u.min()) / (u.max() - u.min()) if u.max() > u.min() else np.full(K, 0.5)
                logw += eta * r; self.payoff_history_.append(u)
            p_bar = p_sum / p_sum.sum()
        self.mixed_strategy_ = p_bar
        A = self._anchor_weights(Xs, y, pos)
        n_cand = int(np.ceil(n_syn / self.keep)) if self.use_filter else n_syn
        C = self._generate(Xmin, A, n_cand, rng.choice(K, size=n_cand, p=p_bar), rng)
        if self.use_filter:
            C = self._filter(Xs, y, pos, C, n_syn)
        return np.vstack([X, sc.inverse_transform(C)]), np.r_[y, np.full(len(C), pos)]

    sample = fit_resample
