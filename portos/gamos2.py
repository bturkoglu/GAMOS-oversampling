"""GAMOS v2 - Game-Theoretic Adaptive Minority Oversampling Strategy.

Oversampling is played as a repeated two-player zero-sum game.
  Generator (row player): pure strategy k = (anchor rule, interpolation range).
     anchor rules decide WHERE minority samples are created:
       uniform   - every minority point equally (SMOTE)
       border    - points whose k-neighbourhood is mixed (danger set, Borderline-SMOTE idea)
       safe      - points whose neighbourhood is mostly minority (Safe-Level idea)
       proximity - points close to the majority class (ProWSyn idea)
       sparse    - points in sparse minority regions (density idea)
     interpolation range: alpha ~ U(0, a_max), a_max in {0.5, 1.0}
     every candidate also gets small feature-weighted Gaussian noise (scale sigma).
  Detector (column player): best-responds with a k-NN discriminator, real minority vs synthetic.
Generator payoff u_k = [I_k + beta*U_k + lam*C_k] / (1 + beta + lam)
  I_k indistinguishability (2 x balanced CV error of the detector, capped at 1)
  U_k utility: mean G-mean of LR and 5-NN proxies trained on real-train + synthetic, scored on a
      held-out real split (a fresh stratified split every round)
  C_k coverage of the real minority set.
U_k and C_k do not depend on the detector, so the game is strategically equivalent to a zero-sum game.
Generator plays full-information Hedge on payoffs rescaled to [0,1] per round; the final synthetic
set is drawn from the time-averaged mixed strategy, then filtered (overlap, near-duplicates) to exact balance.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils import check_random_state

RULES = ("uniform", "border", "safe", "proximity", "sparse")


def _gmean(y_true, y_pred, pos):
    p = y_true == pos
    se = np.mean(y_pred[p] == pos) if p.any() else 0.0
    sp = np.mean(y_pred[~p] != pos) if (~p).any() else 0.0
    return float(np.sqrt(se * sp))


class GAMOS:
    def __init__(self, k_neighbors=5, rules=RULES, amax_grid=(0.5, 1.0), sigma=0.05, n_rounds=10,
                 beta=1.0, lam=0.5, keep=0.8, use_game=True, use_detector=True, use_utility=True,
                 use_filter=True, feature_weighting=True, random_state=None):
        self.k_neighbors = k_neighbors; self.rules = rules; self.amax_grid = amax_grid; self.sigma = sigma
        self.n_rounds = n_rounds; self.beta = beta; self.lam = lam; self.keep = keep
        self.use_game = use_game; self.use_detector = use_detector; self.use_utility = use_utility
        self.use_filter = use_filter; self.feature_weighting = feature_weighting; self.random_state = random_state

    # ---------- anchor weights ----------
    def _anchor_weights(self, X, y, pos):
        """Return dict rule -> probability vector over minority points of (X, y)."""
        Xmin = X[y == pos]; m = len(Xmin)
        k = min(self.k_neighbors, len(X) - 1)
        idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(Xmin, return_distance=False)[:, 1:]
        maj = (y[idx] != pos).sum(axis=1)                       # majority neighbours of each minority point
        Xmaj = X[y != pos]
        dmaj = NearestNeighbors(n_neighbors=1).fit(Xmaj).kneighbors(Xmin)[0][:, 0] if len(Xmaj) else np.ones(m)
        if m > 1:
            km = min(self.k_neighbors, m - 1)
            dmin = NearestNeighbors(n_neighbors=km + 1).fit(Xmin).kneighbors(Xmin)[0][:, 1:].mean(axis=1)
        else:
            dmin = np.ones(m)
        eps = 1e-12
        W = {"uniform": np.ones(m),
             "border": np.where((maj > 0) & (maj < k), maj / k, 0.0),
             "safe": (k - maj) / k,
             "proximity": 1.0 / (dmaj + eps) * (maj < k),
             "sparse": dmin * (maj < k)}
        out = {}
        for r in self.rules:
            w = np.asarray(W[r], float)
            if w.sum() <= eps:
                w = np.ones(m)                                   # rule undefined here -> fall back to uniform
            out[r] = w / w.sum()
        return out

    # ---------- generation ----------
    def _generate(self, Xmin, A, n, strat, rng):
        """strat: array of strategy indices; A: anchor-probability dict for Xmin."""
        m = len(Xmin)
        if m == 1:
            return Xmin[np.zeros(n, int)] + rng.normal(size=(n, Xmin.shape[1])) * self.sigma * self.w_
        k = min(self.k_neighbors, m - 1)
        nbr = NearestNeighbors(n_neighbors=k + 1).fit(Xmin).kneighbors(Xmin, return_distance=False)[:, 1:]
        a = np.empty(n, int); amax = np.empty(n)
        for s in np.unique(strat):
            sel = strat == s; rule, am = self.strategies_[s]
            a[sel] = rng.choice(m, size=sel.sum(), p=A[rule]); amax[sel] = am
        b = nbr[a, rng.randint(0, k, size=n)]
        alpha = rng.uniform(0, 1, size=n) * amax
        Xn = Xmin[a] + alpha[:, None] * (Xmin[b] - Xmin[a])
        return Xn + rng.normal(size=Xn.shape) * self.sigma * self.w_[None, :]

    # ---------- payoff components ----------
    def _indistinguishability(self, Xmin, S, rng):
        n = min(len(Xmin), len(S))
        if n < 3:
            return 1.0
        Z = np.vstack([Xmin[rng.choice(len(Xmin), n, replace=False)], S[rng.choice(len(S), n, replace=False)]])
        t = np.r_[np.zeros(n), np.ones(n)]
        accs = []
        for tr, te in StratifiedKFold(n_splits=3, shuffle=True, random_state=rng.randint(1 << 30)).split(Z, t):
            clf = KNeighborsClassifier(n_neighbors=min(5, len(tr))).fit(Z[tr], t[tr])
            p = clf.predict(Z[te]); tt = t[te]
            accs.append(0.5 * (np.mean(p[tt == 0] == 0) + np.mean(p[tt == 1] == 1)))
        return float(min(1.0, 2.0 * (1.0 - np.mean(accs))))

    def _utility(self, Xtr, ytr, S, Xva, yva, pos):
        Xa = np.vstack([Xtr, S]); ya = np.r_[ytr, np.full(len(S), pos)]
        g = []
        for clf in (LogisticRegression(max_iter=300), KNeighborsClassifier(n_neighbors=min(5, len(Xa)))):
            try:
                g.append(_gmean(yva, clf.fit(Xa, ya).predict(Xva), pos))
            except Exception:
                g.append(0.0)
        return float(np.mean(g))

    @staticmethod
    def _coverage(Xmin, S):
        idx = NearestNeighbors(n_neighbors=1).fit(Xmin).kneighbors(S, return_distance=False)[:, 0]
        return len(np.unique(idx)) / len(Xmin)

    # ---------- filter ----------
    def _filter(self, X, y, pos, C, n_keep):
        if len(C) <= n_keep:
            return C
        k = min(self.k_neighbors, len(X) - 1)
        idx = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(C, return_distance=False)
        overlap = (y[idx] != pos).all(axis=1)                  # no minority among the k real neighbours
        d = NearestNeighbors(n_neighbors=2).fit(np.vstack([X, C])).kneighbors(C)[0][:, 1]
        order = np.lexsort((-d, overlap))                        # keep non-overlap first, least redundant first
        return C[order[:n_keep]]

    # ---------- main ----------
    def fit_resample(self, X, y):
        rng = check_random_state(self.random_state)
        X = np.asarray(X, float); y = np.asarray(y)
        labels, counts = np.unique(y, return_counts=True)
        if len(labels) < 2:
            return X, y
        pos = labels[np.argmin(counts)]
        n_syn = int(counts.max() - counts.min())
        if n_syn <= 0:
            return X, y
        sc = StandardScaler().fit(X); Xs = sc.transform(X)
        Xmin = Xs[y == pos]
        self.w_ = np.ones(X.shape[1])
        if self.feature_weighting:
            try:
                w = np.abs(LogisticRegression(penalty="l1", solver="liblinear", C=1.0).fit(Xs, y).coef_[0])
                if w.sum() > 0:
                    self.w_ = w / w.mean()
            except Exception:
                pass
        self.strategies_ = [(r, a) for r in self.rules for a in self.amax_grid]
        K = len(self.strategies_)
        p_bar = np.full(K, 1.0 / K)
        self.payoff_history_ = []
        can_split = counts.min() >= 4
        if self.use_game and len(Xmin) >= 2:
            eta = np.sqrt(8 * np.log(K) / self.n_rounds)
            logw = np.zeros(K); p_sum = np.zeros(K)
            den = (1.0 if self.use_detector else 0.0) + (self.beta if self.use_utility else 0.0) + self.lam
            for t in range(self.n_rounds):
                p = np.exp(logw - logw.max()); p /= p.sum(); p_sum += p
                if self.use_utility and can_split:
                    tr, va = next(StratifiedShuffleSplit(1, test_size=0.3, random_state=rng.randint(1 << 30)).split(Xs, y))
                    Xtr, ytr, Xva, yva = Xs[tr], y[tr], Xs[va], y[va]
                else:
                    Xtr, ytr, Xva, yva = Xs, y, None, None
                Xmin_tr = Xtr[ytr == pos]
                if len(Xmin_tr) < 2:
                    Xmin_tr, Xtr, ytr, Xva = Xmin, Xs, y, None
                A = self._anchor_weights(Xtr, ytr, pos)
                n_tr = max(1, int(np.sum(ytr != pos) - np.sum(ytr == pos)))
                u = np.zeros(K)
                for k in range(K):
                    S = self._generate(Xmin_tr, A, n_tr, np.full(n_tr, k), rng)
                    I = self._indistinguishability(Xmin_tr, S, rng) if self.use_detector else 0.0
                    U = self._utility(Xtr, ytr, S, Xva, yva, pos) if (self.use_utility and Xva is not None) else 0.0
                    u[k] = (I + self.beta * U + self.lam * self._coverage(Xmin_tr, S)) / den
                r = (u - u.min()) / (u.max() - u.min()) if u.max() > u.min() else np.full(K, 0.5)
                logw += eta * r
                self.payoff_history_.append(u)
            p_bar = p_sum / p_sum.sum()
        self.mixed_strategy_ = p_bar
        A = self._anchor_weights(Xs, y, pos)
        n_cand = int(np.ceil(n_syn / self.keep)) if self.use_filter else n_syn
        C = self._generate(Xmin, A, n_cand, rng.choice(K, size=n_cand, p=p_bar), rng)
        if self.use_filter:
            C = self._filter(Xs, y, pos, C, n_syn)
        C = sc.inverse_transform(C)
        return np.vstack([X, C]), np.r_[y, np.full(len(C), pos)]

    sample = fit_resample
