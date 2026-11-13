"""GAMOS v4: oversampling-strategy choice as a finite two-player zero-sum matrix game.
Row player (generator): K pure strategies k = (theta, partner) [see gamos3].
Column player (adversary / nature): J validation splits of the training data.
Payoff M[k, j] = G-mean utility of proxy classifiers trained on (train_j + synthetic_k) and scored on val_j
                 + lam * coverage  (+ optional detector indistinguishability term).
Both players run Hedge for T_game rounds (Freund & Schapire 1999); the time-averaged row strategy p_bar is an
eps-approximate maximin (robust) mixed strategy with eps = O(sqrt(ln max(K,J) / T_game)).
mode: 'minimax' (game), 'mean' (adversary fixed uniform -> best average pure strategy), 'uniform' (no learning)."""
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import gamos2, gamos3


class GAMOS(gamos3.GAMOS):
    def __init__(self, n_val=6, T_game=500, mode="minimax", use_detector=False, **kw):
        super().__init__(use_detector=use_detector, **kw)
        self.n_val = n_val; self.T_game = T_game; self.mode = mode

    def _payoff_matrix(self, Xs, y, pos, rng):
        K = len(self.strategies_)
        if np.unique(y, return_counts=True)[1].min() < 4:
            return None
        splits = list(StratifiedShuffleSplit(self.n_val, test_size=0.3, random_state=rng.randint(1 << 30)).split(Xs, y))
        M = np.zeros((K, len(splits)))
        for j, (tr, va) in enumerate(splits):
            Xtr, ytr = Xs[tr], y[tr]; Xmin_tr = Xtr[ytr == pos]
            if len(Xmin_tr) < 2:
                return None
            A = self._anchor_weights(Xtr, ytr, pos); n_tr = max(1, int(np.sum(ytr != pos) - np.sum(ytr == pos)))
            for k in range(K):
                S = self._generate(Xmin_tr, A, n_tr, np.full(n_tr, k), rng)
                v = gamos2.GAMOS._utility(self, Xtr, ytr, S, Xs[va], y[va], pos) + self.lam * self._coverage(Xmin_tr, S)
                if self.use_detector:
                    v += self._indistinguishability(Xmin_tr, S, rng)
                M[k, j] = v
        return M

    @staticmethod
    def solve(M, T):
        """Hedge vs Hedge on payoff matrix M (row maximises, column minimises). Returns averaged strategies."""
        K, J = M.shape
        lo, hi = M.min(), M.max(); R = (M - lo) / (hi - lo) if hi > lo else np.full_like(M, 0.5)
        eta_r = np.sqrt(8 * np.log(K) / T); eta_c = np.sqrt(8 * np.log(J) / T)
        lr = np.zeros(K); lc = np.zeros(J); P = np.zeros(K); Q = np.zeros(J)
        for _ in range(T):
            p = np.exp(lr - lr.max()); p /= p.sum(); q = np.exp(lc - lc.max()); q /= q.sum()
            P += p; Q += q
            lr += eta_r * (R @ q); lc -= eta_c * (p @ R)
        return P / T, Q / T

    def _fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, float); y = np.asarray(y)
        labels, counts = np.unique(y, return_counts=True)
        if len(labels) < 2:
            return X, y
        pos = labels[np.argmin(counts)]; n_syn = int(counts.max() - counts.min())
        if n_syn <= 0:
            return X, y
        sc = StandardScaler().fit(X); Xs = sc.transform(X); Xmin = Xs[y == pos]
        self.w_ = np.ones(X.shape[1])
        if self.feature_weighting:
            try:
                w = np.abs(LogisticRegression(penalty="l1", solver="liblinear").fit(Xs, y).coef_[0])
                if w.sum() > 0: self.w_ = w / w.mean()
            except Exception:
                pass
        self.strategies_ = [(t, p) for t in self.thetas for p in self.partners]
        K = len(self.strategies_); p_bar = np.full(K, 1.0 / K); self.M_ = None
        if self.mode != "uniform" and len(Xmin) >= 2:
            M = self._payoff_matrix(Xs, y, pos, rng); self.M_ = M
            if M is not None:
                if self.mode == "minimax":
                    p_bar, self.q_bar_ = self.solve(M, self.T_game)
                else:  # 'mean': best pure strategy on the average split
                    p_bar = np.eye(K)[np.argmax(M.mean(axis=1))]
        self.mixed_strategy_ = p_bar
        A = self._anchor_weights(Xs, y, pos)
        n_cand = int(np.ceil(n_syn / self.keep)) if self.use_filter else n_syn
        C = self._generate(Xmin, A, n_cand, rng.choice(K, size=n_cand, p=p_bar), rng)
        if self.use_filter:
            C = self._filter(Xs, y, pos, C, n_syn)
        return np.vstack([X, sc.inverse_transform(C)]), np.r_[y, np.full(len(C), pos)]
