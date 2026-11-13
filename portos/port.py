"""Portfolio oversampling with equal weights (PORT) and with game-theoretic weights (GAMOS).

Both are thin wrappers around the frozen implementation in gamos6.py (checksums in ../frozen/).
PORT  = GAMOS(generators=SV_PORTFOLIO, proportions=(1.0,), mode="uniform")   # 'G6-sv-uniform' in the result files
GAMOS = GAMOS(generators=SV_PORTFOLIO, proportions=(0.25, 0.5, 1.0))          # 'G6-svp' in the result files
"""
from gamos6 import GAMOS as _G6, SV_PORTFOLIO


class PORT(_G6):
    """Equal-weight portfolio of eight SMOTE variants; balances the classes. API: fit_resample(X, y) -> (X_res, y_res)."""
    def __init__(self, random_state=None):
        super().__init__(generators=SV_PORTFOLIO, proportions=(1.0,), mode="uniform", random_state=random_state)


class GAMOS(_G6):
    """Portfolio whose weights and sampling ratio are learned by a zero-sum game solved with Hedge."""
    def __init__(self, random_state=None, n_val=6, T_game=500, lam=0.5):
        super().__init__(generators=SV_PORTFOLIO, proportions=(0.25, 0.5, 1.0), n_val=n_val, T_game=T_game, lam=lam,
                         mode="minimax", random_state=random_state)
