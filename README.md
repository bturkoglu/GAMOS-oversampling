# PORT and GAMOS: portfolio oversampling for imbalanced data classification (v2.0)

This repository holds the code, the frozen design files and all per-fold results for the paper

> B. Türkoğlu, "Mixing SMOTE variants instead of choosing one: portfolio oversampling for imbalanced data classification" (under review).

- **PORT** builds the training set from the synthetic minority examples of eight SMOTE variants: SMOTE, ProWSyn, PF-SMOTE, SMOTE-IPF, CCR, LVQ-SMOTE, Safe-Level-SMOTE and MWMOTE. Each variant contributes the same share of the class gap. This is the recommended method.
- **GAMOS** uses the same portfolio. It learns the weights and the amount of oversampling by solving a zero-sum game against an adversarial choice of validation split, with the Hedge algorithm.

## Quick start

```python
import sys; sys.path.insert(0, "portos")
from port import PORT, GAMOS

X_res, y_res = PORT(random_state=0).fit_resample(X, y)    # equal weights
X_res, y_res = GAMOS(random_state=0).fit_resample(X, y)   # learned weights
```

Install the pinned dependencies with `pip install -r requirements.txt`. The oversamplers come from [smote-variants](https://github.com/analyticalmindsltd/smote_variants) 1.0.1.

Because each member's share is rounded separately, the two classes can differ by a few examples after PORT.

## Contents

| Folder | Content |
|---|---|
| `portos/` | Implementation. `port.py` is the public API; `gamos6.py` and its helpers `gamos2.py`, `gamos3.py` and `gamos4.py` are the frozen code. |
| `frozen/` | SHA-1 checksums of the frozen code (`FROZEN_v2_design.sha`), recorded before the test runs. Check them with `cd portos && shasum -a 1 gamos2.py gamos3.py gamos4.py gamos6.py`. |
| `experiments/` | `run.py` is the benchmark harness: datasets, cross-validation and the 19 methods. `tuned2.py` is the matched-budget tuning. `stats3.py` and `combine3.py` produce the statistics and tables. The `fig_*.py` scripts draw the figures. |
| `results/` | Per-fold results for Collection C (`res_c`) and Collections A and B (`res_a4`, `res_b4`), the paper's tables (`tables/`) and the dataset lists. |
| `legacy/` | The first public version (v1.0.1), kept for reference. |

## Method names in the result files

| File name | Paper name |
|---|---|
| `G6-sv-uniform` | PORT |
| `G6-svp` | GAMOS |
| `G6-svp-uniform` | PORT-24 |
| `G6-svp-mean` | Best-single |
| `G6-sv` | GAMOS-full |
| `GAMOS-final` | GAMOS-PL |
| `GAMOS-final-uniform` | PL-uniform |
| `T2-<method>` | Tuned baseline (matched budget) |

## Reproducing the experiments

```bash
# Collection C (imbalanced-learn datasets; downloaded to $IMBL_DATA, default ~/imbl_data)
CLFS=kNN,LinearSVM,MLP,RF DSET=imbl python experiments/run.py results_c 1 8 G6-sv-uniform,G6-svp,SMOTE,ProWSyn
# Collection A (KEEL) and Collection B (KEEL, DSET=test2), 3 repeats
CLFS=kNN,LinearSVM,MLP,RF python experiments/run.py results_a 3 8 G6-sv-uniform,G6-svp
python experiments/stats3.py results_c c tables_out
```

The arguments of `run.py` are the output folder, the number of CV repeats, the number of parallel jobs and a comma-separated method list.

## Notes on the libraries used

Two oversamplers in smote-variants 1.0.1 differ from their original papers:

- **CCR** computes the translation of majority examples but returns the majority class unchanged, so it acts as a sphere-based generator. As a stand-alone method it can also generate many more examples than the class gap.
- **SMOTE-IPF** removed few examples in our runs. It gave the same test AUC as SMOTE in 84% of the folds of Collection C.

Both are reported in the paper.

## What changed since v1.0.1

The v1.0.1 code adapted a single interpolation rule with a heuristic payoff. When it was re-implemented and tested, several parts did not work as described in the earlier manuscript, and its results were close to SMOTE. Version 2.0 replaces it with the portfolio methods above. The design was developed on 27 KEEL datasets and frozen before testing. See Section 3.6 of the paper.

## Note on run times

The column `t_os` in the result files was recorded while nine experiments ran in parallel, so it is not a reliable timing measure. Use the sequential timing file described in the paper for run-time comparisons.

## How to cite

Please cite the paper above and this software release (Zenodo concept DOI 10.5281/zenodo.15200386).

## Licence

MIT licence, see `LICENSE`. The oversamplers come from smote-variants, which is also MIT-licensed.
