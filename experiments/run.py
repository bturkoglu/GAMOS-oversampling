"""Benchmark harness: 32 KEEL datasets x RepeatedStratifiedKFold(5 folds x R repeats)
x oversamplers x {kNN, linear SVM, MLP}. One CSV row per (dataset, repeat, fold, method, classifier).
Usage: python -I run.py <out_dir> <repeats> <n_jobs> [method,method,...]"""
import os, sys, time, json, warnings, logging, traceback
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'portos'))
from joblib import Parallel, delayed

DATASETS = ["abalone_17_vs_7_8_9_10", "abalone_20_vs_8_9_10", "abalone_3_vs_11", "abalone9_18", "cleveland_0_vs_4",
            "dermatology_6", "ecoli_0_1_vs_2_3_5", "ecoli_0_1_vs_5", "ecoli_0_2_6_7_vs_3_5", "ecoli_0_3_4_vs_5",
            "ecoli_0_6_7_vs_5", "ecoli3", "flaref", "glass_0_6_vs_5", "led7digit_0_2_4_5_6_7_8_9_vs_1",
            "lymphography_normal_fibrosis", "pima", "segment0", "shuttle_c2_vs_c4", "vehicle1", "vehicle2", "vehicle3",
            "winequality_white_3_9_vs_5", "winequality_white_9_vs_4", "wisconsin", "yeast_0_3_5_9_vs_7_8",
            "yeast_0_5_6_7_9_vs_4", "yeast_1_2_8_9_vs_7", "yeast_1_vs_7", "yeast1", "yeast3", "yeast4"]

DEV = ["abalone_19_vs_10_11_12_13", "abalone_21_vs_8", "car_good", "cm1", "ecoli2", "ecoli4",
       "ecoli_0_1_4_7_vs_2_3_5_6", "glass2", "glass4", "glass6", "habarman", "kc1", "kr_vs_k_zero_vs_eight",
       "new_thyroid1", "page_blocks_1_3_vs_4", "pc1", "poker_9_vs_7", "shuttle_6_vs_2_3", "spectf", "vehicle0",
       "vowel0", "winequality_red_4", "winequality_red_8_vs_6", "yeast5", "yeast_1_4_5_8_vs_7", "yeast_2_vs_8", "german"]
G2 = {"G2": {}, "G2-beta2": {"beta": 2.0}, "G2-sigma0": {"sigma": 0.0}, "G2-noFilter": {"use_filter": False},
      "G2-T20": {"n_rounds": 20}, "G2-noGame": {"use_game": False}, "G2-noDetector": {"use_detector": False},
      "G2-noUtility": {"use_utility": False}, "G2-isoNoise": {"feature_weighting": False}}
for _r in ("uniform", "border", "safe", "proximity", "sparse"):
    for _a in (0.5, 1.0):
        G2[f"ARM-{_r}-{_a}"] = {"rules": (_r,), "amax_grid": (_a,), "use_game": False}
G3 = {"G3": {}, "G3-noGame": {"use_game": False}, "G3-noFilter": {"use_filter": False}, "G3-T20": {"n_rounds": 20},
      "G3-noDetector": {"use_detector": False}, "G3-noUtility": {"use_utility": False}, "G3-sigma0": {"sigma": 0.0},
      "G3-PW": {"thetas": (1.0,), "partners": ("level",), "use_game": False, "use_filter": False, "sigma": 0.0}}
G4 = {"G4": {}, "G4-mean": {"mode": "mean"}, "G4-uniform": {"mode": "uniform"}, "G4-det": {"use_detector": True},
      "G4-noFilter": {"use_filter": False}, "G4-sigma0": {"sigma": 0.0}, "G4-J10": {"n_val": 10}}
TEST2 = ["ecoli1", "ecoli_0_1_3_7_vs_2_6", "ecoli_0_1_4_6_vs_5", "ecoli_0_1_4_7_vs_5_6", "ecoli_0_2_3_4_vs_5",
         "ecoli_0_3_4_6_vs_5", "ecoli_0_3_4_7_vs_5_6", "ecoli_0_4_6_vs_5", "ecoli_0_6_7_vs_3_5", "glass0", "glass1",
         "glass5", "glass_0_1_2_3_vs_4_5_6", "glass_0_1_4_6_vs_2", "glass_0_1_5_vs_2", "glass_0_1_6_vs_2",
         "glass_0_1_6_vs_5", "glass_0_4_vs_5", "iris0", "page_blocks0", "shuttle_c0_vs_c4", "winequality_red_3_vs_5",
         "winequality_red_8_vs_6_7", "winequality_white_3_vs_7", "yeast6", "yeast_0_2_5_6_vs_3_7_8_9",
         "yeast_0_2_5_7_9_vs_3_6_8", "yeast_2_vs_4", "abalone_19_vs_10_11_12_13".replace("19_vs_10_11_12_13","19_vs_10_11_12_13") if False else "car_vgood",
         "kr_vs_k_zero_vs_fifteen", "kddcup_land_vs_portsweep", "shuttle_2_vs_5", "zoo_3"]
_FIN = {"use_filter": False}
G5 = {"G5": {}, "G5-uniform": {"mode": "uniform"}, "G5-mean": {"mode": "mean"}, "G5-knnproxy": {"proxies": ("lr", "knn")}}
FINAL = {"GAMOS-final": dict(_FIN), "GAMOS-final-mean": dict(_FIN, mode="mean"), "GAMOS-final-uniform": dict(_FIN, mode="uniform"),
         "GAMOS-final-det": dict(_FIN, use_detector=True), "GAMOS-final-filter": {"use_filter": True},
         "GAMOS-final-sigma0": dict(_FIN, sigma=0.0), "GAMOS-final-iso": dict(_FIN, feature_weighting=False)}
SV = ["SMOTE", "Borderline_SMOTE1", "Borderline_SMOTE2", "ADASYN", "Safe_Level_SMOTE", "SMOTE_D", "SMOTE_RSB",
      "A_SUWO", "ProWSyn", "polynom_fit_SMOTE_star", "SMOTE_IPF", "MWMOTE", "G_SMOTE", "LVQ_SMOTE", "CCR"]
GAMOS_VARIANTS = {"GAMOS": {}, "GAMOS-noGame": {"use_game": False}, "GAMOS-noDetector": {"use_detector": False},
                  "GAMOS-noUtility": {"use_utility": False}, "GAMOS-noFilter": {"use_filter": False},
                  "GAMOS-isoNoise": {"feature_weighting": False}}
ALL = ["Original"] + SV + list(GAMOS_VARIANTS) + ["GAMOS-v1.0.1"]


IMBL = ["optical_digits", "satimage", "pen_digits", "abalone", "sick_euthyroid", "spectrometer", "isolet", "us_crime",
        "yeast_ml8", "scene", "libras_move", "thyroid_sick", "coil_2000", "arrhythmia", "solar_flare_m0", "oil",
        "wine_quality", "letter_img", "ozone_level", "mammography", "abalone_19"]


def load(name):
    if name.startswith("imbl:"):
        from imblearn.datasets import fetch_datasets
        d = fetch_datasets(data_home=os.environ.get("IMBL_DATA", os.path.expanduser("~/imbl_data")), filter_data=(name[5:],))[name[5:]]
        y = (np.asarray(d.target) == 1).astype(int)
        return np.asarray(d.data, float), y
    import imbalanced_databases as idb
    d = getattr(idb, "load_" + name)()
    return np.asarray(d["data"], float), np.asarray(d["target"]).astype(int)


def oversample(method, X, y, seed):
    if method == "Original":
        return X, y
    if method in GAMOS_VARIANTS:
        from gamos import GAMOS
        return GAMOS(random_state=seed, **GAMOS_VARIANTS[method]).fit_resample(X, y)
    if method.startswith("T-"):
        from tuned import tuned_resample
        return tuned_resample(method[2:], X, y, seed)
    if method == "ROS":
        rng = np.random.RandomState(seed); lab, cnt = np.unique(y, return_counts=True); pos = lab[np.argmin(cnt)]
        idx = rng.choice(np.where(y == pos)[0], cnt.max() - cnt.min(), replace=True)
        return np.vstack([X, X[idx]]), np.r_[y, y[idx]]
    if method.startswith("T2-"):
        from tuned2 import tuned_resample as tr2
        return tr2(method[3:], X, y, seed)
    if method.startswith("G6-"):
        from gamos6 import GAMOS as G6, SV_PORTFOLIO
        base, _, mode = method[3:].partition("-")
        cfg = {"v4p": dict(generators="v4", proportions=(0.25, 0.5, 1.0)), "sv": dict(generators=SV_PORTFOLIO, proportions=(1.0,)),
               "svp": dict(generators=SV_PORTFOLIO, proportions=(0.25, 0.5, 1.0))}[base]
        return G6(random_state=seed, mode=mode or "minimax", **cfg).fit_resample(X, y)
    if method == "GAMOS-final-treeproxy":
        from gamos4b import GAMOS as G
        return G(random_state=seed, use_filter=False).fit_resample(X, y)
    if method in G5:
        from gamos5 import GAMOS as G
        return G(random_state=seed, **G5[method]).fit_resample(X, y)
    if method in FINAL:
        from gamos4 import GAMOS as G
        return G(random_state=seed, **FINAL[method]).fit_resample(X, y)
    if method in G4:
        from gamos4 import GAMOS as G
        return G(random_state=seed, **G4[method]).fit_resample(X, y)
    if method in G3:
        from gamos3 import GAMOS as G
        return G(random_state=seed, **G3[method]).fit_resample(X, y)
    if method in G2:
        from gamos2 import GAMOS as G
        return G(random_state=seed, **G2[method]).fit_resample(X, y)
    if method == "GAMOS-v1.0.1":
        sys.path.insert(0, os.path.join(HERE, "..", "code", "GAMOS-oversampling-1.0.1"))
        from _gameSL import GAME_GEN
        Xr, yr = GAME_GEN(random_state=seed).fit_resample(X, y)
        return Xr, yr.astype(int)
    import smote_variants as sv
    logging.getLogger("smote_variants").setLevel(logging.ERROR)
    return getattr(sv, method)(random_state=seed).sample(X, y)


def classifiers(seed):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import LinearSVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    allc = {"kNN": KNeighborsClassifier(n_neighbors=5),
            "LinearSVM": LinearSVC(C=1.0, max_iter=20000, random_state=seed),
            "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=seed),
            "RF": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)}
    want = os.environ.get("CLFS", "kNN,LinearSVM,MLP").split(",")
    return {k: v for k, v in allc.items() if k in want}


def task(ds, rep, fold, tr, te, methods, out):
    warnings.filterwarnings("ignore")
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    fn = os.path.join(out, "parts", f"{ds.replace(':', '_')}__{rep}__{fold}.csv")
    done = set()
    if os.path.exists(fn):
        done = set(pd.read_csv(fn)["method"].unique())
    todo = [m for m in methods if m not in done]
    if not todo:
        return
    X, y = load(ds)
    sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te]); ytr, yte = y[tr], y[te]
    pos = np.unique(ytr, return_counts=True); pos = pos[0][np.argmin(pos[1])]
    rows = []
    seed = 1000 * rep + fold
    for m in todo:
        t0 = time.process_time(); err = ""
        try:
            Xr, yr = oversample(m, Xtr.copy(), ytr.copy(), seed)
            Xr = np.asarray(Xr, float); yr = np.asarray(yr).astype(int)
            ok = np.isfinite(Xr).all(axis=1); Xr, yr = Xr[ok], yr[ok]
        except Exception as e:
            Xr, yr, err = Xtr, ytr, "FALLBACK:" + repr(e)[:200]
        t_os = time.process_time() - t0
        n_min_after = int(np.sum(yr == pos))
        for cn, clf in classifiers(seed).items():
            try:
                clf.fit(Xr, yr)
                s = clf.decision_function(Xte) if hasattr(clf, "decision_function") else clf.predict_proba(Xte)[:, list(clf.classes_).index(pos)]
                if cn == "LinearSVM" and pos != clf.classes_[1]:
                    s = -s
                auc = roc_auc_score(yte == pos, s)
                p = clf.predict(Xte)
                se = np.mean(p[yte == pos] == pos); sp = np.mean(p[yte != pos] != pos)
                g = float(np.sqrt(se * sp))
            except Exception as e:
                auc, g, se, sp, err = np.nan, np.nan, np.nan, np.nan, err + " CLF:" + repr(e)[:100]
            rows.append(dict(dataset=ds, rep=rep, fold=fold, method=m, clf=cn, auc=auc, gmean=g, sens=se, spec=sp,
                             t_os=t_os, n_min_after=n_min_after, n_maj=int(np.sum(ytr != pos)), err=err))
    df = pd.DataFrame(rows)
    df.to_csv(fn, mode="a", header=not os.path.exists(fn), index=False)


def main():
    out, R, nj = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    methods = sys.argv[4].split(",") if len(sys.argv) > 4 else ALL
    os.makedirs(os.path.join(out, "parts"), exist_ok=True)
    from sklearn.model_selection import RepeatedStratifiedKFold
    jobs = []
    for ds in {"dev": DEV, "test2": TEST2, "imbl": ["imbl:" + x for x in IMBL]}.get(os.environ.get("DSET"), DATASETS):
        X, y = load(ds)
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=R, random_state=42)
        for i, (tr, te) in enumerate(rskf.split(X, y)):
            jobs.append((ds, i // 5, i % 5, tr, te))
    # biggest datasets first for load balance
    jobs.sort(key=lambda j: -len(j[3]))
    Parallel(n_jobs=nj, verbose=5)(delayed(task)(*j, methods, out) for j in jobs)


if __name__ == "__main__":
    main()
