"""Probability calibration for the W/D/L outputs.

The Poisson score-matrix gives raw win/draw/loss probabilities that are
systematically off (notably the draw is under-weighted). We fit a multinomial
logistic recalibration ("matrix scaling") on the *log* of the three raw
probabilities, using out-of-sample historical predictions (model trained on
pre-2023 data, predicting 2023+), then apply it to every W/D/L output.

    fit:    python scripts/calibrate.py
    apply:  handled automatically in predict.summarize() if a calibrator exists
"""
from __future__ import annotations
import json
import os
import numpy as np

from . import config as C

CALIB_PATH = os.path.join(C.MODELS, "calibrator.json")
_EPS = 1e-6


# --------------------------------------------------------------------------
# apply (cheap; used at prediction time)
# --------------------------------------------------------------------------
def load():
    if not os.path.exists(CALIB_PATH):
        return None
    with open(CALIB_PATH) as f:
        d = json.load(f)
    return {"coef": np.array(d["coef"]), "intercept": np.array(d["intercept"])}


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def apply_probs(cal, P):
    """P: (...,3) raw [home,draw,away] -> calibrated (...,3)."""
    P = np.clip(np.asarray(P, dtype=float), _EPS, None)
    X = np.log(P)
    z = X @ cal["coef"].T + cal["intercept"]
    return _softmax(z)


def apply_one(cal, p_home, p_draw, p_away):
    out = apply_probs(cal, np.array([p_home, p_draw, p_away]))
    return float(out[0]), float(out[1]), float(out[2])


# --------------------------------------------------------------------------
# fit (offline)
# --------------------------------------------------------------------------
def build_calibration_data(cutoff="2023-01-01"):
    """Out-of-sample (raw W/D/L probs, outcome) pairs: train < cutoff, predict >=."""
    import pandas as pd
    import xgboost as xgb
    from . import data as D
    from .features import build_world, FEATURE_COLS
    from .predict import score_matrix, wdl_from_matrix

    results = D.load_results()
    long, _ = build_world(results)
    cut = pd.Timestamp(cutoff)
    tr = long[(long.played) & (long.date < cut) & (long.date.dt.year >= C.MIN_TRAIN_YEAR)]
    va = long[(long.played) & (long.date >= cut)]

    dtr = xgb.DMatrix(tr[FEATURE_COLS].to_numpy(float), label=tr["goals"].to_numpy(float),
                      weight=tr["weight"].to_numpy(float), feature_names=FEATURE_COLS)
    try:
        rounds = json.load(open(C.MODEL_META_PATH))["best_rounds"]
    except Exception:
        rounds = 340
    booster = xgb.train(C.XGB_PARAMS, dtr, num_boost_round=rounds)

    lam = booster.predict(xgb.DMatrix(va[FEATURE_COLS].to_numpy(float), feature_names=FEATURE_COLS))
    va = va.assign(lam=lam)
    home = va[va.is_home_persp == 1].set_index("match_id")
    away = va[va.is_home_persp == 0].set_index("match_id")
    P, y = [], []
    for mid in home.index.intersection(away.index):
        lh, la = float(home.loc[mid, "lam"]), float(away.loc[mid, "lam"])
        gh, ga = home.loc[mid, "goals"], away.loc[mid, "goals"]
        if np.isnan(gh) or np.isnan(ga):
            continue
        P.append(wdl_from_matrix(score_matrix(lh, la)))
        y.append(0 if gh > ga else (1 if gh == ga else 2))
    return np.array(P), np.array(y)


def _metrics(P, y):
    P = np.clip(P, _EPS, 1)
    ll = float(-np.log(P[np.arange(len(y)), y]).mean())
    brier = float(((P - np.eye(3)[y]) ** 2).sum(1).mean())
    return ll, brier


def fit_and_report(cutoff="2023-01-01", save=True):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    P, y = build_calibration_data(cutoff)
    X = np.log(np.clip(P, _EPS, None))
    clf = LogisticRegression(max_iter=2000, C=1.0)

    # cross-validated calibrated probs (honest, out-of-fold)
    P_cal_cv = cross_val_predict(clf, X, y, cv=5, method="predict_proba")
    ll0, b0 = _metrics(P, y)
    ll1, b1 = _metrics(P_cal_cv, y)

    draw_actual = float((y == 1).mean())
    print(f"calibration set: {len(y)} matches ( >= {cutoff} , out-of-sample )")
    print(f"  {'':14}{'log-loss':>10}{'Brier':>9}{'mean draw prob':>16}")
    print(f"  {'raw model':14}{ll0:>10.4f}{b0:>9.4f}{P[:,1].mean():>15.1%}")
    print(f"  {'calibrated':14}{ll1:>10.4f}{b1:>9.4f}{P_cal_cv[:,1].mean():>15.1%}")
    print(f"  {'actual':14}{'':>10}{'':>9}{draw_actual:>15.1%}")

    clf.fit(X, y)
    if save:
        json.dump({"coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist(),
                   "classes": clf.classes_.tolist(), "cutoff": cutoff},
                  open(CALIB_PATH, "w"), indent=2)
        print(f"saved calibrator -> {CALIB_PATH}")
    return {"logloss_raw": ll0, "logloss_cal": ll1, "brier_raw": b0, "brier_cal": b1}
