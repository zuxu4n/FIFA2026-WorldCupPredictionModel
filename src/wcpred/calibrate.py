"""Calibration layers on top of the raw Poisson model.

Three fitted corrections, all estimated on *out-of-sample* historical
predictions (model trained on pre-cutoff data, predicting post-cutoff):

1. **Lambda (totals) calibration** — the raw model systematically over-predicts
   goals in certain contexts (notably tournament knockouts, where game-state
   suppresses scoring). We fit a multiplicative factor per match segment and
   scale both lambdas before any probability is derived.
2. **Dixon-Coles rho** — instead of the hardcoded config value, rho is fit by
   profile likelihood on the same out-of-sample scores.
3. **W/D/L calibration** — multinomial logistic recalibration of the
   [home, draw, away] probabilities, conditioned on match *closeness* so the
   draw correction doesn't inflate draws in blowouts (v2; v1 was global).

    fit everything:  python scripts/calibrate.py            (W/D/L, needs 1+2 first)
                     python scripts/backtest_totals.py      (fits 1+2, reports bias)
    apply: automatic in predict.predict_lambdas / predict.summarize.
"""
from __future__ import annotations
import json
import os
import numpy as np

from . import config as C

CALIB_PATH = os.path.join(C.MODELS, "calibrator.json")
TOTALS_PATH = os.path.join(C.MODELS, "totals_calibrator.json")
_EPS = 1e-6

_TOTALS_CACHE: dict | None = None
_TOTALS_LOADED = False


# ==========================================================================
# shared out-of-sample dataset
# ==========================================================================
def build_oos_frame(cutoff: str = "2023-01-01"):
    """Train on pre-cutoff data only; predict every post-cutoff played match.

    Returns a DataFrame (one row per match): lam_h, lam_a, gh, ga, importance,
    is_knockout, elo_diff, date — the raw material for all calibrations.
    """
    import pandas as pd
    import xgboost as xgb
    from . import data as D
    from .features import build_world, FEATURE_COLS

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
    lam = booster.predict(xgb.DMatrix(va[FEATURE_COLS].to_numpy(float),
                                      feature_names=FEATURE_COLS))
    va = va.assign(lam=lam)
    home = va[va.is_home_persp == 1].set_index("match_id")
    away = va[va.is_home_persp == 0].set_index("match_id")
    common = home.index.intersection(away.index)
    h, a = home.loc[common], away.loc[common]
    df = pd.DataFrame({
        "lam_h": h["lam"].to_numpy(float), "lam_a": a["lam"].to_numpy(float),
        "gh": h["goals"].to_numpy(float), "ga": a["goals"].to_numpy(float),
        "importance": h["importance"].to_numpy(float),
        "is_knockout": h["is_knockout"].to_numpy(float),
        "elo_diff": h["elo_diff"].to_numpy(float),
        "date": h["date"].to_numpy(),
    }).dropna(subset=["gh", "ga"])
    return df.reset_index(drop=True)


# ==========================================================================
# 1) lambda (totals) calibration + 2) rho
# ==========================================================================
SEGMENTS = ("knockout", "tournament_group", "qualifier", "other")


def segment_of(importance: float, is_knockout: float) -> str:
    if is_knockout >= 0.5:
        return "knockout"
    if importance >= 3.0:
        return "tournament_group"
    if importance >= 2.0:
        return "qualifier"
    return "other"


def load_totals() -> dict | None:
    global _TOTALS_CACHE, _TOTALS_LOADED
    if not _TOTALS_LOADED:
        _TOTALS_LOADED = True
        if os.path.exists(TOTALS_PATH):
            with open(TOTALS_PATH) as f:
                _TOTALS_CACHE = json.load(f)
    return _TOTALS_CACHE


def invalidate_totals_cache():
    global _TOTALS_LOADED, _TOTALS_CACHE
    _TOTALS_LOADED, _TOTALS_CACHE = False, None


def scale_lambdas(lam_h: float, lam_a: float, importance: float,
                  is_knockout: float) -> tuple[float, float]:
    """Apply the fitted per-segment shrinkage (identity if not fitted)."""
    tot = load_totals()
    if not tot:
        return lam_h, lam_a
    f = tot["factors"].get(segment_of(importance, is_knockout), 1.0)
    return lam_h * f, lam_a * f


def fitted_rho() -> float:
    tot = load_totals()
    if tot and "rho" in tot:
        return float(tot["rho"])
    return C.DIXON_COLES_RHO


def _fit_rho(df, factors) -> float:
    """Profile-likelihood grid search for the Dixon-Coles rho."""
    from .predict import score_matrix
    best, best_ll = C.DIXON_COLES_RHO, -np.inf
    sub = df.sample(min(len(df), 4000), random_state=0)
    for rho in np.arange(-0.20, 0.101, 0.01):
        ll = 0.0
        for r in sub.itertuples(index=False):
            f = factors.get(segment_of(r.importance, r.is_knockout), 1.0)
            M = score_matrix(r.lam_h * f, r.lam_a * f, rho=rho)
            i, j = min(int(r.gh), M.shape[0] - 1), min(int(r.ga), M.shape[1] - 1)
            ll += np.log(max(M[i, j], _EPS))
        if ll > best_ll:
            best_ll, best = ll, float(rho)
    return round(best, 3)


def fit_totals_and_rho(cutoff: str = "2023-01-01", save: bool = True, df=None):
    """Fit per-segment lambda factors + rho; print the bias report."""
    if df is None:
        df = build_oos_frame(cutoff)
    seg = df.apply(lambda r: segment_of(r.importance, r.is_knockout), axis=1)
    factors, report = {}, []
    for s in SEGMENTS:
        m = seg == s
        if m.sum() < 30:
            factors[s] = 1.0
            continue
        pred = float(df.loc[m, "lam_h"].sum() + df.loc[m, "lam_a"].sum())
        act = float(df.loc[m, "gh"].sum() + df.loc[m, "ga"].sum())
        factors[s] = round(act / pred, 4) if pred > 0 else 1.0
        report.append((s, int(m.sum()),
                       (df.loc[m, "lam_h"] + df.loc[m, "lam_a"]).mean(),
                       (df.loc[m, "gh"] + df.loc[m, "ga"]).mean(), factors[s]))

    print(f"totals calibration ({len(df)} OOS matches >= {cutoff}):")
    print(f"  {'segment':18}{'n':>6}{'pred total':>12}{'actual':>9}{'factor':>8}")
    for s, n, p, a, f in report:
        print(f"  {s:18}{n:>6}{p:>12.2f}{a:>9.2f}{f:>8.3f}")

    rho = _fit_rho(df, factors)
    print(f"fitted Dixon-Coles rho: {rho}  (config default {C.DIXON_COLES_RHO})")

    if save:
        json.dump({"factors": factors, "rho": rho, "cutoff": cutoff},
                  open(TOTALS_PATH, "w"), indent=2)
        invalidate_totals_cache()
        print(f"saved -> {TOTALS_PATH}")
    return factors, rho, df


# ==========================================================================
# 3) W/D/L calibration (v2: closeness-conditional)
# ==========================================================================
def _cal_features(P) -> np.ndarray:
    """log-probs + |log(p_home/p_away)| closeness term."""
    P = np.clip(np.asarray(P, dtype=float), _EPS, None)
    L = np.log(P)
    closeness = np.abs(L[..., 0] - L[..., 2])[..., None]
    return np.concatenate([L, closeness], axis=-1)


def load():
    if not os.path.exists(CALIB_PATH):
        return None
    with open(CALIB_PATH) as f:
        d = json.load(f)
    return {"coef": np.array(d["coef"]), "intercept": np.array(d["intercept"]),
            "version": d.get("version", 1)}


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def apply_probs(cal, P):
    """P: (...,3) raw [home,draw,away] -> calibrated (...,3)."""
    P = np.clip(np.asarray(P, dtype=float), _EPS, None)
    X = _cal_features(P) if cal["coef"].shape[1] == 4 else np.log(P)
    z = X @ cal["coef"].T + cal["intercept"]
    return _softmax(z)


def apply_one(cal, p_home, p_draw, p_away):
    out = apply_probs(cal, np.array([p_home, p_draw, p_away]))
    return float(out[0]), float(out[1]), float(out[2])


def build_calibration_data(cutoff="2023-01-01", df=None):
    """(raw-but-lambda-scaled W/D/L probs, outcome) pairs from the OOS frame."""
    from .predict import score_matrix, wdl_from_matrix
    if df is None:
        df = build_oos_frame(cutoff)
    rho = fitted_rho()
    P, y = [], []
    for r in df.itertuples(index=False):
        lh, la = scale_lambdas(r.lam_h, r.lam_a, r.importance, r.is_knockout)
        P.append(wdl_from_matrix(score_matrix(lh, la, rho=rho)))
        y.append(0 if r.gh > r.ga else (1 if r.gh == r.ga else 2))
    return np.array(P), np.array(y)


def _metrics(P, y):
    P = np.clip(P, _EPS, 1)
    ll = float(-np.log(P[np.arange(len(y)), y]).mean())
    brier = float(((P - np.eye(3)[y]) ** 2).sum(1).mean())
    return ll, brier


def fit_and_report(cutoff="2023-01-01", save=True, df=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    P, y = build_calibration_data(cutoff, df=df)
    X = _cal_features(P)
    clf = LogisticRegression(max_iter=2000, C=1.0)

    P_cal_cv = cross_val_predict(clf, X, y, cv=5, method="predict_proba")
    ll0, b0 = _metrics(P, y)
    ll1, b1 = _metrics(P_cal_cv, y)
    draw_actual = float((y == 1).mean())
    print(f"W/D/L calibration set: {len(y)} matches ( >= {cutoff} , out-of-sample )")
    print(f"  {'':14}{'log-loss':>10}{'Brier':>9}{'mean draw prob':>16}")
    print(f"  {'raw model':14}{ll0:>10.4f}{b0:>9.4f}{P[:,1].mean():>15.1%}")
    print(f"  {'calibrated':14}{ll1:>10.4f}{b1:>9.4f}{P_cal_cv[:,1].mean():>15.1%}")
    print(f"  {'actual':14}{'':>10}{'':>9}{draw_actual:>15.1%}")

    if save:
        # Only deploy the recalibration if it actually beats the raw pipeline
        # out-of-fold; otherwise raw probs are better-calibrated — remove any
        # stale calibrator so predictions fall back to raw.
        if ll1 < ll0:
            clf.fit(X, y)
            json.dump({"coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist(),
                       "classes": clf.classes_.tolist(), "cutoff": cutoff, "version": 2},
                      open(CALIB_PATH, "w"), indent=2)
            print(f"saved calibrator -> {CALIB_PATH}")
        else:
            if os.path.exists(CALIB_PATH):
                os.remove(CALIB_PATH)
            print("calibrated CV log-loss does NOT beat raw -> raw probabilities kept "
                  "(no W/D/L calibrator deployed)")
    return {"logloss_raw": ll0, "logloss_cal": ll1, "brier_raw": b0, "brier_cal": b1}
