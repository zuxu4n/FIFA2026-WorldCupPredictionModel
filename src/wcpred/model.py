"""Train / load the XGBoost Poisson expected-goals model."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import xgboost as xgb

from . import config as C
from . import data as D
from .features import build_world, FEATURE_COLS
from .predict import score_matrix, wdl_from_matrix


def _wdl_report(long_valid: pd.DataFrame, lam_pred: np.ndarray) -> dict:
    """Match-level W/D/L log-loss and accuracy on validation rows."""
    df = long_valid.copy()
    df["lam"] = lam_pred
    home = df[df.is_home_persp == 1].set_index("match_id")
    away = df[df.is_home_persp == 0].set_index("match_id")
    common = home.index.intersection(away.index)
    eps = 1e-12
    ll, correct, n = 0.0, 0, 0
    for mid in common:
        lh = float(home.loc[mid, "lam"]); la = float(away.loc[mid, "lam"])
        gh = home.loc[mid, "goals"]; ga = away.loc[mid, "goals"]
        if pd.isna(gh) or pd.isna(ga):
            continue
        wh, dr, wa = wdl_from_matrix(score_matrix(lh, la))
        probs = np.clip([wh, dr, wa], eps, 1)
        if gh > ga:
            outcome, pred_idx = 0, int(np.argmax([wh, dr, wa]))
        elif gh == ga:
            outcome, pred_idx = 1, int(np.argmax([wh, dr, wa]))
        else:
            outcome, pred_idx = 2, int(np.argmax([wh, dr, wa]))
        ll -= np.log(probs[outcome])
        correct += int(pred_idx == outcome)
        n += 1
    return {"n_matches": n,
            "wdl_logloss": round(ll / n, 4) if n else None,
            "wdl_accuracy": round(correct / n, 4) if n else None}


def train(save: bool = True, verbose: bool = True) -> tuple[xgb.Booster, dict]:
    results = D.load_results()
    long, _ = build_world(results)

    train_rows = long[(long.played) & (long.date.dt.year >= C.MIN_TRAIN_YEAR)].copy()
    valid_cut = pd.Timestamp(C.VALID_SINCE)
    tr = train_rows[train_rows.date < valid_cut]
    va = train_rows[train_rows.date >= valid_cut]
    if verbose:
        print(f"train rows: {len(tr):,}  valid rows: {len(va):,}  "
              f"(valid since {C.VALID_SINCE})")

    dtr = xgb.DMatrix(tr[FEATURE_COLS].to_numpy(float), label=tr["goals"].to_numpy(float),
                      weight=tr["weight"].to_numpy(float), feature_names=FEATURE_COLS)
    dva = xgb.DMatrix(va[FEATURE_COLS].to_numpy(float), label=va["goals"].to_numpy(float),
                      weight=va["weight"].to_numpy(float), feature_names=FEATURE_COLS)

    booster = xgb.train(
        C.XGB_PARAMS, dtr, num_boost_round=C.XGB_NUM_ROUNDS,
        evals=[(dtr, "train"), (dva, "valid")],
        early_stopping_rounds=C.XGB_EARLY_STOPPING,
        verbose_eval=100 if verbose else False,
    )
    best_rounds = booster.best_iteration + 1
    lam_va = booster.predict(dva, iteration_range=(0, best_rounds))
    report = _wdl_report(va, lam_va)
    if verbose:
        print(f"best_iteration={best_rounds}  validation W/D/L: {report}")

    # Refit on ALL data (train+valid) for the deployed model.
    dall = xgb.DMatrix(train_rows[FEATURE_COLS].to_numpy(float),
                       label=train_rows["goals"].to_numpy(float),
                       weight=train_rows["weight"].to_numpy(float),
                       feature_names=FEATURE_COLS)
    final = xgb.train({**C.XGB_PARAMS}, dall, num_boost_round=best_rounds,
                      verbose_eval=False)

    meta = {
        "feature_cols": FEATURE_COLS,
        "best_rounds": int(best_rounds),
        "params": C.XGB_PARAMS,
        "validation": report,
        "n_train_all": int(len(train_rows)),
        "trained_through": str(results.loc[results.played, "date"].max().date()),
    }
    if save:
        final.save_model(C.MODEL_PATH)
        with open(C.MODEL_META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if verbose:
            print(f"saved model -> {C.MODEL_PATH}")
    return final, meta


def load_model() -> tuple[xgb.Booster, dict]:
    booster = xgb.Booster()
    booster.load_model(C.MODEL_PATH)
    with open(C.MODEL_META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    return booster, meta


def feature_importance(booster: xgb.Booster, kind: str = "gain") -> list[tuple[str, float]]:
    score = booster.get_score(importance_type=kind)
    return sorted(score.items(), key=lambda kv: kv[1], reverse=True)


def explain(booster: xgb.Booster, row: dict) -> list[tuple[str, float, float]]:
    """Per-feature SHAP contributions for one perspective's expected goals.

    Returns [(feature, contribution_log_lambda, pct_effect_on_goals)] sorted by
    |contribution|; pct_effect = exp(contrib)-1 (the multiplicative effect this
    feature has on the team's expected goals vs. the average).
    """
    X = np.array([[row[c] for c in FEATURE_COLS]], dtype=float)
    dm = xgb.DMatrix(X, feature_names=FEATURE_COLS)
    contribs = booster.predict(dm, pred_contribs=True)[0]  # last entry = bias
    out = [(FEATURE_COLS[i], float(c), float(np.exp(c) - 1.0))
           for i, c in enumerate(contribs[:-1]) if abs(c) > 1e-9]
    return sorted(out, key=lambda t: abs(t[1]), reverse=True)
