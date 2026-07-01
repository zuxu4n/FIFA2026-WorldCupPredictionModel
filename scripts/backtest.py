"""Out-of-sample backtest on the 2026 World Cup matches played so far.

To avoid leakage, the model is trained ONLY on matches before the tournament
started (2026-06-11) and then predicts every WC2026 finals match from its
leak-free pre-match features. Reports accuracy, log-loss, Brier, calibration,
expected-goals error, baselines, and the best calls / biggest misses.

    python scripts/backtest.py [--cutoff 2026-06-11]
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
import xgboost as xgb

from wcpred import config as C
from wcpred import data as D
from wcpred.features import build_world, FEATURE_COLS
from wcpred.predict import score_matrix, wdl_from_matrix

LN3 = np.log(3)


def outcome(hs, as_):  # 0 home win, 1 draw, 2 away win
    return 0 if hs > as_ else (1 if hs == as_ else 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2026-06-11",
                    help="train only on matches strictly before this date")
    args = ap.parse_args()
    cutoff = pd.Timestamp(args.cutoff)

    results = D.load_results()
    long, _ = build_world(results)

    # --- train on PRE-tournament data only -------------------------------
    tr = long[(long.played) & (long.date < cutoff) & (long.date.dt.year >= C.MIN_TRAIN_YEAR)]
    try:
        rounds = json.load(open(C.MODEL_META_PATH))["best_rounds"]
    except Exception:
        rounds = 360
    dtr = xgb.DMatrix(tr[FEATURE_COLS].to_numpy(float), label=tr["goals"].to_numpy(float),
                      weight=tr["weight"].to_numpy(float), feature_names=FEATURE_COLS)
    booster = xgb.train(C.XGB_PARAMS, dtr, num_boost_round=rounds)
    print(f"trained on {len(tr):,} pre-tournament rows (< {cutoff.date()}), {rounds} rounds")

    # --- predict every played WC2026 finals match ------------------------
    wc = results[(results.tournament == C.WC_TOURNAMENT_NAME)
                 & (results.date >= cutoff) & results.played]
    rows = []
    for mid, r in wc.iterrows():
        h = long[(long.match_id == mid) & (long.is_home_persp == 1)]
        a = long[(long.match_id == mid) & (long.is_home_persp == 0)]
        if h.empty or a.empty:
            continue
        X = np.vstack([h[FEATURE_COLS].to_numpy(float), a[FEATURE_COLS].to_numpy(float)])
        lh, la = booster.predict(xgb.DMatrix(X, feature_names=FEATURE_COLS))
        wh, dr, wa = wdl_from_matrix(score_matrix(float(lh), float(la)))
        probs = np.array([wh, dr, wa])
        oc = outcome(r.home_score, r.away_score)
        rows.append({
            "date": r.date.date(), "home": r.home_team, "away": r.away_team,
            "score": f"{int(r.home_score)}-{int(r.away_score)}",
            "lh": lh, "la": la, "p_home": wh, "p_draw": dr, "p_away": wa,
            "pred": int(probs.argmax()), "actual": oc,
            "p_actual": float(probs[oc]),
            "hit": int(probs.argmax() == oc),
            "gh": r.home_score, "ga": r.away_score,
        })
    df = pd.DataFrame(rows)
    n = len(df)

    # --- metrics ----------------------------------------------------------
    eps = 1e-12
    logloss = float(-np.log(df.p_actual.clip(eps)).mean())
    onehot = np.eye(3)[df.actual.to_numpy()]
    P = df[["p_home", "p_draw", "p_away"]].to_numpy()
    brier = float(((P - onehot) ** 2).sum(axis=1).mean())
    acc = df.hit.mean()

    # baselines: higher pre-match Elo wins (no draws), and always-home
    elo_pick = []
    for mid, r in wc.iterrows():
        h = long[(long.match_id == mid) & (long.is_home_persp == 1)]
        if h.empty:
            continue
        elo_pick.append(int(0 if h.iloc[0]["elo_diff"] >= 0 else 2))
    elo_pick = np.array(elo_pick)
    elo_acc = (elo_pick == df.actual.to_numpy()).mean()
    home_acc = (df.actual == 0).mean()

    print(f"\n=== OUT-OF-SAMPLE PERFORMANCE on {n} WC2026 matches ===")
    print(f"  outcome accuracy   : {acc:6.1%}   (model picks the most likely of W/D/L)")
    print(f"    vs higher-Elo pick: {elo_acc:6.1%}")
    print(f"    vs always-home    : {home_acc:6.1%}")
    print(f"    vs random         : {1/3:6.1%}")
    print(f"  log-loss           : {logloss:6.3f}   (random = {LN3:.3f}; lower better)")
    print(f"  Brier score        : {brier:6.3f}   (lower better)")
    print(f"  mean prob on actual: {df.p_actual.mean():6.1%}")
    print(f"  expected-goals MAE : {np.abs(np.r_[df.lh-df.gh, df.la-df.ga]).mean():6.2f} goals/side")

    # calibration on the model's top pick
    print("\n=== Calibration (model's most-likely outcome) ===")
    df["conf"] = P.max(axis=1)
    bins = [(0.33, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.8), (0.8, 1.01)]
    print(f"  {'confidence':>12} {'n':>4} {'predicted':>10} {'actual':>8}")
    for lo, hi in bins:
        m = (df.conf >= lo) & (df.conf < hi)
        if m.sum():
            print(f"  {f'{lo:.0%}-{hi:.0%}':>12} {int(m.sum()):>4} "
                  f"{df.loc[m,'conf'].mean():>10.0%} {df.loc[m,'hit'].mean():>8.0%}")

    lab = {0: "home", 1: "draw", 2: "away"}
    print("\n=== Biggest misses (actual outcome the model rated least likely) ===")
    for _, r in df.nsmallest(5, "p_actual").iterrows():
        print(f"  {r.home} {r.score} {r.away}  -> {lab[r.actual]} won; "
              f"model gave it {r.p_actual:.0%}  (H{r.p_home:.0%}/D{r.p_draw:.0%}/A{r.p_away:.0%})")
    print("\n=== Most confident correct calls ===")
    for _, r in df[df.hit == 1].nlargest(5, "conf").iterrows():
        print(f"  {r.home} {r.score} {r.away}  -> called {lab[r.pred]} at {r.conf:.0%}")

    out = C.OUTPUTS + "/backtest_wc2026.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved per-match detail -> {out}")


if __name__ == "__main__":
    main()
