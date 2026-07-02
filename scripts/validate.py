"""Rolling-origin validation: train strictly before each fold's window,
evaluate W/D/L log-loss/accuracy and total-goals MAE inside it.

    python scripts/validate.py

Writes models/validation_folds.json.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import xgboost as xgb

from wcpred import config as C
from wcpred import data as D
from wcpred.features import build_world, FEATURE_COLS
from wcpred.predict import score_matrix, wdl_from_matrix

FOLDS = [
    ("2020-01-01", "2022-01-01"),
    ("2022-01-01", "2024-01-01"),
    ("2024-01-01", "2026-06-11"),
    ("2026-06-11", "2027-01-01"),   # WC2026 so far
]


def main():
    results = D.load_results()
    long, _ = build_world(results)
    played = long[long.played & (long.date.dt.year >= C.MIN_TRAIN_YEAR)]
    rows = []
    for start, end in FOLDS:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        tr = played[played.date < s]
        te = played[(played.date >= s) & (played.date < e)]
        if te.empty:
            continue
        dtr = xgb.DMatrix(tr[FEATURE_COLS].to_numpy(float),
                          label=tr["goals"].to_numpy(float),
                          weight=tr["weight"].to_numpy(float),
                          feature_names=FEATURE_COLS)
        booster = xgb.train(C.XGB_PARAMS, dtr, num_boost_round=340)
        lam = booster.predict(xgb.DMatrix(te[FEATURE_COLS].to_numpy(float),
                                          feature_names=FEATURE_COLS))
        te = te.assign(lam=lam)
        h = te[te.is_home_persp == 1].set_index("match_id")
        a = te[te.is_home_persp == 0].set_index("match_id")
        common = h.index.intersection(a.index)
        ll, acc, mae, n = 0.0, 0, 0.0, 0
        for mid in common:
            lh, la = float(h.loc[mid, "lam"]), float(a.loc[mid, "lam"])
            gh, ga = h.loc[mid, "goals"], a.loc[mid, "goals"]
            probs = np.clip(wdl_from_matrix(score_matrix(lh, la)), 1e-12, 1)
            oc = 0 if gh > ga else (1 if gh == ga else 2)
            ll -= np.log(probs[oc]); acc += int(int(np.argmax(probs)) == oc)
            mae += abs((lh + la) - (gh + ga)); n += 1
        rows.append({"fold": f"{start[:7]}..{end[:7]}", "n": n,
                     "wdl_logloss": round(ll / n, 4), "wdl_acc": round(acc / n, 4),
                     "totals_mae": round(mae / n, 3)})
        print(f"  {rows[-1]['fold']:>18}  n={n:>5}  logloss {rows[-1]['wdl_logloss']:.4f}"
              f"  acc {rows[-1]['wdl_acc']:.1%}  totals MAE {rows[-1]['totals_mae']:.3f}")

    out = C.MODELS + "/validation_folds.json"
    json.dump(rows, open(out, "w"), indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
