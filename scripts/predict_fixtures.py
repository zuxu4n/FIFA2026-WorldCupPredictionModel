"""Predict every upcoming (unplayed) fixture present in the data.

    python scripts/predict_fixtures.py

Prints a clean table and writes outputs/fixture_predictions.csv.
"""
from __future__ import annotations
import os
import pandas as pd
from _common import load_everything
from wcpred import config as C
from wcpred.predict import predict_fixture

MONTHS = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
          7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}


def main():
    results, world, booster, meta = load_everything()
    upcoming = results[~results.played].sort_values("date")
    if upcoming.empty:
        print("No unplayed fixtures in the data.")
        return

    rows, disp = [], []
    for r in upcoming.itertuples(index=False):
        p = predict_fixture(booster, world, r.home_team, r.away_team,
                            city=r.city, country=r.country,
                            neutral=bool(r.neutral), importance=4.0)
        d = pd.Timestamp(r.date)
        ml = p.top_scorelines[0][0]
        rows.append({"date": d.date(), **p.as_row(), "city": r.city, "country": r.country})
        disp.append((f"{MONTHS[d.month]} {d.day:02d}", r.home_team[:15], r.away_team[:15],
                     f"{p.lam_home:.2f}-{p.lam_away:.2f}",
                     p.p_home_win, p.p_draw, p.p_away_win,
                     f"{ml[0]}-{ml[1]}", p.p_home_advance))

    W = 80
    print("=" * W)
    print(f" FIFA World Cup 2026 - Match Predictions        model through {meta['trained_through']}")
    print("=" * W)
    print(f" {'Date':7}{'Home':>16}      {'Away':<16}{'xG':>10}   {'H':>4}{'D':>5}{'A':>5}  {'ML':>5}{'Adv':>6}")
    print("-" * W)
    last = None
    for dt, h, a, xg, ph, pd_, pa, ml, adv in disp:
        if last is not None and dt != last:
            print("-" * W)
        last = dt
        print(f" {dt:7}{h:>16}  vs  {a:<16}{xg:>10}   "
              f"{ph*100:3.0f}%{pd_*100:4.0f}%{pa*100:4.0f}%  {ml:>5}{adv*100:5.0f}%")
    print("=" * W)
    print(" H/D/A = home win / draw / away win   ML = most likely score   Adv = home advances")

    out = os.path.join(C.OUTPUTS, "fixture_predictions.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n saved -> {out}")


if __name__ == "__main__":
    main()
