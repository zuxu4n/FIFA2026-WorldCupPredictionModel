"""Predict a single matchup.

    python scripts/predict_match.py --home Spain --away France
    python scripts/predict_match.py --home Mexico --away Brazil --city "Mexico City" --country Mexico --host
    python scripts/predict_match.py --home Argentina --away England --knockout
"""
from __future__ import annotations
import argparse
from _common import load_everything
from wcpred.predict import predict_fixture


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--city", default="East Rutherford")
    ap.add_argument("--country", default="United States")
    ap.add_argument("--host", action="store_true",
                    help="treat as a true home game (not neutral)")
    ap.add_argument("--knockout", action="store_true",
                    help="also show advancement (ET + penalties)")
    ap.add_argument("--why", action="store_true",
                    help="show what drove each side's expected goals (SHAP)")
    ap.add_argument("--importance", type=float, default=4.0)
    args = ap.parse_args()

    _, world, booster, meta = load_everything()
    p = predict_fixture(booster, world, args.home, args.away, city=args.city,
                        country=args.country, neutral=not args.host,
                        importance=args.importance)

    print(f"\n  {p.home}  vs  {p.away}   ({args.city}, {args.country}"
          f"{'' if args.host else ', neutral'})")
    print(f"  expected goals:  {p.home} {p.lam_home:.2f}  -  {p.lam_away:.2f} {p.away}")
    print(f"  {p.home} win : {p.p_home_win:6.1%}")
    print(f"  draw        : {p.p_draw:6.1%}")
    print(f"  {p.away} win : {p.p_away_win:6.1%}")
    if args.knockout:
        print(f"  advance (ET+pens):  {p.home} {p.p_home_advance:.1%}  |  "
              f"{p.away} {p.p_away_advance:.1%}")
    print("  most likely scorelines:")
    for (i, j), pr in p.top_scorelines:
        print(f"    {p.home} {i}-{j} {p.away}   {pr:5.1%}")

    if args.why:
        import pandas as pd
        from wcpred.model import explain
        asof = pd.Timestamp("today").normalize()
        for side, opp_, is_home in ((args.home, args.away, True),
                                    (args.away, args.home, False)):
            row = world.matchup_row(side, opp_, city=args.city, country=args.country,
                                    neutral=not args.host, importance=args.importance,
                                    asof=asof, is_home=is_home and args.host)
            print(f"\n  why {side}'s expected goals (top drivers, % effect on xG):")
            for feat, _, pct in explain(booster, row)[:6]:
                print(f"    {feat:20s} {pct:+7.1%}")


if __name__ == "__main__":
    main()
