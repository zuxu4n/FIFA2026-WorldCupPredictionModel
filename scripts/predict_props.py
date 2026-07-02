"""Full derived-market board for one fixture, from the calibrated score matrix.

    python scripts/predict_props.py --home Spain --away Austria --city Inglewood --country "United States"
    python scripts/predict_props.py --home Spain --away Austria --et            # markets incl. extra time
    python scripts/predict_props.py ... --odds my_odds.csv                      # adds market/EV/kelly columns

--odds CSV schema:  selection,american     e.g.  "Total Over 2.5",-188
Selection names must match the board's row labels exactly.
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
from _common import load_everything
from wcpred.odds import american_to_decimal, decimal_to_american, ev, kelly
from wcpred.predict import (predict_lambdas, score_matrix, score_matrix_incl_et,
                            wdl_from_matrix, knockout_advance, top_scorelines)


def board(home, away, lh, la, incl_et: bool):
    M90 = score_matrix(lh, la)
    M = score_matrix_incl_et(lh, la) if incl_et else M90
    G = M.shape[0]

    def P(cond):
        return float(sum(M[i, j] for i in range(G) for j in range(G) if cond(i, j)))

    wh, dr, wa = wdl_from_matrix(M90)
    adv_h, _ = knockout_advance(lh, la)
    rows = [
        (f"{home} win (90')", wh), ("Draw (90')", dr), (f"{away} win (90')", wa),
        (f"{home} or Draw", wh + dr), (f"{away} or Draw", wa + dr),
        (f"{home} DNB", wh / (wh + wa)), (f"{away} DNB", wa / (wh + wa)),
        (f"{home} to advance", adv_h), (f"{away} to advance", 1 - adv_h),
    ]
    for line in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5):
        rows.append((f"Total Over {line}", P(lambda i, j: i + j > line)))
        rows.append((f"Total Under {line}", P(lambda i, j: i + j < line)))
    for line in (0.5, 1.5, 2.5):
        rows.append((f"{home} Over {line}", P(lambda i, j: i > line)))
        rows.append((f"{home} Under {line}", P(lambda i, j: i < line)))
        rows.append((f"{away} Over {line}", P(lambda i, j: j > line)))
        rows.append((f"{away} Under {line}", P(lambda i, j: j < line)))
    rows += [
        ("Both teams score", P(lambda i, j: i >= 1 and j >= 1)),
        (f"{home} clean sheet", P(lambda i, j: j == 0)),
        (f"{away} clean sheet", P(lambda i, j: i == 0)),
        (f"{home} win to nil", P(lambda i, j: i > j and j == 0)),
        (f"{away} win to nil", P(lambda i, j: j > i and i == 0)),
    ]
    for h in (-0.5, -1.0, -1.5):
        rows.append((f"{home} {h:+g} (AH)", P(lambda i, j: i + h > j)))
    return rows, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--city", default="East Rutherford")
    ap.add_argument("--country", default="United States")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--et", action="store_true",
                    help="derive goal markets from the incl-extra-time distribution")
    ap.add_argument("--odds", help="CSV (selection,american) to compute EV against")
    args = ap.parse_args()

    _, world, booster, meta = load_everything()
    lh, la = predict_lambdas(booster, world, args.home, args.away, city=args.city,
                             country=args.country, neutral=not args.host,
                             importance=4.0)
    rows, M = board(args.home, args.away, lh, la, args.et)

    mkt = {}
    if args.odds:
        odf = pd.read_csv(args.odds)
        mkt = {str(r.selection): american_to_decimal(r.american)
               for r in odf.itertuples(index=False)}

    basis = "incl. extra time" if args.et else "regulation (90')"
    print(f"\n {args.home} vs {args.away}  @ {args.city}"
          f"{'' if args.host else ' (neutral)'}   [{basis} goal markets]")
    print(f" model xG: {args.home} {lh:.2f} - {la:.2f} {args.away}"
          f"   (calibrated; model through {meta['trained_through']})")
    print("-" * 66)
    hdr = f" {'market':30}{'model':>8}{'fair':>7}"
    if mkt:
        hdr += f"{'book':>8}{'EV':>8}{'kelly':>7}"
    print(hdr)
    for name, p in rows:
        d = 1 / max(p, 1e-9)
        line = f" {name:30}{p:>7.1%}{decimal_to_american(d) if p>0.01 else '  -':>8}"
        if mkt and name in mkt:
            dd = mkt[name]
            line += f"{decimal_to_american(dd):>8}{ev(p, dd):>+8.1%}{kelly(p, dd):>7.1%}"
        print(line)
    print("-" * 66)
    print(" most likely scorelines (90'):")
    for (i, j), pr in top_scorelines(score_matrix(lh, la), 5):
        print(f"   {args.home} {i}-{j} {args.away}   {pr:5.1%}")


if __name__ == "__main__":
    main()
