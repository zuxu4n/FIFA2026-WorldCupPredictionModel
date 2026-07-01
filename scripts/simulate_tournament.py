"""Monte-Carlo the rest of the World Cup for advancement + title odds.

    python scripts/simulate_tournament.py --sims 10000

By default this follows the OFFICIAL 2026 knockout bracket (real slot map +
venues, incl. altitude in Mexico City). Use --seeded to fall back to the
strength-seeded approximation. Writes outputs/title_odds.csv.
"""
from __future__ import annotations
import argparse
import os
import pandas as pd
from _common import load_everything
from wcpred import config as C
from wcpred.bracket import simulate_official, FINALS_TXT
from wcpred.simulate import simulate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--seeded", action="store_true",
                    help="use the strength-seeded bracket instead of the official one")
    args = ap.parse_args()

    results, world, booster, meta = load_everything()
    use_official = (not args.seeded) and os.path.exists(FINALS_TXT)
    mode = "official 2026 bracket" if use_official else "strength-seeded bracket"
    if args.seeded is False and not use_official:
        print("(official bracket not found — run scripts/import_bracket.py; "
              "falling back to seeded)")
    print(f"simulating {args.sims:,} tournaments  [{mode}]...")

    if use_official:
        out = simulate_official(booster, world, results, n_sims=args.sims, seed=args.seed)
        pct = ["P_win_group", "P_advance", "P_R16", "P_QF", "P_SF", "P_final", "P_champion"]
    else:
        out = simulate(booster, world, results, n_sims=args.sims, seed=args.seed)
        pct = ["P_win_group", "P_advance", "P_QF", "P_SF", "P_final", "P_champion"]

    tab = out["table"].copy()
    disp = tab.copy()
    for c in pct:
        disp[c] = (disp[c] * 100).round(1)
    pd.set_option("display.width", 220)
    print(f"\n=== Advancement & title odds (%)  [{mode}] ===")
    print(disp.head(args.top).to_string(index=False))

    path = os.path.join(C.OUTPUTS, "title_odds.csv")
    tab.to_csv(path, index=False)
    print(f"\nsaved -> {path}")
    print("\nGroups:")
    for g, ts in out["groups"].items():
        print(f"  {g}: {', '.join(ts)}")


if __name__ == "__main__":
    main()
