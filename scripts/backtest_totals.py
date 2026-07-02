"""Quantify + fix the total-goals bias: fit per-segment lambda shrinkage and
the Dixon-Coles rho on out-of-sample predictions, then report before/after.

    python scripts/backtest_totals.py [--cutoff 2023-01-01] [--no-save]
"""
from __future__ import annotations
import argparse
import numpy as np
from wcpred.calibrate import (build_oos_frame, fit_totals_and_rho, segment_of,
                              scale_lambdas, invalidate_totals_cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2023-01-01")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    df = build_oos_frame(args.cutoff)
    factors, rho, _ = fit_totals_and_rho(args.cutoff, save=not args.no_save, df=df)

    # after-calibration residual bias per segment
    invalidate_totals_cache()
    seg = df.apply(lambda r: segment_of(r.importance, r.is_knockout), axis=1)
    print("\nresidual bias after calibration (pred - actual, goals/match):")
    for s in sorted(seg.unique()):
        m = seg == s
        scaled = [sum(scale_lambdas(r.lam_h, r.lam_a, r.importance, r.is_knockout))
                  for r in df[m].itertuples(index=False)]
        actual = (df.loc[m, "gh"] + df.loc[m, "ga"]).to_numpy()
        print(f"  {s:18} n={int(m.sum()):>5}  bias {np.mean(scaled)-actual.mean():+6.3f}"
              f"  (was {(df.loc[m,'lam_h']+df.loc[m,'lam_a']).mean()-actual.mean():+6.3f})")
    # totals MAE
    mae = np.mean([abs(sum(scale_lambdas(r.lam_h, r.lam_a, r.importance, r.is_knockout))
                       - (r.gh + r.ga)) for r in df.itertuples(index=False)])
    print(f"\ntotal-goals MAE (calibrated): {mae:.3f}")


if __name__ == "__main__":
    main()
