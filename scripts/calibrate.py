"""Fit the full calibration stack, in dependency order:

1. lambda (totals) shrinkage per segment + Dixon-Coles rho
2. closeness-conditional W/D/L calibrator (sits on top of 1)

    python scripts/calibrate.py

Both share one out-of-sample dataset (train < cutoff, predict >= cutoff),
so the expensive model refit happens once.
"""
from wcpred.calibrate import build_oos_frame, fit_totals_and_rho, fit_and_report

if __name__ == "__main__":
    df = build_oos_frame()
    fit_totals_and_rho(df=df)
    print()
    fit_and_report(df=df)
