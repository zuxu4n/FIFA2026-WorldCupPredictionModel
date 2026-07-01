"""Predict every upcoming (unplayed) fixture present in the data.

    python scripts/predict_fixtures.py

Writes outputs/fixture_predictions.csv and prints the table.
"""
from __future__ import annotations
import os
import pandas as pd
from _common import load_everything
from wcpred import config as C
from wcpred.predict import predict_fixture


def main():
    results, world, booster, meta = load_everything()
    upcoming = results[~results.played]
    if upcoming.empty:
        print("No unplayed fixtures in the data.")
        return
    rows = []
    for r in upcoming.itertuples(index=False):
        p = predict_fixture(booster, world, r.home_team, r.away_team,
                            city=r.city, country=r.country,
                            neutral=bool(r.neutral), importance=4.0)
        row = p.as_row()
        row = {"date": pd.Timestamp(r.date).date(), **row,
               "city": r.city, "country": r.country}
        rows.append(row)
    df = pd.DataFrame(rows)
    out = os.path.join(C.OUTPUTS, "fixture_predictions.csv")
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print(df.to_string(index=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
