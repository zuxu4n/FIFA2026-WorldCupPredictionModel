"""Schema/parity tests for feature engineering."""
import pandas as pd
from wcpred import data as D
from wcpred.features import build_world, FEATURE_COLS


def test_feature_cols_unique():
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))


def test_matchup_row_matches_schema():
    results = D.load_results()
    _, world = build_world(results)
    row = world.matchup_row("Spain", "France", city="East Rutherford",
                            country="United States", neutral=True,
                            importance=4.0, asof=pd.Timestamp("2026-06-27"),
                            is_home=True)
    # the model consumes exactly FEATURE_COLS, in order
    assert list(row.keys()) == FEATURE_COLS


def test_elo_orders_strong_over_weak():
    results = D.load_results()
    _, world = build_world(results)
    strong = world.state("Brazil").elo
    weak = world.state("San Marino").elo
    assert strong > weak


def test_long_frame_has_targets_and_upcoming():
    results = D.load_results()
    long, _ = build_world(results)
    assert long["played"].sum() > 50_000          # two rows per played match
    assert (~long["played"]).sum() >= 2            # upcoming fixtures present
    assert set(FEATURE_COLS).issubset(long.columns)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
