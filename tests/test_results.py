import numpy as np
import pandas as pd
import pytest
from conftest import make_results

from wcpred.data import DataError, as_of, classify_tournament, clean_results, normalize_name


class TestClassifyTournament:
    def test_qualifiers_never_outrank_world_cup_qualifiers(self):
        # Regression: substring order used to give "UEFA Euro qualification" 3.5 and
        # "African Cup of Nations qualification" 3.0, above World Cup qualifiers (2.5).
        wcq = classify_tournament("FIFA World Cup qualification")
        for name in (
            "UEFA Euro qualification",
            "African Cup of Nations qualification",
            "AFC Asian Cup qualification",
            "CONCACAF Nations League qualification",
        ):
            rule = classify_tournament(name)
            assert rule.importance < wcq.importance, name
            assert not rule.final_tournament, name

    def test_final_tournaments(self):
        assert classify_tournament("FIFA World Cup").importance == 4.0
        assert classify_tournament("FIFA World Cup").final_tournament
        assert classify_tournament("UEFA Euro").final_tournament
        assert classify_tournament("Copa America").final_tournament

    def test_league_and_friendlies_are_not_final_tournaments(self):
        assert not classify_tournament("UEFA Nations League").final_tournament
        assert classify_tournament("Friendly").importance == 1.0

    def test_unknown_tournament_gets_default(self):
        assert classify_tournament("Some Regional Cup").importance == 2.0


def test_normalize_name_strips_accents_and_whitespace():
    assert normalize_name(" Curaçao ") == "Curacao"
    assert normalize_name("Côte d'Ivoire") == "Cote d'Ivoire"
    assert normalize_name(np.nan) == ""


def test_clean_results_types_and_order():
    df = make_results(
        [
            ("2020-02-01", "B", "A", 1, 1, "Friendly", "X", "B", "FALSE"),
            ("2020-01-01", "Curaçao", "A", 2, 0, "Friendly", "X", "Y", "TRUE"),
            ("2020-03-01", "A", "B", None, None, "Friendly", "X", "Y", "1"),
            ("not a date", "A", "B", 0, 0, "Friendly", "X", "Y", "TRUE"),
        ]
    )
    assert list(df["home_team"]) == ["Curacao", "B", "A"]  # sorted, bad date dropped
    assert list(df["neutral"]) == [True, False, True]
    assert list(df["played"]) == [True, True, False]


def test_missing_columns_raise():
    with pytest.raises(DataError, match="missing columns"):
        clean_results(pd.DataFrame({"date": ["2020-01-01"]}))


def _group_and_final(tournament: str, start: str) -> list[tuple]:
    """Four teams play a round robin (3 matches each), then two of them meet again."""
    d0 = pd.Timestamp(start)
    pairs = [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("A", "D"), ("B", "C")]
    rows = [
        (d0 + pd.Timedelta(days=2 * i), h, a, 1, 0, tournament, "X", "Y", True)
        for i, (h, a) in enumerate(pairs)
    ]
    rows.append((d0 + pd.Timedelta(days=16), "A", "B", 2, 1, tournament, "X", "Y", True))
    return rows


def test_knockout_is_a_teams_fourth_match_in_a_final_tournament():
    df = make_results(_group_and_final("FIFA World Cup", "2018-06-14"))
    assert df["is_knockout"].tolist() == [False] * 6 + [True]


def test_league_matches_are_never_knockout():
    # Regression: the old proxy (importance >= 3 and > 14 days into the year's
    # competition) flagged most Nations League and Euro qualifier matches.
    df = make_results(_group_and_final("UEFA Nations League", "2018-09-06"))
    assert not df["is_knockout"].any()


def test_edition_spanning_new_year_is_one_edition():
    # AFCON 2025 ran from December into January; grouping by calendar year split it.
    df = make_results(_group_and_final("African Cup of Nations", "2025-12-21"))
    assert df["date"].iloc[-1].year == 2026
    assert df["is_knockout"].iloc[-1]


def test_separate_editions_restart_the_count():
    rows = _group_and_final("FIFA World Cup", "2018-06-14")
    rows += _group_and_final("FIFA World Cup", "2022-11-20")
    df = make_results(rows)
    assert df["is_knockout"].sum() == 2


def test_as_of_hides_results_but_keeps_fixtures():
    df = make_results(
        [
            ("2026-06-10", "A", "B", 1, 0, "Friendly", "X", "Y", True),
            ("2026-06-11", "A", "B", 2, 2, "Friendly", "X", "Y", True),
        ]
    )
    past = as_of(df, pd.Timestamp("2026-06-11"))
    assert past["played"].tolist() == [True, False]
    assert np.isnan(past["home_score"].iloc[1])
    assert len(past) == 2
    assert df["played"].all()  # input not mutated
