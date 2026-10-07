"""Load and clean the international match results dataset.

Source: github.com/martj42/international_results (one row per match since 1872).
Rows with missing scores are scheduled fixtures; they are kept so that upcoming
matches get the same pre-match features as historical ones.
"""

from __future__ import annotations

import logging
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from wcpred import config as C

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = (
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
)


class DataError(RuntimeError):
    """Raised when an input file is missing or malformed."""


def normalize_name(name: object) -> str:
    """Strip accents and surrounding whitespace ('Curaçao' -> 'Curacao').

    Every team/venue name passes through this so reference tables join to the
    results regardless of accent spelling.
    """
    if not isinstance(name, str):
        return ""
    decomposed = unicodedata.normalize("NFKD", name)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).strip()


def classify_tournament(name: str) -> C.TournamentRule:
    """Return the first TournamentRule whose pattern occurs in `name`."""
    lowered = name.lower()
    for rule in C.TOURNAMENT_RULES:
        if rule.pattern in lowered:
            return rule
    return C.TournamentRule(pattern="", importance=C.DEFAULT_IMPORTANCE)


def load_results(path: Path | None = None) -> pd.DataFrame:
    """Read and clean results.csv (see `clean_results` for the output schema)."""
    path = path or C.PATHS.results_csv
    if not path.exists():
        raise DataError(f"{path} not found. Run `wcpred fetch-data` first.")
    return clean_results(pd.read_csv(path, encoding="utf-8"))


def clean_results(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise a raw results table.

    Output columns: the REQUIRED_COLUMNS plus `played` (bool), `importance`
    (float), `final_tournament` (bool) and `is_knockout` (bool), sorted by date.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise DataError(f"results table is missing columns: {missing}")

    df = raw.loc[:, list(REQUIRED_COLUMNS)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    n_bad = int(df["date"].isna().sum())
    if n_bad:
        log.warning("dropping %d rows with unparseable dates", n_bad)
        df = df.dropna(subset=["date"])

    for col in ("home_team", "away_team", "tournament", "city", "country"):
        df[col] = df[col].map(normalize_name)
    df["neutral"] = df["neutral"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
    for col in ("home_score", "away_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["played"] = df["home_score"].notna() & df["away_score"].notna()

    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    return add_competition_info(df)


def add_competition_info(df: pd.DataFrame) -> pd.DataFrame:
    """Add importance, final-tournament and knockout-stage columns.

    A final-tournament match is labelled knockout when both teams have already
    played GROUP_STAGE_MATCHES matches in the same edition. This uses only the
    fixture schedule (never results), so it is leak-free.
    """
    rules = {t: classify_tournament(t) for t in df["tournament"].unique()}
    df["importance"] = df["tournament"].map(lambda t: rules[t].importance).astype(float)
    df["final_tournament"] = df["tournament"].map(lambda t: rules[t].final_tournament).astype(bool)

    edition = pd.Series("", index=df.index)
    ft = df[df["final_tournament"]]
    for _, rows in ft.groupby("tournament"):
        dates = rows["date"]
        new_edition = dates.diff().dt.days.fillna(np.inf) > C.EDITION_GAP_DAYS
        starts = dates.where(new_edition).ffill()
        edition.loc[rows.index] = rows["tournament"] + "@" + starts.dt.strftime("%Y-%m-%d")

    # Number of earlier matches each side has played in the same edition.
    appearances = pd.concat(
        [
            pd.DataFrame({"row": ft.index, "edition": edition[ft.index], "team": ft["home_team"]}),
            pd.DataFrame({"row": ft.index, "edition": edition[ft.index], "team": ft["away_team"]}),
        ]
    ).sort_values("row", kind="stable")
    appearances["prior"] = appearances.groupby(["edition", "team"]).cumcount()
    min_prior = appearances.groupby("row")["prior"].min()

    df["is_knockout"] = False
    df.loc[min_prior.index, "is_knockout"] = min_prior >= C.GROUP_STAGE_MATCHES
    return df


def as_of(results: pd.DataFrame, asof: pd.Timestamp | None) -> pd.DataFrame:
    """Hide every result on or after `asof` (scores blanked, played=False).

    Fixtures stay in the table so their schedule is still known, exactly as it
    would have been on that date.
    """
    if asof is None:
        return results
    df = results.copy()
    future = df["date"] >= asof
    df.loc[future, ["home_score", "away_score"]] = np.nan
    df.loc[future, "played"] = False
    return df


def load_shootouts(path: Path | None = None, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """Penalty-shootout winners (date, home_team, away_team, winner); empty if absent."""
    path = path or C.PATHS.shootouts_csv
    columns = ["date", "home_team", "away_team", "winner"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("home_team", "away_team", "winner"):
        df[col] = df[col].map(normalize_name)
    if asof is not None:
        df = df[df["date"] < asof]
    return df.loc[:, columns].reset_index(drop=True)
