"""Static and slowly-changing reference tables used as features.

Provenance and accuracy of each table are documented in data/README.md. The
altitude and climate tables are hand-curated approximations; FIFA rankings and
squad ages come from public datasets via the import scripts in scripts/.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wcpred import config as C
from wcpred.data.results import DataError, normalize_name

CONFEDERATION_CODES = {
    "UEFA": 0,
    "CONMEBOL": 1,
    "CONCACAF": 2,
    "CAF": 3,
    "AFC": 4,
    "OFC": 5,
}
UNKNOWN_CONFEDERATION = 6


def _read_optional(directory: Path, name: str) -> pd.DataFrame | None:
    path = directory / name
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8")
    for col in ("team", "city", "country"):
        if col in df.columns:
            df[col] = df[col].map(normalize_name)
    return df


def _columns(df: pd.DataFrame, *cols: str) -> Iterator[tuple[Any, ...]]:
    """Iterate rows of selected columns as plain tuples."""
    return zip(*(df[c].tolist() for c in cols), strict=True)


def _read(directory: Path, name: str) -> pd.DataFrame:
    df = _read_optional(directory, name)
    if df is None:
        raise DataError(f"reference table {directory / name} is missing")
    return df


@dataclass
class ReferenceData:
    """All reference lookups, keyed by normalised team / (city, country) names."""

    confederation: dict[str, int]
    team_altitude: dict[str, float]
    venue_altitude: dict[tuple[str, str], float]
    team_climate: dict[str, tuple[float, float]]
    venue_climate: dict[tuple[str, str], tuple[float, float]]
    market_value: dict[str, float]
    squad_age: dict[tuple[int, str], float]
    fifa_rankings: pd.DataFrame  # columns: date, team, total_points (sorted by date)

    @classmethod
    def load(cls, directory: Path | None = None, asof: pd.Timestamp | None = None) -> ReferenceData:
        """Load every table; time-stamped tables are truncated before `asof`."""
        directory = directory or C.PATHS.reference_dir
        meta = _read(directory, "team_meta.csv")
        venues = _read(directory, "venues.csv")
        team_climate = _read(directory, "team_climate.csv")
        venue_climate = _read(directory, "venue_climate.csv")
        market = _read(directory, "market_values.csv")
        ages = _read_optional(directory, "squad_ages.csv")
        rankings = _read_optional(directory, "fifa_rankings.csv")

        if rankings is None:
            rankings = pd.DataFrame(columns=["date", "team", "total_points"])
        rankings["date"] = pd.to_datetime(rankings["date"], errors="coerce")
        rankings = rankings.dropna(subset=["date", "total_points"])
        if ages is None:
            ages = pd.DataFrame(columns=["year", "team", "avg_age"])
        if asof is not None:
            rankings = rankings[rankings["date"] < asof]
            ages = ages[ages["year"] <= asof.year]

        return cls(
            confederation={
                t: CONFEDERATION_CODES.get(c, UNKNOWN_CONFEDERATION)
                for t, c in _columns(meta, "team", "confederation")
            },
            team_altitude={t: float(a) for t, a in _columns(meta, "team", "home_altitude_m")},
            venue_altitude={
                (c, co): float(a) for c, co, a in _columns(venues, "city", "country", "altitude_m")
            },
            team_climate={
                t: (float(tc), float(h))
                for t, tc, h in _columns(team_climate, "team", "home_temp_c", "home_humidity")
            },
            venue_climate={
                (c, co): (float(tc), float(h))
                for c, co, tc, h in _columns(venue_climate, "city", "country", "temp_c", "humidity")
            },
            market_value={t: float(v) for t, v in _columns(market, "team", "squad_value_eur_m")},
            squad_age={
                (int(y), t): float(a) for y, t, a in _columns(ages, "year", "team", "avg_age")
            },
            fifa_rankings=(
                rankings.loc[:, ["date", "team", "total_points"]]
                .sort_values("date", kind="stable")
                .reset_index(drop=True)
            ),
        )

    # -- venue lookups ---------------------------------------------------
    # Fallback order for both: exact venue -> the host country's own home
    # conditions (national team name == country) -> the home side's -> default.

    def altitude(self, city: str, country: str, home_team: str) -> float:
        if (city, country) in self.venue_altitude:
            return self.venue_altitude[(city, country)]
        for key in (country, home_team):
            if key in self.team_altitude:
                return self.team_altitude[key]
        return C.DEFAULT_ALTITUDE_M

    def climate(self, city: str, country: str, home_team: str) -> tuple[float, float]:
        if (city, country) in self.venue_climate:
            return self.venue_climate[(city, country)]
        for key in (country, home_team):
            if key in self.team_climate:
                return self.team_climate[key]
        return (C.DEFAULT_TEMP_C, C.DEFAULT_HUMIDITY)

    def home_temp(self, team: str) -> float:
        return self.team_climate.get(team, (C.DEFAULT_TEMP_C, C.DEFAULT_HUMIDITY))[0]

    # -- time-varying lookups -------------------------------------------

    def ranking_points(self, teams: np.ndarray, dates: np.ndarray) -> np.ndarray:
        """FIFA ranking points of each team as of each date (NaN if unknown)."""
        if self.fifa_rankings.empty or len(teams) == 0:
            return np.full(len(teams), np.nan)
        left = pd.DataFrame(
            {"date": pd.to_datetime(dates), "team": teams, "_order": np.arange(len(teams))}
        ).sort_values("date", kind="stable")
        merged = pd.merge_asof(
            left, self.fifa_rankings, on="date", by="team", direction="backward"
        ).sort_values("_order")
        return merged["total_points"].to_numpy(dtype=float)

    def world_cup_squad_age(self, year: int, team: str) -> float:
        """Average age of a team's World Cup squad that year (NaN if not at that WC)."""
        return self.squad_age.get((year, team), np.nan)
