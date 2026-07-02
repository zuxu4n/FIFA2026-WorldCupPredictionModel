"""Load and clean the raw match data and the static reference tables."""
from __future__ import annotations
import unicodedata
import numpy as np
import pandas as pd

from . import config as C


def strip_accents(s: str) -> str:
    """'Curaçao' -> 'Curacao' so reference tables join regardless of accents."""
    if not isinstance(s, str):
        return s
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    ).strip()


def tournament_importance(name: str) -> float:
    n = str(name).lower()
    for key, w in C.TOURNAMENT_IMPORTANCE:
        if key in n:
            return w
    return C.DEFAULT_IMPORTANCE


def load_results() -> pd.DataFrame:
    """Return the cleaned match table.

    Columns: date, home_team, away_team, home_score, away_score, tournament,
    city, country, neutral (bool), importance (float), played (bool).
    Rows with NA scores are kept (they are upcoming fixtures to predict).
    """
    df = pd.read_csv(C.RESULTS_CSV, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    for col in ("home_team", "away_team", "tournament", "city", "country"):
        df[col] = df[col].astype(str).map(strip_accents)
    # neutral may be TRUE/FALSE strings or bools
    df["neutral"] = (
        df["neutral"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
    )
    for col in ("home_score", "away_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["played"] = df["home_score"].notna() & df["away_score"].notna()
    df["importance"] = df["tournament"].map(tournament_importance)
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    return df


def load_team_meta() -> pd.DataFrame:
    df = pd.read_csv(C.TEAM_META_CSV, encoding="utf-8")
    df["team"] = df["team"].map(strip_accents)
    return df


def team_confederation() -> dict[str, str]:
    df = load_team_meta()
    return dict(zip(df["team"], df["confederation"]))


def team_home_altitude() -> dict[str, float]:
    df = load_team_meta()
    return dict(zip(df["team"], df["home_altitude_m"].astype(float)))


def load_venue_altitudes() -> dict[tuple[str, str], float]:
    df = pd.read_csv(C.VENUES_CSV, encoding="utf-8")
    df["city"] = df["city"].map(strip_accents)
    df["country"] = df["country"].map(strip_accents)
    return {
        (r.city, r.country): float(r.altitude_m) for r in df.itertuples(index=False)
    }


def load_market_values() -> dict[str, float]:
    df = pd.read_csv(C.MARKET_VALUES_CSV, encoding="utf-8")
    df["team"] = df["team"].map(strip_accents)
    return dict(zip(df["team"], df["squad_value_eur_m"].astype(float)))


def load_injuries() -> dict[str, float]:
    """team -> market value (EUR millions) currently unavailable (injuries,
    suspensions). Manually maintained in data/reference/injuries.csv; applied
    at predict time by reducing the squad-value feature. Empty if absent.
    """
    import os
    path = os.path.join(C.DATA_REF, "injuries.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, encoding="utf-8")
    if df.empty or "team" not in df.columns:
        return {}
    df["team"] = df["team"].map(strip_accents)
    df["value_out_eur_m"] = pd.to_numeric(df["value_out_eur_m"], errors="coerce")
    df = df.dropna(subset=["value_out_eur_m"])
    return df.groupby("team")["value_out_eur_m"].sum().to_dict()


def load_squad_ages() -> dict[tuple[int, str], float]:
    """(tournament_year, team) -> average squad age. Empty if file absent.

    Joined temporally to matches (a year's squad age for that year's matches),
    so it carries no future leakage. Schema: year,team,avg_age,age_std,n_players.
    """
    import os
    path = os.path.join(C.DATA_REF, "squad_ages.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, encoding="utf-8")
    df["team"] = df["team"].map(strip_accents)
    return {(int(r.year), r.team): float(r.avg_age) for r in df.itertuples(index=False)}


def load_fifa_rankings() -> pd.DataFrame | None:
    """Optional. CSV schema: date, team, rank, total_points.

    Returns a frame sorted by date for as-of joins, or None if not provided.
    """
    import os
    if not os.path.exists(C.FIFA_RANKINGS_CSV):
        return None
    df = pd.read_csv(C.FIFA_RANKINGS_CSV, encoding="utf-8")
    df["team"] = df["team"].map(strip_accents)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "total_points" not in df.columns and "points" in df.columns:
        df = df.rename(columns={"points": "total_points"})
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def load_team_climate() -> pd.DataFrame:
    df = pd.read_csv(C.TEAM_CLIMATE_CSV, encoding="utf-8")
    df["team"] = df["team"].map(strip_accents)
    return df


class ClimateResolver:
    """Resolve match-day temperature (C) and humidity (%) at a venue.

    Priority: actual per-date forecast override (weather_overrides.csv, written
    by scripts/fetch_weather.py) -> curated WC2026 venue climate normal -> the
    match country's home-team climate -> a mild default. Historical training
    uses normals (systematic effect); overrides sharpen upcoming predictions.
    """

    def __init__(self):
        vc = pd.read_csv(C.VENUE_CLIMATE_CSV, encoding="utf-8")
        vc["city"] = vc["city"].map(strip_accents)
        vc["country"] = vc["country"].map(strip_accents)
        self.venues = {(r.city, r.country): (float(r.temp_c), float(r.humidity))
                       for r in vc.itertuples(index=False)}
        tc = load_team_climate()
        self.team_temp = dict(zip(tc["team"], tc["home_temp_c"].astype(float)))
        self.team_humid = dict(zip(tc["team"], tc["home_humidity"].astype(float)))
        # per-date forecast overrides (optional)
        import os
        self.overrides: dict[tuple[str, str, str], tuple[float, float]] = {}
        ov_path = os.path.join(C.DATA_REF, "weather_overrides.csv")
        if os.path.exists(ov_path):
            ov = pd.read_csv(ov_path, encoding="utf-8")
            ov["city"] = ov["city"].map(strip_accents)
            self.overrides = {
                (str(r.date), r.city, strip_accents(str(r.country))):
                    (float(r.temp_c), float(r.humidity))
                for r in ov.itertuples(index=False)
            }

    def resolve(self, city: str, country: str, home_team: str | None = None,
                date=None):
        key = (strip_accents(city), strip_accents(country))
        if date is not None and self.overrides:
            dkey = (pd.Timestamp(date).date().isoformat(), key[0], key[1])
            if dkey in self.overrides:
                return self.overrides[dkey]
        if key in self.venues:
            return self.venues[key]
        co = strip_accents(country)
        if co in self.team_temp:                      # match country is a team name
            return self.team_temp[co], self.team_humid[co]
        if home_team and strip_accents(home_team) in self.team_temp:
            t = strip_accents(home_team)
            return self.team_temp[t], self.team_humid[t]
        return 22.0, 60.0


class AltitudeResolver:
    """Resolve the altitude (m) of a match venue.

    Priority: exact (city, country) in venues.csv -> the home nation's typical
    home-venue altitude (if the match is in that nation) -> sea-level default.
    """

    def __init__(self):
        self.venues = load_venue_altitudes()
        self.team_alt = team_home_altitude()
        # country -> a representative team's home altitude (for in-country fallback)
        meta = load_team_meta()
        self.country_alt: dict[str, float] = {}
        for _, r in meta.iterrows():
            # team name often == country for national sides
            self.country_alt.setdefault(r["team"], float(r["home_altitude_m"]))

    def resolve(self, city: str, country: str, home_team: str | None = None) -> float:
        key = (strip_accents(city), strip_accents(country))
        if key in self.venues:
            return self.venues[key]
        co = strip_accents(country)
        if co in self.country_alt:           # match country is a national team name
            return self.country_alt[co]
        if home_team and strip_accents(home_team) in self.team_alt:
            return self.team_alt[strip_accents(home_team)]
        return 25.0  # sea-level-ish default
