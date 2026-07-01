"""Feature engineering: turn matches into a long (one row per team-perspective)
table of leak-free features whose target is the goals that team scored.

The same arithmetic is used for (a) the historical training matrix and (b)
ad-hoc matchups during knockout simulation, via `_persp_feat`, so the two can
never silently drift apart (tests/test_features.py cross-checks them).
"""
from __future__ import annotations
from collections import deque, OrderedDict
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from .elo import compute_elo

CONFED_CODE = {
    "UEFA": 0, "CONMEBOL": 1, "CONCACAF": 2, "CAF": 3,
    "AFC": 4, "OFC": 5, "Other": 6,
}

# Canonical, ordered list of model features (train and predict must agree).
FEATURE_COLS = [
    "elo_team", "elo_opp", "elo_diff",
    "mv_team_log", "mv_opp_log", "mv_diff_log",
    "rank_pts_team", "rank_pts_opp", "rank_pts_diff",
    "is_home", "is_neutral", "at_home_country",
    "alt_match", "team_home_alt", "alt_gain",
    "team_avg_age", "opp_avg_age", "age_diff",
    "team_gf5", "team_ga5", "team_ppg5", "team_gf10", "team_ga10", "team_ppg10",
    "opp_gf5", "opp_ga5", "opp_ppg5", "opp_gf10", "opp_ga10", "opp_ppg10",
    "team_days_since", "opp_days_since", "team_nprior", "opp_nprior",
    "same_confederation", "confed_team", "confed_opp",
    "importance", "year_frac",
]


def _persp_feat(team: dict, opp: dict, ctx: dict) -> "OrderedDict":
    """Assemble one perspective's features. Inputs may be scalars or numpy
    arrays (arithmetic is elementwise either way)."""
    mv_t = np.log1p(team["mv"])
    mv_o = np.log1p(opp["mv"])
    same_conf = (np.asarray(team["confed"]) == np.asarray(opp["confed"])).astype(float)
    f = OrderedDict()
    f["elo_team"] = team["elo"]; f["elo_opp"] = opp["elo"]
    f["elo_diff"] = team["elo"] - opp["elo"]
    f["mv_team_log"] = mv_t; f["mv_opp_log"] = mv_o; f["mv_diff_log"] = mv_t - mv_o
    f["rank_pts_team"] = team["rank"]; f["rank_pts_opp"] = opp["rank"]
    f["rank_pts_diff"] = team["rank"] - opp["rank"]
    f["is_home"] = ctx["is_home"]; f["is_neutral"] = ctx["is_neutral"]
    f["at_home_country"] = ctx["at_home_country"]
    f["alt_match"] = ctx["alt_match"]; f["team_home_alt"] = team["alt_home"]
    f["alt_gain"] = ctx["alt_match"] - team["alt_home"]
    f["team_avg_age"] = team["avg_age"]
    f["opp_avg_age"] = opp["avg_age"]
    f["age_diff"] = team["avg_age"] - opp["avg_age"]
    for w in C.FORM_WINDOWS:
        f[f"team_gf{w}"] = team[f"gf{w}"]; f[f"team_ga{w}"] = team[f"ga{w}"]
        f[f"team_ppg{w}"] = team[f"ppg{w}"]
    for w in C.FORM_WINDOWS:
        f[f"opp_gf{w}"] = opp[f"gf{w}"]; f[f"opp_ga{w}"] = opp[f"ga{w}"]
        f[f"opp_ppg{w}"] = opp[f"ppg{w}"]
    f["team_days_since"] = team["days"]; f["opp_days_since"] = opp["days"]
    f["team_nprior"] = team["nprior"]; f["opp_nprior"] = opp["nprior"]
    f["same_confederation"] = same_conf
    f["confed_team"] = team["confed"]; f["confed_opp"] = opp["confed"]
    f["importance"] = ctx["importance"]; f["year_frac"] = ctx["year_frac"]
    return f


def _form_means(hist: deque):
    """Return dict of gf/ga/ppg over each window from a team's recent history
    (list of (gf, ga, pts)). NaN when no prior matches."""
    out = {}
    n = len(hist)
    recent = list(hist)
    for w in C.FORM_WINDOWS:
        sub = recent[-w:]
        if sub:
            gf = np.mean([x[0] for x in sub]); ga = np.mean([x[1] for x in sub])
            ppg = np.mean([x[2] for x in sub])
        else:
            gf = ga = ppg = np.nan
        out[f"gf{w}"] = gf; out[f"ga{w}"] = ga; out[f"ppg{w}"] = ppg
    out["nprior"] = float(n)
    return out


@dataclass
class TeamState:
    """End-of-history snapshot for a team, for building ad-hoc matchup rows."""
    elo: float
    mv: float
    rank: float
    alt_home: float
    confed: int
    hist: deque
    last_date: pd.Timestamp | None
    nprior: int

    def strength(self, asof: pd.Timestamp) -> dict:
        d = {"elo": self.elo, "mv": self.mv, "rank": self.rank,
             "alt_home": self.alt_home, "confed": self.confed}
        d.update(_form_means(self.hist))
        d["days"] = (np.nan if self.last_date is None
                     else float((asof - self.last_date).days))
        return d


@dataclass
class FeatureWorld:
    """Everything needed to featurise any matchup after a chronological pass."""
    states: dict
    alt: D.AltitudeResolver
    confed: dict
    squad_age: dict = field(default_factory=dict)
    default_confed: int = CONFED_CODE["Other"]

    def state(self, team: str) -> TeamState:
        team = D.strip_accents(team)
        if team in self.states:
            return self.states[team]
        # unseen team: neutral defaults
        return TeamState(C.ELO_START, np.nan, np.nan, 25.0,
                         self.confed.get(team, self.default_confed),
                         deque(maxlen=max(C.FORM_WINDOWS)), None, 0)

    def matchup_row(self, team: str, opp: str, *, city: str, country: str,
                    neutral: bool, importance: float, asof: pd.Timestamp,
                    is_home: bool) -> dict:
        ts = self.state(team).strength(asof)
        os_ = self.state(opp).strength(asof)
        ts["avg_age"] = self.squad_age.get((asof.year, D.strip_accents(team)), np.nan)
        os_["avg_age"] = self.squad_age.get((asof.year, D.strip_accents(opp)), np.nan)
        alt_match = self.alt.resolve(city, country, home_team=team if is_home else opp)
        ctx = {
            "is_home": float(is_home and not neutral),
            "is_neutral": float(neutral),
            "at_home_country": float(D.strip_accents(country) == D.strip_accents(team)),
            "alt_match": alt_match,
            "importance": importance,
            "year_frac": asof.year + asof.dayofyear / 365.0,
        }
        return _persp_feat(ts, os_, ctx)


def _asof_points(df: pd.DataFrame, rankings: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
    """FIFA ranking points for home/away as of each match date (or NaN)."""
    n = len(df)
    if rankings is None or "total_points" not in rankings.columns:
        return np.full(n, np.nan), np.full(n, np.nan)
    rk = rankings.dropna(subset=["total_points"]).sort_values("date")

    def lookup(team_col):
        left = df[["date", team_col]].rename(columns={team_col: "team"}).copy()
        left["_ord"] = np.arange(n)
        left = left.sort_values("date")
        merged = pd.merge_asof(left, rk[["date", "team", "total_points"]],
                               on="date", by="team", direction="backward")
        merged = merged.sort_values("_ord")
        return merged["total_points"].to_numpy()

    return lookup("home_team"), lookup("away_team")


def build_world(results: pd.DataFrame) -> tuple[pd.DataFrame, FeatureWorld]:
    """Run the chronological passes and return (long feature frame, FeatureWorld).

    The long frame has FEATURE_COLS + meta columns (match_id, date, team, opp,
    is_home_persp, played, goals, weight). Rows with played=False are upcoming
    fixtures ready to be predicted.
    """
    results = results.reset_index(drop=True)
    pre_home, pre_away, final_elo, _ = compute_elo(results)

    mv = D.load_market_values()
    confed = D.team_confederation()
    alt_res = D.AltitudeResolver()
    rankings = D.load_fifa_rankings()
    rank_home, rank_away = _asof_points(results, rankings)
    squad_age = D.load_squad_ages()

    n = len(results)
    W = max(C.FORM_WINDOWS)
    hist: dict[str, deque] = {}
    last_date: dict[str, pd.Timestamp] = {}

    # per-row, per-side form arrays
    arr = {side: {k: np.full(n, np.nan) for k in
                  [f"gf{w}" for w in C.FORM_WINDOWS] +
                  [f"ga{w}" for w in C.FORM_WINDOWS] +
                  [f"ppg{w}" for w in C.FORM_WINDOWS] + ["days", "nprior"]}
           for side in ("h", "a")}

    ht = results["home_team"].to_numpy(); at = results["away_team"].to_numpy()
    hs = results["home_score"].to_numpy(); as_ = results["away_score"].to_numpy()
    played = results["played"].to_numpy(); dates = results["date"].to_numpy()
    dts = results["date"]

    for i in range(n):
        for side, team in (("h", ht[i]), ("a", at[i])):
            h = hist.get(team)
            fm = _form_means(h) if h is not None else _form_means(deque())
            for k, v in fm.items():
                arr[side][k][i] = v
            ld = last_date.get(team)
            arr[side]["days"][i] = (np.nan if ld is None
                                    else float((dates[i] - ld) / np.timedelta64(1, "D")))
        if played[i]:
            ph = 3 if hs[i] > as_[i] else (1 if hs[i] == as_[i] else 0)
            pa = 3 if as_[i] > hs[i] else (1 if hs[i] == as_[i] else 0)
            hist.setdefault(ht[i], deque(maxlen=W)).append((hs[i], as_[i], ph))
            hist.setdefault(at[i], deque(maxlen=W)).append((as_[i], hs[i], pa))
            last_date[ht[i]] = dates[i]; last_date[at[i]] = dates[i]

    # static per-side maps
    def smap(teams, d, default=np.nan):
        return np.array([d.get(t, default) for t in teams], dtype=float)

    confed_code_map = {t: CONFED_CODE.get(c, 6) for t, c in confed.items()}
    home_confed = np.array([confed_code_map.get(t, 6) for t in ht], dtype=float)
    away_confed = np.array([confed_code_map.get(t, 6) for t in at], dtype=float)
    home_mv = smap(ht, mv); away_mv = smap(at, mv)
    home_alt = np.array([alt_res.team_alt.get(D.strip_accents(t), 25.0) for t in ht])
    away_alt = np.array([alt_res.team_alt.get(D.strip_accents(t), 25.0) for t in at])
    alt_match = np.array([alt_res.resolve(c, co, hm) for c, co, hm in
                          zip(results["city"], results["country"], ht)])

    neutral = results["neutral"].to_numpy()
    importance = results["importance"].to_numpy()
    country = results["country"].to_numpy()
    year_frac = dts.dt.year.to_numpy() + dts.dt.dayofyear.to_numpy() / 365.0
    years = dts.dt.year.to_numpy()
    home_age = np.array([squad_age.get((int(y), D.strip_accents(t)), np.nan)
                         for y, t in zip(years, ht)])
    away_age = np.array([squad_age.get((int(y), D.strip_accents(t)), np.nan)
                         for y, t in zip(years, at)])

    def side_state(side, teams, elo, mvv, rankv, altv, confv, agev):
        s = {"elo": elo, "mv": mvv, "rank": rankv, "alt_home": altv,
             "confed": confv, "avg_age": agev}
        for k in arr[side]:
            if k == "nprior":
                s["nprior"] = arr[side]["nprior"]
            elif k == "days":
                s["days"] = arr[side]["days"]
            else:
                s[k] = arr[side][k]
        return s

    home_state = side_state("h", ht, pre_home, home_mv, rank_home, home_alt, home_confed, home_age)
    away_state = side_state("a", at, pre_away, away_mv, rank_away, away_alt, away_confed, away_age)

    base_ctx_home = {
        "is_home": (~neutral).astype(float), "is_neutral": neutral.astype(float),
        "at_home_country": np.array([float(D.strip_accents(co) == D.strip_accents(t))
                                     for co, t in zip(country, ht)]),
        "alt_match": alt_match, "importance": importance, "year_frac": year_frac,
    }
    base_ctx_away = {
        "is_home": np.zeros(n), "is_neutral": neutral.astype(float),
        "at_home_country": np.array([float(D.strip_accents(co) == D.strip_accents(t))
                                     for co, t in zip(country, at)]),
        "alt_match": alt_match, "importance": importance, "year_frac": year_frac,
    }

    feat_home = _persp_feat(home_state, away_state, base_ctx_home)
    feat_away = _persp_feat(away_state, home_state, base_ctx_away)

    def to_frame(feat, team, opp, goals, is_home_persp):
        d = pd.DataFrame(feat)
        d["match_id"] = np.arange(n)
        d["date"] = results["date"].values
        d["team"] = team; d["opp"] = opp
        d["is_home_persp"] = is_home_persp
        d["played"] = played
        d["goals"] = goals
        return d

    long = pd.concat(
        [to_frame(feat_home, ht, at, hs, 1), to_frame(feat_away, at, ht, as_, 0)],
        ignore_index=True,
    )
    # recency + importance sample weight
    age_years = (pd.Timestamp("today").normalize() - long["date"]).dt.days / 365.25
    decay = np.power(0.5, age_years / C.RECENCY_HALFLIFE_YEARS)
    long["weight"] = long["importance"] * decay

    # final per-team states for ad-hoc matchups
    states: dict[str, TeamState] = {}
    all_teams = set(ht) | set(at)
    for t in all_teams:
        states[t] = TeamState(
            elo=final_elo.get(t, C.ELO_START),
            mv=mv.get(t, np.nan),
            rank=np.nan,  # latest rank filled below if available
            alt_home=alt_res.team_alt.get(D.strip_accents(t), 25.0),
            confed=confed_code_map.get(t, 6),
            hist=hist.get(t, deque(maxlen=W)),
            last_date=last_date.get(t),
            nprior=len(hist.get(t, [])),
        )
    if rankings is not None and "total_points" in rankings.columns:
        latest = rankings.sort_values("date").groupby("team")["total_points"].last()
        for t, s in states.items():
            if t in latest.index:
                s.rank = float(latest[t])

    world = FeatureWorld(states=states, alt=alt_res, confed=confed_code_map,
                         squad_age=squad_age)
    return long, world
