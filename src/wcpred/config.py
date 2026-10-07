"""Central configuration: file locations, data sources, and model constants.

Everything that would otherwise be a magic number lives here so experiments are
changed in one place and recorded in the saved model metadata.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Paths:
    """Project directory layout. Override the root with the WCPRED_ROOT env var."""

    root: Path

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def reference_dir(self) -> Path:
        return self.root / "data" / "reference"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def results_csv(self) -> Path:
        return self.raw_dir / "results.csv"

    @property
    def shootouts_csv(self) -> Path:
        return self.raw_dir / "shootouts.csv"

    def model_dir(self, asof: pd.Timestamp | None = None) -> Path:
        """Directory of a trained model; point-in-time models get their own folder."""
        name = "model" if asof is None else f"model-asof-{asof.date().isoformat()}"
        return self.artifacts_dir / name


PATHS = Paths(Path(os.environ.get("WCPRED_ROOT", Path(__file__).resolve().parents[2])))

# --------------------------------------------------------------------------
# Raw data source (pinned for reproducibility)
# --------------------------------------------------------------------------
# github.com/martj42/international_results (CC0). Pinning a commit means every
# number in the README can be regenerated exactly; `wcpred fetch-data --ref master`
# pulls the latest data instead.
RESULTS_REPO_RAW = "https://raw.githubusercontent.com/martj42/international_results"
RESULTS_REF = "b7a3a8ee5d37780c6a03101c188658fd439e241d"  # 2026-10-07
RESULTS_SHA256 = {
    "results.csv": "b38434f8f74cd9faacb269eb1c97760fc2dfac43fcb211779e0bd2cbb48f89e3",
    "shootouts.csv": "072b445adf5183fbe981087857e5b64c8f17abfdbf257b1822ceacb4a37919fa",
}

# --------------------------------------------------------------------------
# Competitions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TournamentRule:
    """Maps tournament names containing `pattern` (case-insensitive) to a weight.

    `importance` scales the Elo K-factor and the training sample weight.
    `final_tournament` marks group-then-knockout events (World Cup, Euro, ...),
    the only competitions where a knockout stage is detected.
    """

    pattern: str
    importance: float
    final_tournament: bool = False


# First match wins, so qualifiers must come before the tournaments they qualify
# for ("UEFA Euro qualification" must not match "uefa euro").
TOURNAMENT_RULES: tuple[TournamentRule, ...] = (
    TournamentRule("fifa world cup qualification", 2.5),
    TournamentRule("qualification", 2.2),
    TournamentRule("fifa world cup", 4.0, final_tournament=True),
    TournamentRule("confederations cup", 3.5, final_tournament=True),
    TournamentRule("uefa euro", 3.5, final_tournament=True),
    TournamentRule("copa am", 3.5, final_tournament=True),
    TournamentRule("african cup", 3.0, final_tournament=True),
    TournamentRule("afc asian cup", 3.0, final_tournament=True),
    TournamentRule("gold cup", 2.8, final_tournament=True),
    TournamentRule("nations league", 3.0),
    TournamentRule("friendly", 1.0),
)
DEFAULT_IMPORTANCE = 2.0
WORLD_CUP = "FIFA World Cup"
WORLD_CUP_IMPORTANCE = 4.0

# A final tournament's group stage gives every team this many matches; a team's
# next match in the same tournament edition is treated as a knockout match.
GROUP_STAGE_MATCHES = 3
# Matches of one tournament more than this many days apart belong to different
# editions (handles editions that span New Year, e.g. AFCON 2025 in Dec-Jan).
EDITION_GAP_DAYS = 60

# --------------------------------------------------------------------------
# Elo (World Football Elo style)
# --------------------------------------------------------------------------
ELO_START = 1500.0
ELO_K = 40.0  # K for importance == DEFAULT_IMPORTANCE; scaled linearly with importance
ELO_HOME_ADV = 65.0  # rating points added to the home side in non-neutral matches

# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
FORM_WINDOWS = (5, 10)  # rolling form windows (matches)
MIN_TRAIN_YEAR = 1990  # Elo is computed from 1872, but only 1990+ rows are trained on
RECENCY_HALFLIFE_YEARS = 6.0  # sample-weight half-life, anchored at the training cutoff

# Defaults used when a team or venue is missing from the curated reference tables.
DEFAULT_ALTITUDE_M = 25.0
DEFAULT_TEMP_C = 22.0
DEFAULT_HUMIDITY = 60.0

# Columns produced by feature engineering, grouped so whole families can be
# switched on/off (`wcpred train --include/--exclude GROUP`).
_FORM_COLS = [
    f"{side}_{stat}{w}"
    for side in ("team", "opp")
    for w in FORM_WINDOWS
    for stat in ("gf", "ga", "ppg")
]
FEATURE_GROUPS: dict[str, list[str]] = {
    "elo": ["elo_team", "elo_opp", "elo_diff"],
    "form": [
        *_FORM_COLS,
        "team_days_since",
        "opp_days_since",
        "team_nprior",
        "opp_nprior",
    ],
    "venue": ["is_home", "is_neutral", "at_home_country"],
    "competition": ["importance", "is_knockout", "year_frac"],
    "confederation": ["same_confederation", "confed_team", "confed_opp"],
    "fifa_ranking": ["rank_pts_team", "rank_pts_opp", "rank_pts_diff"],
    "squad_age": ["team_avg_age", "opp_avg_age", "age_diff"],
    "altitude": ["alt_match", "team_home_alt", "alt_gain"],
    "climate": ["temp_match", "team_home_temp", "heat_gain", "humidity_match"],
    "market_value": ["mv_team_log", "mv_opp_log", "mv_diff_log"],
}
ALL_FEATURES: list[str] = [c for cols in FEATURE_GROUPS.values() for c in cols]

# market_value is off by default: the only data is a single June-2026 snapshot,
# so using it for historical matches leaks future information (see README).
DEFAULT_FEATURE_GROUPS: tuple[str, ...] = tuple(g for g in FEATURE_GROUPS if g != "market_value")


def feature_columns(groups: tuple[str, ...] | list[str]) -> list[str]:
    """Ordered feature columns for a set of group names."""
    unknown = set(groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"unknown feature group(s): {sorted(unknown)}")
    return [c for g in FEATURE_GROUPS if g in groups for c in FEATURE_GROUPS[g]]


# --------------------------------------------------------------------------
# XGBoost goals model
# --------------------------------------------------------------------------
XGB_PARAMS: dict[str, object] = {
    "objective": "count:poisson",
    "eval_metric": "poisson-nloglik",
    "eta": 0.03,
    "max_depth": 5,
    "min_child_weight": 8,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
    "reg_alpha": 0.5,
    "max_delta_step": 0.7,  # recommended for Poisson stability
    "tree_method": "hist",
    "seed": 42,
    "nthread": 4,  # fixed thread count keeps results identical across machines
}
XGB_MAX_ROUNDS = 1200
XGB_EARLY_STOPPING = 60
# The last N years before the training cutoff are held out to pick the number of
# boosting rounds and to fit the calibration (rho, totals factors).
INNER_VALID_YEARS = 3

# --------------------------------------------------------------------------
# Score distribution
# --------------------------------------------------------------------------
MAX_GOALS = 12  # score matrices cover 0..MAX_GOALS goals per side
DIXON_COLES_RHO = -0.05  # fallback when rho is not fitted
RHO_BOUNDS = (-0.3, 0.2)
EXTRA_TIME_FRACTION = 1.0 / 3.0  # 30 minutes of extra time ~ a third of a match's goals
PENALTY_HOME_WIN_PROB = 0.5

# --------------------------------------------------------------------------
# Evaluation and simulation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Fold:
    """Evaluation window; the model is trained only on matches before `start`."""

    name: str
    start: str
    end: str
    tournament: str | None = None  # restrict scoring to one competition


EVAL_FOLDS: tuple[Fold, ...] = (
    Fold("2020-21", "2020-01-01", "2022-01-01"),
    Fold("2022-23", "2022-01-01", "2024-01-01"),
    Fold("2024-26 (pre-WC)", "2024-01-01", "2026-06-11"),
    Fold("World Cup 2026", "2026-06-11", "2026-07-20", tournament=WORLD_CUP),
)
WC2026_START = pd.Timestamp("2026-06-11")
WC2026_KNOCKOUT_START = pd.Timestamp("2026-06-28")
WC2026_END = pd.Timestamp("2026-07-19")
DEFAULT_SIM_RUNS = 10_000
DEFAULT_SEED = 0
