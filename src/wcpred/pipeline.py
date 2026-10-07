"""End-to-end steps behind the CLI: train, evaluate, load a predictor, simulate.

`asof` runs every step as of a past date: results on or after it are hidden,
time-stamped reference data is truncated, and the model is trained only on
earlier matches. `asof=2026-06-11` reproduces the pre-tournament forecast.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from wcpred import config as C
from wcpred.data.download import sha256
from wcpred.data.reference import ReferenceData
from wcpred.data.results import as_of, load_results, load_shootouts
from wcpred.evaluation.backtest import FoldResult, run_backtest
from wcpred.features.build import FeatureWorld, build_features
from wcpred.models.goals import GoalsModel, TrainingConfig, fit_goals_model
from wcpred.predict import Predictor
from wcpred.simulation.bracket import TournamentFormat
from wcpred.simulation.tournament import SimulationResult, simulate_world_cup

log = logging.getLogger(__name__)


def _asof_str(asof: pd.Timestamp | None) -> str | None:
    return None if asof is None else asof.date().isoformat()


def load_world(asof: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame, FeatureWorld]:
    """(as-of results, long feature table, feature world)."""
    results = as_of(load_results(), asof)
    long, world = build_features(results, ReferenceData.load(asof=asof))
    return results, long, world


def train(
    asof: pd.Timestamp | None = None, config: TrainingConfig | None = None, save: bool = True
) -> GoalsModel:
    """Train on everything played before `asof` (default: all played matches)."""
    _, long, world = load_world(asof)
    if world.last_played is None:
        raise ValueError("no played matches to train on")
    cutoff = asof if asof is not None else world.last_played + pd.Timedelta(days=1)
    model = fit_goals_model(long, cutoff, config)
    model.meta["asof"] = _asof_str(asof)
    model.meta["results_sha256"] = sha256(C.PATHS.results_csv.read_bytes())
    if save:
        model.save(C.PATHS.model_dir(asof))
        log.info("saved model to %s", C.PATHS.model_dir(asof))
    return model


def evaluate(
    config: TrainingConfig | None = None, folds: Sequence[C.Fold] = C.EVAL_FOLDS
) -> tuple[list[FoldResult], pd.DataFrame]:
    _, long, _ = load_world()
    return run_backtest(long, folds, config)


def load_predictor(asof: pd.Timestamp | None = None) -> tuple[Predictor, pd.DataFrame]:
    """The saved model for `asof` plus features rebuilt from the same data."""
    model = GoalsModel.load(C.PATHS.model_dir(asof))
    if model.meta.get("asof") != _asof_str(asof):
        raise ValueError(
            f"model in {C.PATHS.model_dir(asof)} was trained with "
            f"asof={model.meta.get('asof')}; retrain it"
        )
    results, _, world = load_world(asof)
    return Predictor(model, world), results


def simulate(
    asof: pd.Timestamp | None = None, n_runs: int = C.DEFAULT_SIM_RUNS, seed: int = C.DEFAULT_SEED
) -> tuple[SimulationResult, Predictor]:
    predictor, results = load_predictor(asof)
    result = simulate_world_cup(
        results,
        TournamentFormat.load(),
        predictor,
        n_runs=n_runs,
        seed=seed,
        shootouts=load_shootouts(asof=asof),
    )
    return result, predictor
