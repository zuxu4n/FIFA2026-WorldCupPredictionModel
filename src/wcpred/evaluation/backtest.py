"""Rolling-origin (walk-forward) evaluation.

For each fold [start, end) a fresh model is trained with `fit_goals_model` on
matches before `start` only, then scored on every played match inside the
fold. Model parameters stay frozen within the fold, while pre-match features
(Elo, form) keep updating match by match, exactly as they would in live use.

Two simple baselines are fitted on the same training matches so the model's
numbers have a reference point:

* base_rate: historical home/draw/away frequencies (separately for neutral and
  non-neutral venues);
* elo_logit: multinomial logistic regression on the pre-match Elo difference
  and a home-venue flag.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from wcpred import config as C
from wcpred.evaluation.metrics import outcome_labels, score_all, scoreline_log_loss
from wcpred.models.goals import TrainingConfig, fit_goals_model, match_pairs, training_rows
from wcpred.models.scoreline import outcome_probs, score_matrices

log = logging.getLogger(__name__)


@dataclass
class FoldResult:
    name: str
    start: str
    end: str
    n_matches: int
    model: dict[str, float]
    baselines: dict[str, dict[str, float]]
    best_rounds: int
    rho: float
    totals_factors: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _base_rate_probs(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    labels = outcome_labels(train["gh"].to_numpy(), train["ga"].to_numpy())
    out = np.empty((len(test), 3))
    for neutral in (True, False):
        counts = np.bincount(labels[train["neutral"].to_numpy() == neutral], minlength=3) + 1.0
        out[test["neutral"].to_numpy() == neutral] = counts / counts.sum()  # +1 smoothing
    return out


def _elo_features(pairs: pd.DataFrame) -> np.ndarray:
    return np.column_stack([pairs["elo_diff"] / 100.0, (~pairs["neutral"]).astype(float)])


def _elo_logit_probs(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    labels = outcome_labels(train["gh"].to_numpy(), train["ga"].to_numpy())
    clf = LogisticRegression(max_iter=1000).fit(_elo_features(train), labels)
    return np.asarray(clf.predict_proba(_elo_features(test)))


def evaluate_fold(
    long: pd.DataFrame, fold: C.Fold, config: TrainingConfig | None = None
) -> tuple[FoldResult, pd.DataFrame]:
    """Train before the fold starts, score its matches. Returns (summary, per-match)."""
    config = config or TrainingConfig()
    name, start, end = fold.name, pd.Timestamp(fold.start), pd.Timestamp(fold.end)
    model = fit_goals_model(long, start, config)

    mask = long["played"] & (long["date"] >= start) & (long["date"] < end)
    if fold.tournament is not None:
        mask &= long["tournament"] == fold.tournament
    test_rows = long[mask]
    if test_rows.empty:
        raise ValueError(f"fold {name!r} has no played matches in [{start.date()}, {end.date()})")
    pairs = match_pairs(test_rows, model.predict_raw(test_rows))
    lh, la = model.calibration.scale(pairs["lam_h"], pairs["lam_a"], pairs["segment"])
    matrices = score_matrices(lh, la, model.calibration.rho)
    probs = outcome_probs(matrices)
    gh, ga = pairs["gh"].to_numpy(), pairs["ga"].to_numpy()
    labels = outcome_labels(gh, ga)

    metrics = score_all(probs, labels)
    metrics["scoreline_log_loss"] = scoreline_log_loss(matrices, gh, ga)
    metrics["goal_bias"] = float(((lh + la) - (gh + ga)).mean())
    metrics["totals_mae"] = float(np.abs((lh + la) - (gh + ga)).mean())

    train = match_pairs(training_rows(long, start, config.min_train_year))
    baselines = {
        "base_rate": score_all(_base_rate_probs(train, pairs), labels),
        "elo_logit": score_all(_elo_logit_probs(train, pairs), labels),
    }
    result = FoldResult(
        name=name,
        start=start.date().isoformat(),
        end=end.date().isoformat(),
        n_matches=len(pairs),
        model=metrics,
        baselines=baselines,
        best_rounds=int(model.meta["best_rounds"]),
        rho=model.calibration.rho,
        totals_factors=model.calibration.totals,
    )
    detail = pairs.loc[:, ["date", "home", "away", "gh", "ga", "segment"]].assign(
        xg_home=lh,
        xg_away=la,
        p_home=probs[:, 0],
        p_draw=probs[:, 1],
        p_away=probs[:, 2],
        fold=name,
    )
    log.info(
        "%s: n=%d log-loss %.4f accuracy %.3f",
        name,
        len(pairs),
        metrics["log_loss"],
        metrics["accuracy"],
    )
    return result, detail


def run_backtest(
    long: pd.DataFrame, folds: Sequence[C.Fold] = C.EVAL_FOLDS, config: TrainingConfig | None = None
) -> tuple[list[FoldResult], pd.DataFrame]:
    results, details = [], []
    for fold in folds:
        res, detail = evaluate_fold(long, fold, config)
        results.append(res)
        details.append(detail)
    return results, pd.concat(details, ignore_index=True)
