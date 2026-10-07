"""Walk-forward evaluation and probability scoring rules."""

from wcpred.evaluation.backtest import FoldResult, evaluate_fold, run_backtest
from wcpred.evaluation.metrics import (
    accuracy,
    brier_score,
    log_loss,
    outcome_labels,
    score_all,
    scoreline_log_loss,
)

__all__ = [
    "FoldResult",
    "accuracy",
    "brier_score",
    "evaluate_fold",
    "log_loss",
    "outcome_labels",
    "run_backtest",
    "score_all",
    "scoreline_log_loss",
]
