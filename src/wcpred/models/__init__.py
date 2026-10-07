"""Expected-goals model, calibration, and the score-distribution math."""

from wcpred.models.calibration import Calibration
from wcpred.models.goals import (
    GoalsModel,
    ModelNotFoundError,
    TrainingConfig,
    fit_goals_model,
    match_pairs,
)
from wcpred.models.scoreline import (
    advance_probability,
    outcome_probs,
    score_matrices,
    score_matrix,
    top_scorelines,
)

__all__ = [
    "Calibration",
    "GoalsModel",
    "ModelNotFoundError",
    "TrainingConfig",
    "advance_probability",
    "fit_goals_model",
    "match_pairs",
    "outcome_probs",
    "score_matrices",
    "score_matrix",
    "top_scorelines",
]
