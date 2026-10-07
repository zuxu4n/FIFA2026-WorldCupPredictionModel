"""Scoring rules for home-win / draw / away-win probability forecasts."""

from __future__ import annotations

import numpy as np

HOME, DRAW, AWAY = 0, 1, 2
_EPS = 1e-12


def outcome_labels(goals_home: np.ndarray, goals_away: np.ndarray) -> np.ndarray:
    """0 = home win, 1 = draw, 2 = away win."""
    return np.where(goals_home > goals_away, HOME, np.where(goals_home == goals_away, DRAW, AWAY))


def log_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log probability of the observed outcome (uniform guess = ln 3 = 1.099)."""
    p = np.clip(probs[np.arange(len(labels)), labels], _EPS, 1.0)
    return float(-np.log(p).mean())


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error against the one-hot outcome, summed over the 3 classes."""
    return float(((probs - np.eye(3)[labels]) ** 2).sum(axis=1).mean())


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    """Share of matches where the most likely outcome happened."""
    return float((probs.argmax(axis=1) == labels).mean())


def scoreline_log_loss(
    matrices: np.ndarray, goals_home: np.ndarray, goals_away: np.ndarray
) -> float:
    """Mean negative log probability of the exact final score, from (n, G, G) matrices."""
    g = matrices.shape[1] - 1
    gh = np.minimum(goals_home.astype(int), g)
    ga = np.minimum(goals_away.astype(int), g)
    p = np.clip(matrices[np.arange(len(gh)), gh, ga], _EPS, 1.0)
    return float(-np.log(p).mean())


def score_all(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "log_loss": log_loss(probs, labels),
        "brier": brier_score(probs, labels),
        "accuracy": accuracy(probs, labels),
    }
