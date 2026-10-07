"""XGBoost Poisson model of the goals a team scores, with training and persistence.

Training is strictly chronological. Given a cutoff date, only matches before it
are used, and the last INNER_VALID_YEARS of those are held out to

  * choose the number of boosting rounds (early stopping), and
  * fit the calibration (totals factors, Dixon-Coles rho) on predictions the
    model has not been trained on.

The model is then refitted on every match before the cutoff with the chosen
number of rounds. `wcpred train` and every evaluation fold use this same
function, so the evaluated pipeline is the deployed pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from wcpred import __version__
from wcpred import config as C
from wcpred.models.calibration import Calibration, fit_rho, fit_totals

log = logging.getLogger(__name__)

MODEL_FILE = "booster.json"
META_FILE = "meta.json"


class ModelNotFoundError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class TrainingConfig:
    feature_groups: tuple[str, ...] = C.DEFAULT_FEATURE_GROUPS
    params: dict[str, Any] = field(default_factory=lambda: dict(C.XGB_PARAMS))
    max_rounds: int = C.XGB_MAX_ROUNDS
    early_stopping_rounds: int = C.XGB_EARLY_STOPPING
    inner_valid_years: int = C.INNER_VALID_YEARS
    min_train_year: int = C.MIN_TRAIN_YEAR
    halflife_years: float = C.RECENCY_HALFLIFE_YEARS
    # Off by default: in the rolling-origin backtest it did not improve W/D/L or
    # scoreline log loss, and the fitted factors were unstable between folds.
    calibrate_totals: bool = False
    fixed_rho: float | None = None  # None: fit rho; 0.0: plain independent Poisson

    @property
    def feature_cols(self) -> list[str]:
        return C.feature_columns(self.feature_groups)


def sample_weights(
    dates: pd.Series, importance: pd.Series, anchor: pd.Timestamp, halflife_years: float
) -> np.ndarray:
    """Match importance x exponential recency decay, measured back from `anchor`.

    Anchoring at the training cutoff (not the wall clock) keeps training
    deterministic: the same data and cutoff always give the same weights.
    """
    age_years = (anchor - dates).dt.days.to_numpy(dtype=float) / 365.25
    return importance.to_numpy(dtype=float) * np.power(0.5, age_years / halflife_years)


def match_pairs(rows: pd.DataFrame, lam: np.ndarray | None = None) -> pd.DataFrame:
    """Join the two perspectives of each match: one row with lam_h, lam_a, gh, ga."""
    df = rows.assign(lam=np.nan if lam is None else lam)
    home = df[df["is_home_persp"] == 1].set_index("match_id")
    away = df[df["is_home_persp"] == 0].set_index("match_id")
    common = home.index.intersection(away.index)
    home, away = home.loc[common], away.loc[common]
    return pd.DataFrame(
        {
            "date": home["date"],
            "home": home["team"],
            "away": home["opp"],
            "lam_h": home["lam"].to_numpy(dtype=float),
            "lam_a": away["lam"].to_numpy(dtype=float),
            "gh": home["goals"].to_numpy(dtype=float),
            "ga": away["goals"].to_numpy(dtype=float),
            "neutral": home["is_neutral"].to_numpy(dtype=float) == 1.0,
            "elo_diff": home["elo_diff"].to_numpy(dtype=float),
            "segment": home["segment"],
        },
        index=common,
    )


@dataclass
class GoalsModel:
    booster: xgb.Booster
    feature_cols: list[str]
    calibration: Calibration
    meta: dict[str, Any]

    def predict_raw(self, features: pd.DataFrame) -> np.ndarray:
        """Uncalibrated expected goals for each row of a feature table."""
        dm = xgb.DMatrix(
            features[self.feature_cols].to_numpy(dtype=float), feature_names=self.feature_cols
        )
        return np.asarray(self.booster.predict(dm), dtype=float)

    def contributions(self, features: pd.DataFrame) -> pd.DataFrame:
        """Per-feature SHAP contributions to log(expected goals), one row per input row."""
        dm = xgb.DMatrix(
            features[self.feature_cols].to_numpy(dtype=float), feature_names=self.feature_cols
        )
        contrib = self.booster.predict(dm, pred_contribs=True)
        return pd.DataFrame(contrib[:, :-1], columns=self.feature_cols)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(directory / MODEL_FILE)
        meta = {
            **self.meta,
            "feature_cols": self.feature_cols,
            "calibration": self.calibration.to_dict(),
        }
        (directory / META_FILE).write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> GoalsModel:
        if not (directory / MODEL_FILE).exists() or not (directory / META_FILE).exists():
            raise ModelNotFoundError(f"no trained model in {directory}")
        meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        unknown = set(meta["feature_cols"]) - set(C.ALL_FEATURES)
        if unknown:
            raise ValueError(
                f"model uses features this version does not build: {unknown}; "
                "retrain with `wcpred train`"
            )
        booster = xgb.Booster()
        booster.load_model(directory / MODEL_FILE)
        return cls(
            booster=booster,
            feature_cols=list(meta["feature_cols"]),
            calibration=Calibration.from_dict(meta["calibration"]),
            meta=meta,
        )


def _dmatrix(
    rows: pd.DataFrame, cols: list[str], anchor: pd.Timestamp, config: TrainingConfig
) -> xgb.DMatrix:
    weights = sample_weights(rows["date"], rows["importance"], anchor, config.halflife_years)
    return xgb.DMatrix(
        rows[cols].to_numpy(dtype=float),
        label=rows["goals"].to_numpy(dtype=float),
        weight=weights,
        feature_names=cols,
    )


def training_rows(long: pd.DataFrame, cutoff: pd.Timestamp, min_year: int) -> pd.DataFrame:
    """Played perspective rows strictly before `cutoff`."""
    mask = long["played"] & (long["date"] < cutoff) & (long["date"].dt.year >= min_year)
    return long[mask]


def fit_goals_model(
    long: pd.DataFrame, cutoff: pd.Timestamp, config: TrainingConfig | None = None
) -> GoalsModel:
    """Train on all played matches before `cutoff` (see module docstring)."""
    config = config or TrainingConfig()
    cols = config.feature_cols
    rows = training_rows(long, cutoff, config.min_train_year)
    inner_cutoff = cutoff - pd.DateOffset(years=config.inner_valid_years)
    fit_rows = rows[rows["date"] < inner_cutoff]
    valid_rows = rows[rows["date"] >= inner_cutoff]
    if fit_rows.empty or valid_rows.empty:
        raise ValueError(f"not enough data before {cutoff.date()} to train and validate")
    log.info(
        "training on %s rows before %s (inner validation: %s rows from %s)",
        f"{len(rows):,}",
        cutoff.date(),
        f"{len(valid_rows):,}",
        inner_cutoff.date(),
    )

    # 1) choose the number of rounds on the inner hold-out
    dvalid = _dmatrix(valid_rows, cols, cutoff, config)
    inner = xgb.train(
        config.params,
        _dmatrix(fit_rows, cols, inner_cutoff, config),
        num_boost_round=config.max_rounds,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=config.early_stopping_rounds,
        verbose_eval=False,
    )
    best_rounds = inner.best_iteration + 1

    # 2) calibrate on the inner model's out-of-sample predictions
    lam = inner.predict(dvalid, iteration_range=(0, best_rounds))
    pairs = match_pairs(valid_rows, lam)
    lh, la = pairs["lam_h"].to_numpy(), pairs["lam_a"].to_numpy()
    gh, ga = pairs["gh"].to_numpy(), pairs["ga"].to_numpy()
    segments = pairs["segment"].to_numpy()
    calibration = Calibration()
    if config.calibrate_totals:
        calibration.totals = fit_totals(lh, la, gh, ga, segments)
        lh, la = calibration.scale(lh, la, segments)
    calibration.rho = fit_rho(lh, la, gh, ga) if config.fixed_rho is None else config.fixed_rho
    log.info(
        "best rounds %d, rho %.4f, totals %s", best_rounds, calibration.rho, calibration.totals
    )

    # 3) refit on everything before the cutoff
    booster = xgb.train(
        config.params, _dmatrix(rows, cols, cutoff, config), num_boost_round=best_rounds
    )
    meta = {
        "wcpred_version": __version__,
        "cutoff": cutoff.date().isoformat(),
        "trained_through": rows["date"].max().date().isoformat(),
        "n_train_rows": len(rows),
        "n_train_matches": int(rows["match_id"].nunique()),
        "best_rounds": best_rounds,
        "inner_valid_from": inner_cutoff.date().isoformat(),
        "n_inner_valid_matches": len(pairs),
        "config": {**asdict(config), "feature_groups": list(config.feature_groups)},
    }
    return GoalsModel(booster=booster, feature_cols=cols, calibration=calibration, meta=meta)
