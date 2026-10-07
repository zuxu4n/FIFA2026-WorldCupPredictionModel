"""Feature-group ablation: rerun the rolling-origin backtest with groups removed/added.

    python scripts/ablation.py

Writes outputs/ablation.json and prints a Markdown table (used in docs/methodology.md).
Every variant uses the same folds, data and seeds as `wcpred evaluate`.
"""

from __future__ import annotations

import json

import numpy as np

from wcpred import config as C
from wcpred.evaluation.backtest import run_backtest
from wcpred.models.goals import TrainingConfig
from wcpred.pipeline import load_world

DEFAULT = C.DEFAULT_FEATURE_GROUPS


def without(*groups: str) -> tuple[str, ...]:
    return tuple(g for g in DEFAULT if g not in groups)


VARIANTS: dict[str, TrainingConfig] = {
    "default": TrainingConfig(),
    "+ market_value (leaky)": TrainingConfig(feature_groups=(*DEFAULT, "market_value")),
    "- climate": TrainingConfig(feature_groups=without("climate")),
    "- altitude": TrainingConfig(feature_groups=without("altitude")),
    "- fifa_ranking": TrainingConfig(feature_groups=without("fifa_ranking")),
    "- squad_age": TrainingConfig(feature_groups=without("squad_age")),
    "- climate, altitude, squad_age": TrainingConfig(
        feature_groups=without("climate", "altitude", "squad_age")
    ),
    "+ totals calibration": TrainingConfig(calibrate_totals=True),
    "no Dixon-Coles (rho = 0)": TrainingConfig(fixed_rho=0.0),
}


def main() -> None:
    _, long, _ = load_world()
    rows = {}
    for name, config in VARIANTS.items():
        results, _ = run_backtest(long, C.EVAL_FOLDS, config)
        n = np.array([r.n_matches for r in results])
        ll = np.array([r.model["log_loss"] for r in results])
        acc = np.array([r.model["accuracy"] for r in results])
        score_ll = np.array([r.model["scoreline_log_loss"] for r in results])
        rows[name] = {
            "folds": {r.name: r.model for r in results},
            "pooled_log_loss": float((ll * n).sum() / n.sum()),
            "pooled_accuracy": float((acc * n).sum() / n.sum()),
            "pooled_scoreline_log_loss": float((score_ll * n).sum() / n.sum()),
        }
        print(f"{name:<34} pooled log-loss {rows[name]['pooled_log_loss']:.4f}", flush=True)

    C.PATHS.outputs_dir.mkdir(parents=True, exist_ok=True)
    (C.PATHS.outputs_dir / "ablation.json").write_text(json.dumps(rows, indent=2))

    fold_names = [f.name for f in C.EVAL_FOLDS]
    print("\n| Variant | W/D/L log loss | Scoreline log loss | " + " | ".join(fold_names) + " |")
    print("|---|---:|---:|" + "---:|" * len(fold_names))
    base = rows["default"]
    for name, row in rows.items():
        cells = []
        for key in ("pooled_log_loss", "pooled_scoreline_log_loss"):
            delta = "" if name == "default" else f" ({row[key] - base[key]:+.4f})"
            cells.append(f"{row[key]:.4f}{delta}")
        cells += [f"{row['folds'][f]['log_loss']:.4f}" for f in fold_names]
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
