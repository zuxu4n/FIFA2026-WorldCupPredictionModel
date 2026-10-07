"""Command-line interface: `wcpred <command>` (or `python -m wcpred <command>`)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence

import numpy as np
import pandas as pd

from wcpred import __version__, pipeline
from wcpred import config as C
from wcpred.data.download import fetch_results
from wcpred.data.results import DataError
from wcpred.evaluation.backtest import FoldResult
from wcpred.features.build import UnknownTeamError
from wcpred.match import MatchContext
from wcpred.models.goals import ModelNotFoundError, TrainingConfig


def _date(value: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value).normalize()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r} (use YYYY-MM-DD)") from exc


def _training_config(args: argparse.Namespace) -> TrainingConfig:
    groups = [g for g in C.DEFAULT_FEATURE_GROUPS if g not in args.exclude]
    groups += [g for g in args.include if g not in groups]
    return TrainingConfig(
        feature_groups=tuple(groups), calibrate_totals=args.totals_calibration
    )


# ---------------------------------------------------------------- commands


def cmd_fetch_data(args: argparse.Namespace) -> None:
    for path in fetch_results(ref=args.ref):
        print(f"wrote {path}")


def cmd_train(args: argparse.Namespace) -> None:
    model = pipeline.train(asof=args.asof, config=_training_config(args))
    meta = model.meta
    print(
        f"Trained on {meta['n_train_matches']:,} matches "
        f"({meta['n_train_rows']:,} team-match rows) through {meta['trained_through']}"
    )
    print(
        f"Boosting rounds: {meta['best_rounds']} (early stopping on "
        f"{meta['n_inner_valid_matches']:,} held-out matches since {meta['inner_valid_from']})"
    )
    print(f"Dixon-Coles rho: {model.calibration.rho:+.4f}")
    if model.calibration.totals:
        factors = ", ".join(f"{k} {v:.3f}" for k, v in sorted(model.calibration.totals.items()))
        print(f"Goal-total factors: {factors}")
    gain = {k: float(np.sum(v)) for k, v in model.booster.get_score(importance_type="gain").items()}
    print("\nTop features by gain:")
    for name, value in sorted(gain.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {name:<20} {value:8.1f}")
    print(f"\nSaved to {C.PATHS.model_dir(args.asof)}")


def _fold_table(results: Sequence[FoldResult]) -> str:
    header = (
        f"{'Fold':<18}{'Matches':>8}  {'Log loss':>8} {'Brier':>6} {'Acc':>6}"
        f"  | {'Elo-logit LL':>12} {'Acc':>6}  | {'Base-rate LL':>12}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        m, elo, base = r.model, r.baselines["elo_logit"], r.baselines["base_rate"]
        lines.append(
            f"{r.name:<18}{r.n_matches:>8,}  {m['log_loss']:>8.4f} {m['brier']:>6.3f} "
            f"{m['accuracy']:>6.1%}  | {elo['log_loss']:>12.4f} {elo['accuracy']:>6.1%}  "
            f"| {base['log_loss']:>12.4f}"
        )
    return "\n".join(lines)


def cmd_evaluate(args: argparse.Namespace) -> None:
    config = _training_config(args)
    results, detail = pipeline.evaluate(config)
    print(_fold_table(results))
    print(
        "\nLog loss: lower is better (uniform guess = 1.0986). "
        "Each fold's model is trained only on earlier matches."
    )
    C.PATHS.outputs_dir.mkdir(parents=True, exist_ok=True)
    out = C.PATHS.outputs_dir / "evaluation.json"
    payload = {
        "feature_groups": list(config.feature_groups),
        "calibrate_totals": config.calibrate_totals,
        "folds": [r.to_dict() for r in results],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    detail.to_csv(C.PATHS.outputs_dir / "evaluation_matches.csv", index=False)
    print(f"Saved {out} and per-match predictions to evaluation_matches.csv")


def cmd_predict(args: argparse.Namespace) -> None:
    predictor, _ = pipeline.load_predictor(args.asof)
    date = args.date or predictor.default_date()
    ctx = MatchContext(
        date=date,
        city=args.city,
        country=args.country,
        neutral=not args.host,
        tournament=args.tournament,
        knockout=args.knockout,
    )
    p = predictor.predict(args.home, args.away, ctx)
    venue = f"{ctx.city}, {ctx.country}{'' if args.host else ', neutral venue'}"
    print(f"{p.home} vs {p.away}  ({venue}; {ctx.tournament}, {date.date()})")
    print(f"Expected goals:  {p.home} {p.xg_home:.2f} - {p.xg_away:.2f} {p.away}\n")
    width = max(len(p.home), len(p.away)) + 5
    print(f"{p.home + ' win':<{width}} {p.p_home_win:6.1%}")
    print(f"{'Draw':<{width}} {p.p_draw:6.1%}")
    print(f"{p.away + ' win':<{width}} {p.p_away_win:6.1%}")
    if args.knockout:
        print(
            f"\nTo advance (incl. extra time and penalties): {p.home} "
            f"{p.p_home_advance:.1%}, {p.away} {1 - p.p_home_advance:.1%}"
        )
    print("\nMost likely scores:")
    for (i, j), prob in p.top_scorelines:
        print(f"  {i}-{j}  {prob:5.1%}")
    if args.explain:
        for team, drivers in predictor.explain(p.home, p.away, ctx).items():
            print(f"\nLargest effects on {team}'s expected goals (SHAP):")
            for feature, pct in drivers:
                print(f"  {feature:<20} {pct:+7.1%}")


def cmd_simulate(args: argparse.Namespace) -> None:
    start = time.perf_counter()
    result, predictor = pipeline.simulate(asof=args.asof, n_runs=args.runs, seed=args.seed)
    elapsed = time.perf_counter() - start
    table = result.table.copy()
    table.insert(2, "elo", [round(predictor.world.states[t].elo) for t in table["team"]])
    data_note = f"results before {args.asof.date()}" if args.asof else "all results"
    print(f"World Cup 2026: {result.n_runs:,} simulations (seed {result.seed}, {data_note})\n")
    shown = table.head(args.top).rename(
        columns={
            "team": "Team",
            "group": "Grp",
            "elo": "Elo",
            "P_win_group": "Win grp",
            "P_R32": "R32",
            "P_R16": "R16",
            "P_QF": "QF",
            "P_SF": "SF",
            "P_Final": "Final",
            "P_champion": "Champion",
        }
    )
    pct_cols = ["Win grp", "R32", "R16", "QF", "SF", "Final", "Champion"]
    print(shown.to_string(index=False, formatters={c: "{:6.1%}".format for c in pct_cols}))
    C.PATHS.outputs_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_asof_{args.asof.date()}" if args.asof else ""
    out = C.PATHS.outputs_dir / f"simulation{suffix}.csv"
    table.to_csv(out, index=False)
    print(f"\nSaved full table to {out} ({elapsed:.1f}s)")


# ------------------------------------------------------------------ parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wcpred",
        description="International football prediction and World Cup 2026 simulation.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="show progress logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch-data", help="download the pinned results dataset")
    p.add_argument(
        "--ref",
        default=C.RESULTS_REF,
        help="upstream git ref (default: pinned commit; 'master' for latest)",
    )
    p.set_defaults(func=cmd_fetch_data)

    def add_asof(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--asof",
            type=_date,
            default=None,
            metavar="DATE",
            help="use only results before DATE (e.g. 2026-06-11 = pre-tournament)",
        )

    def add_features(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--include",
            action="append",
            default=[],
            choices=list(C.FEATURE_GROUPS),
            metavar="GROUP",
            help="add a feature group (e.g. market_value)",
        )
        p.add_argument(
            "--exclude",
            action="append",
            default=[],
            choices=list(C.FEATURE_GROUPS),
            metavar="GROUP",
            help="drop a feature group",
        )
        p.add_argument(
            "--totals-calibration",
            action="store_true",
            help="enable the per-segment goal-total calibration (off by default)",
        )

    p = sub.add_parser("train", help="train the expected-goals model")
    add_asof(p)
    add_features(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="rolling-origin backtest with baselines")
    add_features(p)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", help="predict a single match")
    p.add_argument("--home", required=True)
    p.add_argument("--away", required=True)
    p.add_argument(
        "--date",
        type=_date,
        default=None,
        help="match date (default: day after the last result in the data)",
    )
    p.add_argument("--city", default="East Rutherford")
    p.add_argument("--country", default="United States")
    p.add_argument("--host", action="store_true", help="home side plays at home (not neutral)")
    p.add_argument("--tournament", default=C.WORLD_CUP)
    p.add_argument(
        "--knockout",
        action="store_true",
        help="knockout match: also report advancement after extra time/penalties",
    )
    p.add_argument("--explain", action="store_true", help="show the main feature effects")
    add_asof(p)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("simulate", help="Monte Carlo simulation of the 2026 World Cup")
    p.add_argument("--runs", type=int, default=C.DEFAULT_SIM_RUNS)
    p.add_argument("--seed", type=int, default=C.DEFAULT_SEED)
    p.add_argument("--top", type=int, default=16, help="rows to print")
    add_asof(p)
    p.set_defaults(func=cmd_simulate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        args.func(args)
    except ModelNotFoundError as exc:
        asof = f" --asof {args.asof.date()}" if getattr(args, "asof", None) else ""
        print(f"error: {exc}. Train it first: wcpred train{asof}", file=sys.stderr)
        return 1
    except (DataError, UnknownTeamError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
