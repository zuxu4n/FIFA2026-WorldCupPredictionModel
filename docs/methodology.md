# Methodology

Details behind the README: how features are kept leak-free, how the model is
trained and evaluated, what each modelling component measurably contributes,
and what changed from the first version of this project.

All numbers come from `wcpred evaluate`, `python scripts/ablation.py` and
`python scripts/benchmark_simulation.py` on the pinned dataset (commit `b7a3a8e`
of martj42/international_results).

## 1. Leakage controls

| Risk | Control | Test |
|---|---|---|
| Features using the match's own or later results | One chronological pass; each row stores Elo/form *before* the match | `test_changing_a_result_never_changes_earlier_features` |
| Train/serve skew | Training rows and ad-hoc fixtures use the same `perspective_features` function | `test_prediction_rows_match_training_rows` (bit-for-bit equal) |
| Model trained on post-cutoff data | `fit_goals_model` only sees rows before the cutoff | `test_training_never_sees_results_after_the_cutoff` |
| Hyper-parameter selection on test data | Boosting rounds and rho are chosen on an inner window *before* the cutoff | used by every fold |
| Time-stamped reference data | FIFA rankings use an as-of join; squad ages only apply to that year's World Cup; `--asof` truncates both | `ReferenceData.load(asof=...)` |
| Single-snapshot squad values | `market_value` group off by default | ablation below |
| Wall-clock dependence | Sample weights are anchored at the training cutoff; prediction dates are explicit | `test_sample_weights_are_anchored_not_wall_clock` |

## 2. Training

For a cutoff date `T` (the start of an evaluation fold, the `--asof` date, or the
day after the last result):

1. Rows: both team perspectives of every played match from 1990 to `T`.
   Weight = tournament importance x 0.5^(age / 6 years), age measured from `T`.
2. Inner hold-out: matches in `[T - 3 years, T)`. XGBoost (`count:poisson`) is
   fitted on earlier rows with early stopping on this window, which picks the
   number of boosting rounds.
3. Dixon-Coles rho is fitted by maximum likelihood on the hold-out's actual
   scorelines, using the inner model's predictions (bounded to [-0.3, 0.2]).
4. The model is refitted on all rows before `T` with the chosen number of rounds.

Training on the full pinned dataset takes about 12 seconds on the machine in section 6.

## 3. From expected goals to probabilities

The model predicts each side's expected goals, lambda. The score matrix is a
product of two Poisson distributions (0-12 goals each) times the Dixon-Coles
factor tau, which changes only the 0-0, 1-0, 0-1 and 1-1 cells, then renormalised.
Summing the cells below, on and above the diagonal gives home win, draw and away
win. A knockout tie adds 30 minutes of extra time (lambda / 3 per side) and a
50/50 penalty shootout.

## 4. Evaluation

Rolling-origin folds; for each fold a fresh model is trained with the procedure
above using only matches before the fold starts. Within a fold, model parameters
are frozen but pre-match features (Elo, form) update after every match, as they
would when forecasting live.

Baselines are fitted on the same training matches:

- **base rate**: historical home/draw/away frequencies, separately for neutral
  and home venues;
- **Elo logistic**: multinomial logistic regression on the pre-match Elo
  difference and a home-venue flag.

| Fold | Matches | Model log loss | Brier | Accuracy | Elo-logit log loss | Elo-logit accuracy | Base-rate log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2020-21 | 1,471 | 0.8358 | 0.490 | 61.8% | 0.8422 | 62.3% | 1.0473 |
| 2022-23 | 2,034 | 0.8848 | 0.521 | 60.2% | 0.8890 | 59.9% | 1.0419 |
| 2024-01 to 2026-06-10 | 2,564 | 0.8642 | 0.508 | 59.8% | 0.8699 | 59.8% | 1.0504 |
| World Cup 2026 | 104 | 0.8418 | 0.490 | 66.3% | 0.8546 | 64.4% | 1.0554 |
| **Pooled** | **6,173** | **0.8639** | **0.508** | **60.5%** | **0.8693** | **60.5%** | **1.0470** |

Reading the table:

- The model has a lower (better) log loss and Brier score than the Elo baseline
  in every fold, but the margin is small (0.004-0.013), and pooled accuracy is
  the same. The gain is better-calibrated probabilities, not more correct picks.
- Pre-match Elo explains most of the predictable signal in international
  football. The XGBoost model adds form, venue and competition context on top.
- The World Cup 2026 fold has only 104 matches: its 66.3% accuracy is not
  statistically distinguishable from the other folds.

## 5. Ablation

Each row reruns the full four-fold backtest with one change. Pooled = weighted by
fold size (6,173 matches). Scoreline log loss is the negative log probability of
the exact final score.

| Variant | W/D/L log loss | Scoreline log loss | 2020-21 | 2022-23 | 2024-26 (pre-WC) | World Cup 2026 |
|---|---:|---:|---:|---:|---:|---:|
| default | 0.8639 | 2.8246 | 0.8358 | 0.8848 | 0.8642 | 0.8418 |
| + market_value (leaky) | 0.8622 (-0.0016) | 2.8224 (-0.0022) | 0.8310 | 0.8858 | 0.8627 | 0.8297 |
| - climate | 0.8647 (+0.0008) | 2.8251 (+0.0004) | 0.8365 | 0.8862 | 0.8650 | 0.8364 |
| - altitude | 0.8638 (-0.0001) | 2.8235 (-0.0011) | 0.8362 | 0.8839 | 0.8647 | 0.8377 |
| - fifa_ranking | 0.8638 (-0.0001) | 2.8243 (-0.0003) | 0.8319 | 0.8841 | 0.8662 | 0.8567 |
| - squad_age | 0.8645 (+0.0006) | 2.8253 (+0.0007) | 0.8360 | 0.8860 | 0.8643 | 0.8515 |
| - climate, altitude, squad_age | 0.8647 (+0.0009) | 2.8256 (+0.0009) | 0.8369 | 0.8839 | 0.8662 | 0.8454 |
| + totals calibration | 0.8640 (+0.0001) | 2.8262 (+0.0016) | 0.8360 | 0.8854 | 0.8638 | 0.8444 |
| no Dixon-Coles (rho = 0) | 0.8645 (+0.0006) | 2.8248 (+0.0002) | 0.8379 | 0.8849 | 0.8644 | 0.8451 |

What this shows:

- Differences of about 0.001 log loss are within fold-to-fold noise. The
  hand-curated groups (climate, altitude, squad age) are not clearly helping or
  hurting, so they stay on as documented, switchable groups.
- `market_value` looks slightly better, but it is the leaky feature described in
  `data/README.md`, so its gain cannot be trusted and it stays off.
- Per-segment totals calibration did not improve either metric and its fitted
  factors varied a lot between folds, so it is off by default (`--totals-calibration`).
- Dixon-Coles has a small positive effect on W/D/L log loss (0.0006). The fitted rho
  ranged from -0.004 to -0.074 across folds, so the size of the low-score
  correction itself varies with the period.

## 6. Simulation performance

`python scripts/benchmark_simulation.py` times `simulate_world_cup` on the
pre-tournament scenario (`--asof 2026-06-11`: 72 group matches and the full
knockout bracket to simulate), excluding data loading and feature building.
Best of 3 runs, Intel Core i9-12900KF, Windows 11, Python 3.14, NumPy 2.5,
XGBoost 3.4:

| Implementation | 1,000 runs | 10,000 runs |
|---|---:|---:|
| Original (`bracket.simulate_official`, commit `6443a2f`) | 31.2 s | not measured |
| Current (`simulation.simulate_world_cup`) | 0.78 s | 1.76 s |

The original was measured from a clean checkout of `6443a2f` with its own model
trained on the same data, timing only the `simulate_official` call. It simulated
one tournament at a time and called the model separately for each new knockout
pairing (3,458 single-fixture calls in a 1,000-run measurement). The current
version simulates all runs at once with NumPy arrays and makes one batched model
call for the group stage plus one per knockout round.

## 7. Changes from the first version

Bugs fixed (each has a regression test):

1. **Tournament weights.** Substring matching checked "uefa euro" before
   "qualification", so Euro qualifiers got importance 3.5 and AFCON/Asian Cup
   qualifiers 3.0, more than World Cup qualifiers (2.5). This inflated their
   Elo K-factor and training weight.
2. **Knockout flag.** "importance >= 3 and more than 14 days into the
   competition's calendar year" flagged 5,069 matches since 1990, about 4,500 of
   them qualifier or Nations League group games, and mislabelled editions that
   span New Year.
   It is now "both teams have already played three matches in this edition of a
   final tournament"; it marks exactly the 32 knockout matches of the 2026 World Cup.
3. **Wall-clock dependence.** Training weights and prediction-time features
   (days since last match, year, days into the competition) used today's date,
   so identical commands gave different results on different days.
4. **Look-ahead in validation.** The fold models reused the number of boosting
   rounds chosen on 2023+ data, and the fitted rho could be fitted on the
   evaluation period itself. Both are now chosen inside each fold's training window.
5. **Squad-age leakage.** A World Cup year's squad ages were applied to every
   match that year, including friendlies played before the squads were named.

Other changes:

- `market_value` disabled by default (single 2026 snapshot, see `data/README.md`).
- `days_into_comp` feature removed. It duplicated the knockout signal, and for a
  hypothetical fixture it had no meaningful value.
- The W/D/L recalibrator (logistic regression on outcome probabilities) was
  removed. It had disabled itself because it did not beat the raw probabilities.
- The randomised third-place slotting was replaced by FIFA's official 495-row
  allocation table, and the bracket is now a validated CSV instead of a
  regex-parsed, partly filled-in text file.
- Betting and live-tournament tooling (odds API, Kelly staking, weather
  forecasts, manual injury adjustments) was removed to keep the project focused
  on the forecasting pipeline.
