# World Cup Predictor: XGBoost Poisson expected goals → W/D/L odds

Predicts international football matches by modelling **expected goals for each
side** with gradient-boosted trees (`xgboost`, `count:poisson` objective), then
converting the two expected-goal rates (λ_home, λ_away) into **win / draw / loss**
probabilities with a Dixon-Coles-adjusted bivariate Poisson. A Monte-Carlo engine
rolls the rest of the 2026 World Cup forward — along the **official knockout
bracket** (real Round-of-32 slot map + venues) — for advancement and title odds.

Built and validated against **live data** — the 2026 World Cup is mid-tournament,
so the model trains on the 60 group games already played and predicts the rest.

```
results + Elo + form + FIFA rank + market value + host + altitude
        │
        ▼
   XGBoost  ──►  λ_home , λ_away   (expected goals, count:poisson)
        │
        ▼
   Dixon-Coles bivariate Poisson  ──►  P(win) / P(draw) / P(loss), scorelines
        │
        ▼
   Monte-Carlo  ──►  group advancement & title odds
```

## What feeds the model

| Signal | Source | Feature(s) |
|---|---|---|
| Previous international results | live dataset (1872→now) | **Elo** (pre-match, goal-diff & importance weighted), rolling **form** (GF/GA/PPG over last 5 & 10) |
| Results from the World Cup so far | same dataset | folded straight into Elo/form (and weighted up) |
| FIFA rankings | optional CSV | as-of ranking points (off by default — Elo covers most of it) |
| Squad market values | Transfermarkt CSV | log squad value, team vs opponent gap |
| Squad age | openfootball rosters (all WCs) | avg squad age team/opp/diff, joined per tournament year (leak-free) |
| Home / host | dataset `neutral` + `country` | `is_home`, `is_neutral`, `at_home_country` (host-nation bump even in "neutral" WC games) |
| Altitude | curated venue/team elevations | match altitude, team's home altitude, **altitude gain** (acclimatisation) |
| Heat & humidity | curated team/venue climate | match temp/humidity, **heat gain** (match temp vs team's home climate) |
| Match importance | `tournament` column | friendly→WC weighting (feature **and** sample weight) |
| Confederation | curated | team/opponent confederation, same-confederation flag |

Recency is handled with an exponential sample-weight half-life so recent, important
matches matter most. All time-varying features use **pre-match** snapshots (no leakage).

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use source .venv/bin/activate on *nix
pip install -e .

python scripts/make_reference_data.py   # write reference CSVs (once)
python scripts/fetch_data.py            # download live results (refresh anytime)
python scripts/import_bracket.py PATH   # official 2026 bracket (openfootball zip)
python scripts/import_squads.py PATH    # squad ages, all WCs (openfootball zip)
python scripts/import_fifa_rankings.py PATH   # optional FIFA-ranking history
python scripts/train.py                 # train + save model (~30s)

python scripts/predict_match.py --home Spain --away France
python scripts/predict_fixtures.py      # all upcoming fixtures -> outputs/
python scripts/simulate_tournament.py --sims 10000   # title odds -> outputs/
```

## Usage examples

```bash
# A neutral knockout tie, with extra-time/penalty advancement
python scripts/predict_match.py --home Argentina --away England --knockout

# Host + altitude effect (Mexico at the Azteca, ~2240 m)
python scripts/predict_match.py --home Mexico --away Brazil \
    --city "Mexico City" --country Mexico --host
```

`predict_match.py` prints expected goals, W/D/L, the five most likely scorelines,
and (with `--knockout`) advancement probability.

## How outcomes are computed

For a fixture the model is run from both perspectives to get `λ_home` and `λ_away`.
The joint scoreline distribution is `P(i,j) = Poisson(i;λ_home)·Poisson(j;λ_away)`
with the **Dixon-Coles** low-score correction (parameter `ρ`) that fixes the
0-0/1-0/0-1/1-1 cells independent Poisson gets wrong. Summing the upper/lower
triangle and diagonal gives win/draw/loss. Knockout advancement adds a 30-minute
extra-time matrix and a penalty coin-flip.

## Calibration stack

Fitted on out-of-sample history (`python scripts/calibrate.py`), applied
automatically at prediction time:

- **Lambda (totals) shrinkage** per match segment (knockout / tournament group /
  qualifier / other) — corrects the systematic goal over-prediction a raw
  Poisson model shows in tournament football (`models/totals_calibrator.json`).
- **Fitted Dixon-Coles rho** (profile likelihood) instead of a hardcoded value.
- **W/D/L recalibration is self-disabling**: it is only deployed if it beats the
  raw pipeline on cross-validated log-loss (currently it does not — raw wins).
- A **knockout-stage feature** (`is_knockout`, `days_into_comp`) lets the model
  itself learn that knockout games are lower-scoring.

## Validation

Rolling-origin folds (`python scripts/validate.py`), each trained strictly on
earlier data:

| fold | n | W/D/L log-loss | accuracy |
|---|---|---|---|
| 2020–21 | 1,462 | 0.833 | 62.1% |
| 2022–23 | 2,024 | 0.883 | 60.3% |
| 2024–26 | 2,544 | 0.862 | 60.5% |
| WC2026 so far | 79 | 0.879 | **63.3%** |

(random log-loss = 1.099). `scripts/backtest.py` reproduces the WC2026 fold;
`scripts/backtest_totals.py` reports the goal-total bias per segment.

## Betting tooling (informational — no proven edge vs sharp books)

- `scripts/fetch_odds.py` — live odds (The Odds API), de-vig, EV/Kelly per
  selection, `--blend 0.5` for market-shrunk probabilities.
- `scripts/predict_props.py` — full derived board (result, advance, totals
  ladder, team totals, BTTS, handicaps, clean sheets) with `--et` for
  incl-extra-time markets and `--odds file.csv` for edge columns.
- `scripts/predict_match.py --why` — SHAP breakdown of what drove each side's
  expected goals.
- `scripts/fetch_weather.py` — real kickoff-day forecasts (Open-Meteo) override
  climate normals for upcoming fixtures.
- `data/reference/injuries.csv` — list squad value out per team (injuries /
  suspensions); applied as a market-value reduction at predict time.

## Project layout

```
src/wcpred/
  config.py      paths, hyper-parameters, data URLs
  data.py        load/clean results + reference tables (accent-robust joins)
  elo.py         chronological World-Football Elo (pre-match snapshots)
  features.py    leak-free long-format feature matrix + ad-hoc matchup featuriser
  model.py       train/load XGBoost Poisson, validation report, importances
  predict.py     λ → W/D/L (Dixon-Coles), scorelines, knockout advancement
  simulate.py    reconstruct groups, Monte-Carlo group + strength-seeded knockout
  bracket.py     parse official 2026 bracket, venue-aware knockout along real paths
scripts/         fetch_data / import_bracket / import_fifa_rankings / train /
                 predict_match / predict_fixtures / simulate_tournament
tests/           probability-math + feature-schema tests  (pytest)
data/            raw/ (results) + reference/ (altitude, market values, …)  — see data/README.md
```

## Limitations & honest caveats

- **Squad market values are approximate placeholders** — replacing them with real
  Transfermarkt figures is the biggest easy accuracy win (see `data/README.md`).
- The knockout simulator follows the **official 2026 bracket** (slot map +
  venues, imported via `scripts/import_bracket.py` from the openfootball
  dataset). Best-third placement uses a bipartite match over each slot's
  eligible-group set — a faithful relaxation of FIFA's official table, not the
  exact lookup. A `--seeded` fallback (strength-seeded bracket) lives in
  `simulate.py`.
- No player-level data (injuries, lineups, suspensions) — team-level only.
- Altitude is curated for the venues that matter (WC2026 hosts + famous high
  cities); other historical venues fall back to country/sea-level defaults.

## Want it sharper?

See **`data/README.md`** for exactly what data to drop in (market values, FIFA
rankings) and the expected CSV schemas. The pipeline reads whatever you provide
and ignores what you don't.
