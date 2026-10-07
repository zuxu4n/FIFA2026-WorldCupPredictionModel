# Data

Two kinds of data: the match results (downloaded, not committed) and small
reference tables (committed). Everything the model uses is listed here with its
source and how much to trust it.

## Match results (`data/raw/`, downloaded)

```bash
wcpred fetch-data              # pinned commit, SHA-256 verified
wcpred fetch-data --ref master # latest upstream data (results will differ from the README)
```

| File | Source | Contents |
|---|---|---|
| `results.csv` | [martj42/international_results](https://github.com/martj42/international_results) (CC0), commit `b7a3a8e` (2026-10-07) | 49,945 international matches, 1872-11-30 to 2026-10-06, 337 teams |
| `shootouts.csv` | same | penalty-shootout winners (used to lock drawn knockout matches) |

Schema of `results.csv` (rows with blank scores are treated as scheduled fixtures):

```
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2026-07-19,Spain,Argentina,1,0,FIFA World Cup,East Rutherford,United States,TRUE
```

Elo ratings are computed over all 49,945 matches; the model trains on the 32,821
played since 1990.

## Reference tables (`data/reference/`, committed)

| File | Rows | Source | Reliability |
|---|---:|---|---|
| `wc2026_groups_official.csv` | 48 | [openfootball/world-cup](https://github.com/openfootball/world-cup) `2026--usa/cup.txt` | exact |
| `wc2026_bracket.csv` | 32 | openfootball `2026--usa/cup_finals.txt`: match numbers, dates, venues, slot codes | exact (checked against the played knockout matches) |
| `wc2026_third_place_table.csv` | 495 | FIFA World Cup 2026 regulations, Annex C, via the Wikipedia template (`scripts/import_third_place_table.py`) | exact (validated against bracket eligibility and the real 2026 case) |
| `fifa_rankings.csv` | 67,883 | [Dato-Futbol/fifa-ranking](https://github.com/Dato-Futbol/fifa-ranking) (`scripts/import_fifa_rankings.py`) | real, but ends 2024-09-19: later matches reuse the last snapshot |
| `squad_ages.csv` | 537 | openfootball World Cup squad lists 1930-2026 (`scripts/import_squads.py`) | real; only used for World Cup matches of that year |
| `team_meta.csv` | 64 | hand-curated (`scripts/make_reference_data.py`) | confederation exact; home altitude approximate |
| `venues.csv` | 32 | hand-curated | approximate stadium altitudes |
| `team_climate.csv` | 64 | hand-curated | **approximate** warm-season temperature/humidity |
| `venue_climate.csv` | 16 | hand-curated | **approximate** June-July normals for the 2026 venues |
| `market_values.csv` | 59 | hand-transcribed from a June 2026 article citing Transfermarkt; non-finalists are rough estimates | **approximate, single snapshot; not used by default** |

### Known gaps

- `team_meta.csv` and the climate tables cover only 64 teams. The other teams in
  the results get an "unknown" confederation and default altitude/climate values.
- Venue lookups fall back from the exact (city, country) to the host country's
  own conditions, then the home team's, then a default. Most historical matches
  therefore get country-level approximations.
- Climate normals are applied regardless of season (a January friendly in Texas
  gets the June value).

### Why market value is off by default

There is only one snapshot (June 2026). Using it as a feature for a 1995 or 2021
match tells the model how strong a team *would become*, which is look-ahead
leakage. It also marks the 48 eventual qualifiers, since they are the teams with
values. The feature group is kept for experiments (`--include market_value`), and
its effect is reported in the ablation in [`docs/methodology.md`](../docs/methodology.md).

## Regenerating reference data

```bash
python scripts/make_reference_data.py            # curated tables (team_meta, venues, climate, market values)
python scripts/import_fifa_rankings.py PATH      # PATH: Dato-Futbol/fifa-ranking zip or csv
python scripts/import_squads.py PATH             # PATH: openfootball/world-cup zip or folder
python scripts/import_third_place_table.py       # downloads the Annex C table
```

All team names are accent-stripped on load (`Curaçao` and `Curacao` match).
