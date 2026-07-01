# Data — what's here and how to provide your own

The model needs four kinds of data. **Only `results.csv` is required**, and it is
fetched automatically. Everything else has working starter data baked in and can
be improved by replacing a CSV — nothing in the code changes.

| Data | File | Status | How to improve |
|------|------|--------|----------------|
| International match results (1872→now, incl. WC2026 so far) | `raw/results.csv` | ✅ auto-fetched, live | `python scripts/fetch_data.py` to refresh |
| Squad market values | `reference/market_values.csv` | ⚠️ **approximate** starter values | replace with Transfermarkt numbers (below) |
| FIFA rankings | `reference/fifa_rankings.csv` | ➖ optional, not present | add the file to enable rank features |
| Venue & team altitude | `reference/venues.csv`, `reference/team_meta.csv` | ✅ curated for WC2026 | refine elevations if you like |

---

## 1. `raw/results.csv`  (required — auto-fetched)

Source: [martj42/international_results](https://github.com/martj42/international_results),
updated continually (it already contains the in-progress World Cup — played
matches have scores, upcoming fixtures have blank scores).

Schema (don't change it — the loader expects these columns):

```
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2026-06-27,Panama,England,,,FIFA World Cup,East Rutherford,United States,TRUE
```

`home_score`/`away_score` blank = an upcoming fixture to predict. `neutral=TRUE`
means it's not played at the home team's ground (all WC2026 games are neutral,
but `country` still identifies host nations for the host-advantage feature).

**Offline?** Drop a CSV with this schema at `data/raw/results.csv` yourself.

## 2. `reference/market_values.csv`  (you should improve this)

This is the **single most valuable thing you can provide.** The committed file has
*approximate* squad market values; replace them with real
[Transfermarkt](https://www.transfermarkt.com/) "Market value of the squad"
figures (in € millions) for sharper predictions.

```
team,squad_value_eur_m,as_of
France,1300,2026-06
Panama,40,2026-06
```

Use the exact team names from `results.csv` (accents are handled automatically,
e.g. `Curacao` ≈ `Curaçao`). Teams you omit are simply treated as "unknown"
(the model handles missing values natively).

## 3. `reference/fifa_rankings.csv`  (optional — now populated)

Switches on the FIFA-ranking features (an as-of join on ranking points). Schema:

```
date,team,rank,total_points
2026-04-03,Argentina,1,1886.16
```

Populated from the [Dato-Futbol/fifa-ranking](https://github.com/Dato-Futbol/fifa-ranking)
historical dump (Dec 1992 → Sept 2024, 335 snapshots) via:

```
python scripts/import_fifa_rankings.py path/to/fifa-ranking-master.zip
```

The importer maps the ranking source's team spellings onto the match-dataset
spellings (IR Iran → Iran, USA → United States, Korea Republic → South Korea, …)
and derives a per-date rank. A point-in-time/as-of join is used, so more
snapshots = better; note this dump ends 2024-09, so recent matches reuse the
last snapshot as a slow-moving prior. **Honest note:** ranking points correlate
strongly with Elo, so this is ranked high in feature importance but adds little
incremental accuracy on top of Elo.

## 3b. `reference/squad_ages.csv`  (optional — populated)

Average squad age per `(tournament year, team)` for every World Cup 1930→2026,
imported from the [openfootball](https://github.com/openfootball/world-cup)
squad rosters:

```
python scripts/import_squads.py path/to/worldcup-master.zip
```

Joined to matches by tournament year (2014 ages for 2014 matches, 2026 for
2026), so it carries no future leakage. **Honest note:** only World Cup matches
get a value, so it's a weak, low-coverage signal — the model uses it at modest
importance but it barely moves validation.

## 4. `reference/venues.csv` & `reference/team_meta.csv`  (curated)

- `venues.csv` — `(city, country) → altitude_m`. All 16 WC2026 host venues plus
  famous high-altitude football cities (La Paz, Quito, Bogotá…). The altitude /
  acclimatisation feature uses the gap between a match's elevation and the
  visiting side's usual home elevation.
- `team_meta.csv` — `team → confederation, home_altitude_m`.

Regenerate or edit both with `python scripts/make_reference_data.py` (the numbers
live in that script, fully editable).
