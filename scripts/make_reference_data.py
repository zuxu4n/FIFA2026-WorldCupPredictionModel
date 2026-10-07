"""Generate the reference CSVs used as static features.

Run:  python scripts/make_reference_data.py

These files are *editable starter data*. The numbers are approximate, public
ballpark values curated by hand so the pipeline runs end-to-end out of the box.
Replace any of them with better sources whenever you have them (see data/README.md):

  - data/reference/team_meta.csv      team -> confederation, typical home-venue altitude (m)
  - data/reference/venues.csv         (city, country) -> venue altitude (m)
  - data/reference/market_values.csv  team -> squad market value (EUR millions)  [APPROXIMATE]

Altitudes are the elevation of each team's *usual home match venue* (what the
squad is acclimatised to), which is what drives the altitude/acclimatisation
effect — e.g. Bolivia (La Paz 3640 m), Ecuador (Quito 2850 m), Mexico
(Mexico City 2240 m).
"""

from __future__ import annotations

import csv
import os

from wcpred.config import PATHS

REF = str(PATHS.reference_dir)

# --- team -> (confederation, typical home-venue altitude in metres) --------
# Altitudes are approximate elevations of the country's usual home venue.
TEAM_META: dict[str, tuple[str, int]] = {
    # CONMEBOL
    "Argentina": ("CONMEBOL", 25),
    "Bolivia": ("CONMEBOL", 3640),
    "Brazil": ("CONMEBOL", 700),
    "Chile": ("CONMEBOL", 570),
    "Colombia": ("CONMEBOL", 18),
    "Ecuador": ("CONMEBOL", 2850),
    "Paraguay": ("CONMEBOL", 130),
    "Peru": ("CONMEBOL", 150),
    "Uruguay": ("CONMEBOL", 40),
    "Venezuela": ("CONMEBOL", 900),
    # CONCACAF
    "Canada": ("CONCACAF", 80),
    "Mexico": ("CONCACAF", 2240),
    "United States": ("CONCACAF", 150),
    "Panama": ("CONCACAF", 10),
    "Haiti": ("CONCACAF", 30),
    "Curacao": ("CONCACAF", 5),
    "Costa Rica": ("CONCACAF", 1170),
    "Jamaica": ("CONCACAF", 10),
    "Honduras": ("CONCACAF", 990),
    # UEFA
    "Austria": ("UEFA", 190),
    "Belgium": ("UEFA", 30),
    "Bosnia and Herzegovina": ("UEFA", 510),
    "Croatia": ("UEFA", 120),
    "Czech Republic": ("UEFA", 200),
    "England": ("UEFA", 30),
    "France": ("UEFA", 35),
    "Germany": ("UEFA", 150),
    "Netherlands": ("UEFA", 5),
    "Norway": ("UEFA", 20),
    "Portugal": ("UEFA", 100),
    "Scotland": ("UEFA", 50),
    "Spain": ("UEFA", 660),
    "Sweden": ("UEFA", 20),
    "Switzerland": ("UEFA", 430),
    "Turkey": ("UEFA", 120),
    "Italy": ("UEFA", 90),
    "Poland": ("UEFA", 110),
    "Denmark": ("UEFA", 10),
    "Serbia": ("UEFA", 120),
    "Wales": ("UEFA", 30),
    "Ukraine": ("UEFA", 180),
    # CAF
    "Algeria": ("CAF", 25),
    "Cape Verde": ("CAF", 30),
    "DR Congo": ("CAF", 280),
    "Egypt": ("CAF", 70),
    "Ghana": ("CAF", 60),
    "Ivory Coast": ("CAF", 50),
    "Morocco": ("CAF", 250),
    "Senegal": ("CAF", 20),
    "South Africa": ("CAF", 1750),
    "Tunisia": ("CAF", 10),
    "Nigeria": ("CAF", 480),
    "Cameroon": ("CAF", 700),
    # AFC
    "Australia": ("AFC", 20),
    "Iran": ("AFC", 1200),
    "Iraq": ("AFC", 34),
    "Japan": ("AFC", 40),
    "Jordan": ("AFC", 770),
    "Qatar": ("AFC", 10),
    "Saudi Arabia": ("AFC", 610),
    "South Korea": ("AFC", 40),
    "Uzbekistan": ("AFC", 450),
    "China PR": ("AFC", 50),
    # OFC
    "New Zealand": ("OFC", 20),
}

# --- WC2026 host venues -> altitude (m) ------------------------------------
VENUES: dict[tuple[str, str], int] = {
    ("Arlington", "United States"): 184,
    ("Atlanta", "United States"): 320,
    ("East Rutherford", "United States"): 3,
    ("Foxborough", "United States"): 70,
    ("Houston", "United States"): 24,
    ("Inglewood", "United States"): 30,
    ("Kansas City", "United States"): 270,
    ("Miami Gardens", "United States"): 3,
    ("Philadelphia", "United States"): 12,
    ("Santa Clara", "United States"): 9,
    ("Seattle", "United States"): 5,
    ("Toronto", "Canada"): 80,
    ("Vancouver", "Canada"): 5,
    ("Mexico City", "Mexico"): 2240,
    ("Guadalupe", "Mexico"): 500,  # Monterrey metro (Estadio BBVA)
    ("Zapopan", "Mexico"): 1566,  # Guadalajara metro (Estadio Akron)
    # A few famous high-altitude football cities, so the altitude effect is
    # learned from history too (used by the city->altitude resolver).
    ("La Paz", "Bolivia"): 3640,
    ("El Alto", "Bolivia"): 4090,
    ("Quito", "Ecuador"): 2850,
    ("Cuenca", "Ecuador"): 2560,
    ("Bogota", "Colombia"): 2640,
    ("Barranquilla", "Colombia"): 18,
    ("Cusco", "Peru"): 3400,
    ("Lima", "Peru"): 150,
    ("Addis Ababa", "Ethiopia"): 2355,
    ("Johannesburg", "South Africa"): 1753,
    ("Toluca", "Mexico"): 2660,
    ("Pachuca", "Mexico"): 2400,
    ("San Jose", "Costa Rica"): 1170,
    ("Tehran", "Iran"): 1200,
    ("Madrid", "Spain"): 660,
    ("Sucre", "Bolivia"): 2810,
}

# --- WC2026 squad market values (EUR millions) -----------------------------
# Finalists: Transfermarkt squad values as reported in a June 2026 planetfootball.com
# article ("World Cup 2026: all 48 squads ranked by market value"), transcribed by
# hand and not independently verified. Non-qualified teams below the divider are
# rough estimates. This is a single snapshot, so the market_value feature group is
# OFF by default (applying 2026 values to past matches leaks future information).
MARKET_VALUES: dict[str, int] = {
    # --- 48 WC2026 finalists (reported Transfermarkt values, EUR millions) ---
    "France": 1520,
    "England": 1360,
    "Spain": 1220,
    "Portugal": 1010,
    "Germany": 947,
    "Brazil": 928,
    "Argentina": 808,
    "Netherlands": 754,
    "Norway": 590,
    "Belgium": 548,
    "Ivory Coast": 522,
    "Senegal": 478,
    "Turkey": 474,
    "Morocco": 448,
    "Sweden": 406,
    "Croatia": 387,
    "United States": 386,
    "Ecuador": 369,
    "Uruguay": 359,
    "Switzerland": 333,
    "Colombia": 302,
    "Japan": 271,
    "Algeria": 257,
    "Austria": 245,
    "Ghana": 235,
    "Canada": 199,
    "Mexico": 192,
    "Czech Republic": 188,
    "Scotland": 170,
    "Paraguay": 154,
    "Bosnia and Herzegovina": 146,
    "DR Congo": 144,
    "South Korea": 139,
    "Egypt": 116,
    "Uzbekistan": 85,
    "Australia": 77,
    "Tunisia": 70,
    "Haiti": 56,
    "Cape Verde": 49,
    "South Africa": 49,
    "Saudi Arabia": 41,
    "Panama": 35,
    "New Zealand": 34,
    "Iran": 32,
    "Curacao": 26,
    "Iraq": 21,
    "Jordan": 20,
    "Qatar": 20,
    # --- non-qualified teams (approximate, for historical coverage) ---
    "Italy": 700,
    "Denmark": 320,
    "Serbia": 300,
    "Nigeria": 280,
    "Poland": 320,
    "Cameroon": 280,
    "Venezuela": 120,
    "Bolivia": 25,
    "Peru": 90,
    "Chile": 150,
    "Costa Rica": 40,
}


# --- team home climate (typical match-day conditions) ---------------------
# APPROXIMATE warm-season temperature (C) and relative humidity (%) of each
# team's home region — their acclimatisation baseline. Also used as the climate
# of a match *played* in that country (team name ~= country), with WC2026 venue
# overrides below. Note high-altitude "home" cities stay mild (Quito, La Paz,
# Mexico City, Johannesburg) even near the equator.
TEAM_CLIMATE: dict[str, tuple[int, int]] = {
    "Argentina": (24, 60),
    "Bolivia": (18, 45),
    "Brazil": (28, 70),
    "Chile": (20, 60),
    "Colombia": (24, 65),
    "Ecuador": (18, 75),
    "Paraguay": (28, 60),
    "Peru": (20, 80),
    "Uruguay": (22, 70),
    "Venezuela": (28, 75),
    "Canada": (20, 60),
    "Mexico": (22, 50),
    "United States": (26, 60),
    "Panama": (30, 80),
    "Haiti": (30, 75),
    "Curacao": (29, 75),
    "Costa Rica": (24, 80),
    "Jamaica": (30, 75),
    "Honduras": (28, 70),
    "Austria": (22, 60),
    "Belgium": (19, 70),
    "Bosnia and Herzegovina": (24, 55),
    "Croatia": (26, 60),
    "Czech Republic": (21, 60),
    "England": (18, 70),
    "France": (23, 60),
    "Germany": (21, 65),
    "Netherlands": (19, 75),
    "Norway": (17, 70),
    "Portugal": (26, 60),
    "Scotland": (16, 75),
    "Spain": (30, 40),
    "Sweden": (19, 65),
    "Switzerland": (22, 65),
    "Turkey": (28, 55),
    "Italy": (28, 60),
    "Poland": (21, 65),
    "Denmark": (19, 70),
    "Serbia": (24, 55),
    "Wales": (17, 75),
    "Ukraine": (23, 60),
    "Algeria": (30, 50),
    "Cape Verde": (27, 70),
    "DR Congo": (30, 75),
    "Egypt": (33, 45),
    "Ghana": (30, 80),
    "Ivory Coast": (30, 80),
    "Morocco": (28, 55),
    "Senegal": (30, 70),
    "South Africa": (22, 55),
    "Tunisia": (32, 55),
    "Nigeria": (31, 75),
    "Cameroon": (29, 80),
    "Australia": (24, 55),
    "Iran": (33, 30),
    "Iraq": (40, 25),
    "Japan": (28, 70),
    "Jordan": (30, 40),
    "Qatar": (40, 60),
    "Saudi Arabia": (38, 35),
    "South Korea": (27, 70),
    "Uzbekistan": (33, 40),
    "China PR": (28, 65),
    "New Zealand": (18, 70),
}

# --- WC2026 host-venue climate (late June - July) --------------------------
VENUE_CLIMATE: dict[tuple[str, str], tuple[int, int]] = {
    ("Arlington", "United States"): (36, 50),
    ("Atlanta", "United States"): (31, 65),
    ("East Rutherford", "United States"): (29, 65),
    ("Foxborough", "United States"): (27, 65),
    ("Houston", "United States"): (34, 75),
    ("Inglewood", "United States"): (26, 65),
    ("Kansas City", "United States"): (32, 60),
    ("Miami Gardens", "United States"): (32, 72),
    ("Philadelphia", "United States"): (30, 65),
    ("Santa Clara", "United States"): (28, 55),
    ("Seattle", "United States"): (24, 60),
    ("Toronto", "Canada"): (26, 65),
    ("Vancouver", "Canada"): (22, 65),
    ("Mexico City", "Mexico"): (23, 55),
    ("Guadalupe", "Mexico"): (35, 55),
    ("Zapopan", "Mexico"): (28, 50),
}


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path}  ({len(rows)} rows)")


def main():
    write_csv(
        os.path.join(REF, "team_meta.csv"),
        ["team", "confederation", "home_altitude_m"],
        [[t, c, a] for t, (c, a) in sorted(TEAM_META.items())],
    )
    write_csv(
        os.path.join(REF, "venues.csv"),
        ["city", "country", "altitude_m"],
        [[c, co, a] for (c, co), a in sorted(VENUES.items())],
    )
    write_csv(
        os.path.join(REF, "market_values.csv"),
        ["team", "squad_value_eur_m", "as_of"],
        [[t, v, "2026-06"] for t, v in sorted(MARKET_VALUES.items())],
    )
    write_csv(
        os.path.join(REF, "team_climate.csv"),
        ["team", "home_temp_c", "home_humidity"],
        [[t, tp, h] for t, (tp, h) in sorted(TEAM_CLIMATE.items())],
    )
    write_csv(
        os.path.join(REF, "venue_climate.csv"),
        ["city", "country", "temp_c", "humidity"],
        [[c, co, tp, h] for (c, co), (tp, h) in sorted(VENUE_CLIMATE.items())],
    )


if __name__ == "__main__":
    main()
