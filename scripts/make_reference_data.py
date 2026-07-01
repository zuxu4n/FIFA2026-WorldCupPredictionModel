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

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "..", "data", "reference")
os.makedirs(REF, exist_ok=True)

# --- team -> (confederation, typical home-venue altitude in metres) --------
# Altitudes are approximate elevations of the country's usual home venue.
TEAM_META: dict[str, tuple[str, int]] = {
    # CONMEBOL
    "Argentina": ("CONMEBOL", 25), "Bolivia": ("CONMEBOL", 3640),
    "Brazil": ("CONMEBOL", 700), "Chile": ("CONMEBOL", 570),
    "Colombia": ("CONMEBOL", 18), "Ecuador": ("CONMEBOL", 2850),
    "Paraguay": ("CONMEBOL", 130), "Peru": ("CONMEBOL", 150),
    "Uruguay": ("CONMEBOL", 40), "Venezuela": ("CONMEBOL", 900),
    # CONCACAF
    "Canada": ("CONCACAF", 80), "Mexico": ("CONCACAF", 2240),
    "United States": ("CONCACAF", 150), "Panama": ("CONCACAF", 10),
    "Haiti": ("CONCACAF", 30), "Curacao": ("CONCACAF", 5),
    "Costa Rica": ("CONCACAF", 1170), "Jamaica": ("CONCACAF", 10),
    "Honduras": ("CONCACAF", 990),
    # UEFA
    "Austria": ("UEFA", 190), "Belgium": ("UEFA", 30),
    "Bosnia and Herzegovina": ("UEFA", 510), "Croatia": ("UEFA", 120),
    "Czech Republic": ("UEFA", 200), "England": ("UEFA", 30),
    "France": ("UEFA", 35), "Germany": ("UEFA", 150),
    "Netherlands": ("UEFA", 5), "Norway": ("UEFA", 20),
    "Portugal": ("UEFA", 100), "Scotland": ("UEFA", 50),
    "Spain": ("UEFA", 660), "Sweden": ("UEFA", 20),
    "Switzerland": ("UEFA", 430), "Turkey": ("UEFA", 120),
    "Italy": ("UEFA", 90), "Poland": ("UEFA", 110),
    "Denmark": ("UEFA", 10), "Serbia": ("UEFA", 120),
    "Wales": ("UEFA", 30), "Ukraine": ("UEFA", 180),
    # CAF
    "Algeria": ("CAF", 25), "Cape Verde": ("CAF", 30),
    "DR Congo": ("CAF", 280), "Egypt": ("CAF", 70),
    "Ghana": ("CAF", 60), "Ivory Coast": ("CAF", 50),
    "Morocco": ("CAF", 250), "Senegal": ("CAF", 20),
    "South Africa": ("CAF", 1750), "Tunisia": ("CAF", 10),
    "Nigeria": ("CAF", 480), "Cameroon": ("CAF", 700),
    # AFC
    "Australia": ("AFC", 20), "Iran": ("AFC", 1200),
    "Iraq": ("AFC", 34), "Japan": ("AFC", 40),
    "Jordan": ("AFC", 770), "Qatar": ("AFC", 10),
    "Saudi Arabia": ("AFC", 610), "South Korea": ("AFC", 40),
    "Uzbekistan": ("AFC", 450), "China PR": ("AFC", 50),
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
    ("Guadalupe", "Mexico"): 500,       # Monterrey metro (Estadio BBVA)
    ("Zapopan", "Mexico"): 1566,        # Guadalajara metro (Estadio Akron)
    # A few famous high-altitude football cities, so the altitude effect is
    # learned from history too (used by the city->altitude resolver).
    ("La Paz", "Bolivia"): 3640, ("El Alto", "Bolivia"): 4090,
    ("Quito", "Ecuador"): 2850, ("Cuenca", "Ecuador"): 2560,
    ("Bogota", "Colombia"): 2640, ("Barranquilla", "Colombia"): 18,
    ("Cusco", "Peru"): 3400, ("Lima", "Peru"): 150,
    ("Addis Ababa", "Ethiopia"): 2355, ("Johannesburg", "South Africa"): 1753,
    ("Toluca", "Mexico"): 2660, ("Pachuca", "Mexico"): 2400,
    ("San Jose", "Costa Rica"): 1170, ("Tehran", "Iran"): 1200,
    ("Madrid", "Spain"): 660, ("Sucre", "Bolivia"): 2810,
}

# --- WC2026 squad market values (EUR millions) -----------------------------
# REAL Transfermarkt squad values for all 48 finalists (as of June 2026), via
# planetfootball.com's "World Cup 2026: all 48 squads ranked by market value".
# Non-qualified teams below the divider keep approximate values purely so the
# feature is populated for historical training matches. Refresh from
# transfermarkt.com when squads change (see data/README.md).
MARKET_VALUES: dict[str, int] = {
    # --- 48 WC2026 finalists (real Transfermarkt, EUR millions) ---
    "France": 1520, "England": 1360, "Spain": 1220, "Portugal": 1010,
    "Germany": 947, "Brazil": 928, "Argentina": 808, "Netherlands": 754,
    "Norway": 590, "Belgium": 548, "Ivory Coast": 522, "Senegal": 478,
    "Turkey": 474, "Morocco": 448, "Sweden": 406, "Croatia": 387,
    "United States": 386, "Ecuador": 369, "Uruguay": 359, "Switzerland": 333,
    "Colombia": 302, "Japan": 271, "Algeria": 257, "Austria": 245,
    "Ghana": 235, "Canada": 199, "Mexico": 192, "Czech Republic": 188,
    "Scotland": 170, "Paraguay": 154, "Bosnia and Herzegovina": 146,
    "DR Congo": 144, "South Korea": 139, "Egypt": 116, "Uzbekistan": 85,
    "Australia": 77, "Tunisia": 70, "Haiti": 56, "Cape Verde": 49,
    "South Africa": 49, "Saudi Arabia": 41, "Panama": 35, "New Zealand": 34,
    "Iran": 32, "Curacao": 26, "Iraq": 21, "Jordan": 20, "Qatar": 20,
    # --- non-qualified teams (approximate, for historical coverage) ---
    "Italy": 700, "Denmark": 320, "Serbia": 300, "Nigeria": 280,
    "Poland": 320, "Cameroon": 280, "Venezuela": 120, "Bolivia": 25,
    "Peru": 90, "Chile": 150, "Costa Rica": 40,
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


if __name__ == "__main__":
    main()
