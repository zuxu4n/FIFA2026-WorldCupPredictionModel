"""Fetch actual forecast conditions for upcoming fixtures from Open-Meteo
(free, keyless) and write data/reference/weather_overrides.csv. The
ClimateResolver prefers these per-date values over climate normals, so a
scorching afternoon in Dallas reads differently from a mild evening.

    python scripts/fetch_weather.py
"""
from __future__ import annotations
import json
import os
import urllib.request
import pandas as pd
from wcpred import config as C
from wcpred import data as D

COORDS_CSV = os.path.join(C.DATA_REF, "venue_coords.csv")
OUT = os.path.join(C.DATA_REF, "weather_overrides.csv")


def _forecast(lat: float, lon: float, day: str):
    """(daily max temp C, daily mean RH %) for one date, or None if unavailable."""
    url = ("https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}"
           "&daily=temperature_2m_max,relative_humidity_2m_mean"
           f"&start_date={day}&end_date={day}&timezone=auto")
    req = urllib.request.Request(url, headers={"User-Agent": "wcpred/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read())
    try:
        return (float(d["daily"]["temperature_2m_max"][0]),
                float(d["daily"]["relative_humidity_2m_mean"][0]))
    except (KeyError, IndexError, TypeError):
        return None


def main():
    coords = pd.read_csv(COORDS_CSV, encoding="utf-8")
    coords["city"] = coords["city"].map(D.strip_accents)
    cmap = {(r.city, r.country): (r.lat, r.lon) for r in coords.itertuples(index=False)}

    res = D.load_results()
    up = res[~res.played]
    rows, missing = [], set()
    for r in up.itertuples(index=False):
        key = (r.city, r.country)
        if key not in cmap:
            missing.add(key)
            continue
        day = pd.Timestamp(r.date).date().isoformat()
        try:
            fc = _forecast(*cmap[key], day)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {r.city} {day}: {e}")
            continue
        if fc:
            rows.append([day, r.city, r.country, round(fc[0], 1), round(fc[1])])
            print(f"  {day}  {r.city:<18} {fc[0]:5.1f}C  {fc[1]:3.0f}%RH")
    if missing:
        print("no coordinates for:", sorted(missing))
    df = pd.DataFrame(rows, columns=["date", "city", "country", "temp_c", "humidity"])
    df = df.drop_duplicates(subset=["date", "city", "country"])
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(df)} fixture-days)")


if __name__ == "__main__":
    main()
