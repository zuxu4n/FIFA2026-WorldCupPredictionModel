"""Download / refresh the raw international results dataset.

    python scripts/fetch_data.py

Source: github.com/martj42/international_results (updated continually, including
the in-progress World Cup). Falls back with a clear message if offline.
"""
from __future__ import annotations
import urllib.request
from wcpred import config as C


def _get(url: str, dest: str):
    print(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "wcpred/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    with open(dest, "wb") as f:
        f.write(data)
    print(f"  -> {dest}  ({len(data):,} bytes)")


def main():
    try:
        _get(C.RESULTS_URL, C.RESULTS_CSV)
        _get(C.SHOOTOUTS_URL, C.SHOOTOUTS_CSV)
        print("done.")
    except Exception as e:  # noqa: BLE001
        print(f"\nFETCH FAILED: {e}\n"
              "If you are offline, manually place results.csv (and optionally\n"
              "shootouts.csv) in data/raw/ — see data/README.md for the schema.")


if __name__ == "__main__":
    main()
