"""Download all series, cache raw files, build the release-aligned daily panel.

    python -m oilfc.data.build [--offline]   (--offline reuses data/raw/*.parquet)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from .align import align_asof
from .eia import fetch_eia_spot
from .fred import fetch_fred

ROOT = Path(__file__).resolve().parents[3]
RAW, PROCESSED = ROOT / "data" / "raw", ROOT / "data" / "processed"


def _get(name: str, fetch, offline: bool) -> pd.Series:
    path = RAW / f"{name}.parquet"
    if offline and path.exists():
        return pd.read_parquet(path)[name]
    s = fetch().rename(name)
    s.to_frame().to_parquet(path)
    return s


def build(offline: bool = False) -> pd.DataFrame:
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    prices = {n: _get(n, lambda k=v["key"]: fetch_eia_spot(k), offline) for n, v in cfg["eia"].items()}
    target = cfg["target"]
    index = prices[target].dropna().index  # origins = days with a target observation

    cols = {n: s.reindex(index) for n, s in prices.items()}  # spot prices: lag 0
    for name, spec in cfg["fred"].items():
        s = _get(name, lambda i=spec["id"]: fetch_fred(i), offline)
        a = align_asof(s, index, spec["lag"]["kind"], spec["lag"]["n"])
        cols[name] = a["value"]
        cols[f"{name}_age"] = a["release_age_days"]
    panel = pd.DataFrame(cols, index=index)
    panel.to_parquet(PROCESSED / "panel_daily.parquet")
    return panel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    p = build(ap.parse_args().offline)
    print(p.tail(3).T.to_string())
    print(f"\n{len(p)} rows, {p.index.min().date()} .. {p.index.max().date()}")
