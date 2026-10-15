"""FRED downloader (public CSV endpoint, no API key)."""
from __future__ import annotations

import io

import pandas as pd
import requests

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def parse_fred_csv(text: str, series_id: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text), na_values=["."])
    date_col, val_col = df.columns[0], df.columns[1]
    s = pd.Series(
        pd.to_numeric(df[val_col], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(pd.to_datetime(df[date_col]), name="date"),
        name=series_id,
    )
    return s.sort_index()


def fetch_fred(series_id: str, timeout: int = 60) -> pd.Series:
    r = requests.get(FRED_CSV, params={"id": series_id}, timeout=timeout)
    r.raise_for_status()
    return parse_fred_csv(r.text, series_id)
