"""Release-date alignment: the leakage guard for covariates.

A value observed on date d becomes usable at availability(d) = d + lag.
`align_asof` returns, for each target date t, the latest value with availability <= t.
"""
from __future__ import annotations

import pandas as pd


def availability(index: pd.DatetimeIndex, kind: str, n: int) -> pd.DatetimeIndex:
    if kind == "bdays":
        return index + pd.offsets.BDay(n) if n else index
    if kind == "days":
        return index + pd.Timedelta(days=n)
    raise ValueError(f"unknown lag kind: {kind}")


def align_asof(series: pd.Series, target_index: pd.DatetimeIndex, kind: str = "bdays", n: int = 0) -> pd.DataFrame:
    """Returns DataFrame[value, obs_date, release_age_days] indexed by target_index."""
    s = series.dropna().sort_index()
    left = pd.DataFrame({"t": pd.DatetimeIndex(target_index).sort_values()})
    right = pd.DataFrame(
        {"avail": availability(s.index, kind, n), "obs_date": s.index, "value": s.to_numpy()}
    ).sort_values("avail")
    m = pd.merge_asof(left, right, left_on="t", right_on="avail", direction="backward")
    out = pd.DataFrame(
        {
            "value": m["value"].to_numpy(),
            "obs_date": m["obs_date"].to_numpy(),
            "release_age_days": (m["t"] - m["avail"]).dt.days.to_numpy(),
        },
        index=pd.DatetimeIndex(m["t"], name="date"),
    )
    return out
