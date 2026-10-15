import numpy as np
import pandas as pd

from oilfc.data.align import align_asof, availability


def _monthly():
    idx = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])
    return pd.Series([1.0, 2.0, 3.0], index=idx)


def test_release_alignment_monthly_lag():
    # CPI-style: dated 1st of month, usable 45 days later (Jan -> Feb 15, Feb -> Mar 17).
    targets = pd.to_datetime(["2020-02-10", "2020-02-15", "2020-03-16", "2020-03-17"])
    a = align_asof(_monthly(), targets, "days", 45)
    assert a["value"].isna().iloc[0]            # nothing released yet
    assert a["value"].tolist()[1:] == [1.0, 1.0, 2.0]
    assert a["obs_date"].iloc[3] == pd.Timestamp("2020-02-01")
    assert a["release_age_days"].iloc[3] == 0


def test_no_future_rows_availability_never_exceeds_target():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=400)
    s = pd.Series(rng.normal(size=len(idx)), index=idx)
    targets = pd.bdate_range("2015-06-01", periods=200)
    for kind, n in [("bdays", 0), ("bdays", 5), ("days", 45)]:
        a = align_asof(s, targets, kind, n).dropna()
        avail = availability(pd.DatetimeIndex(a["obs_date"]), kind, n)
        assert (avail <= a.index).all()


def test_no_future_rows_perturbing_unreleased_values_changes_nothing():
    idx = pd.bdate_range("2018-01-01", periods=300)
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    t = pd.Timestamp("2018-09-03")
    base = align_asof(s, pd.DatetimeIndex([t]), "bdays", 5)["value"].iloc[0]
    s2 = s.copy()
    s2[s2.index > t - pd.offsets.BDay(5)] = 1e9   # only values not yet available at t
    assert align_asof(s2, pd.DatetimeIndex([t]), "bdays", 5)["value"].iloc[0] == base


def test_zero_lag_uses_same_day_value():
    idx = pd.bdate_range("2020-01-01", periods=5)
    s = pd.Series([10.0, 11, 12, 13, 14], index=idx)
    a = align_asof(s, idx, "bdays", 0)
    assert a["value"].tolist() == [10.0, 11, 12, 13, 14]


def test_gaps_forward_fill_with_age():
    idx = pd.to_datetime(["2020-01-02", "2020-01-08"])
    s = pd.Series([5.0, 6.0], index=idx)
    a = align_asof(s, pd.to_datetime(["2020-01-06", "2020-01-08"]), "bdays", 0)
    assert a["value"].tolist() == [5.0, 6.0]
    assert a["release_age_days"].tolist() == [4, 0]
