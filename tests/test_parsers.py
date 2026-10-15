import pandas as pd

from oilfc.data.eia import clean_eia_frame
from oilfc.data.fred import parse_fred_csv


def test_parse_fred_csv_handles_missing_marker():
    text = "observation_date,VIXCLS\n2020-01-01,.\n2020-01-02,12.5\n2020-01-03,13.1\n"
    s = parse_fred_csv(text, "VIXCLS")
    assert s.name == "VIXCLS" and s.index.name == "date"
    assert s.isna().tolist() == [True, False, False]
    assert s.iloc[1] == 12.5


def test_clean_eia_frame_skips_header_rows():
    raw = pd.DataFrame(
        [
            ["Back to Contents", "Data 1: Europe Brent Spot Price"],
            ["Sourcekey", "RBRTE"],
            ["Date", "Europe Brent Spot Price FOB"],
            ["1987-05-21", 18.45],
            ["1987-05-20", 18.63],
        ]
    )
    s = clean_eia_frame(raw, "RBRTE")
    assert s.name == "RBRTE" and s.index.is_monotonic_increasing
    assert s.loc["1987-05-20"] == 18.63 and len(s) == 2
