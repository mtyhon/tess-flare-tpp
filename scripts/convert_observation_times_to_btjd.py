"""Convert the TESS FFI sector/orbit observation-window table to BTJD.

The MIT observation-times table (data/tess_ffi_observation_times.csv) gives each
sector/orbit's data-collection window as UTC calendar timestamps. The flare
catalog (data/tess_flares_feinstein.csv) instead uses BTJD:

    BTJD = BJD_TDB - 2457000

To let the two tables share a time axis for building per-star TPP observation
windows, this script converts each UTC timestamp to BTJD via astropy: UTC ->
TDB (leap seconds + the 32.184s TT-TAI offset), then to a Julian date, then
shifted by 2457000.

Caveat: this is a geocentric UTC->TDB conversion, not a full barycentric
light-travel-time correction. That correction depends on each target's sky
position (RA/Dec) and can shift arrival times by up to ~8.3 minutes, but the
sector/orbit windows here span days, so the omission is immaterial for
matching a flare's BTJD to the sector/orbit that covers it. It would matter
for sub-minute precision at a window boundary — recompute per-target with
astropy's light_travel_time_correction if that's ever needed.

One row (Sector 26, Orbit 60) has a corrupted End Time (1971-12-31, a
sentinel/null value, not a real timestamp) in the source table; it is kept
in the output with its BTJD columns and `valid` flag set to False rather
than silently converted or dropped.
"""
import pandas as pd
from astropy.time import Time

RAW_PATH = "data/tess_ffi_observation_times.csv"
OUT_PATH = "data/tess_ffi_observation_times_btjd.csv"
BTJD_OFFSET = 2457000.0


def utc_to_btjd(series: pd.Series) -> pd.Series:
    times = Time(series.to_numpy(), scale="utc")
    return pd.Series(times.tdb.jd - BTJD_OFFSET, index=series.index)


def main() -> None:
    df = pd.read_csv(RAW_PATH, comment="#").dropna(how="all")
    df["Start Time"] = pd.to_datetime(df["Start Time"])
    df["End Time"] = pd.to_datetime(df["End Time"])

    df["valid"] = df["End Time"] > df["Start Time"]
    invalid = df.loc[~df["valid"]]
    if len(invalid):
        print(f"Warning: {len(invalid)} row(s) with a corrupted End Time <= Start Time:")
        print(invalid.to_string(index=False))

    df["Start_BTJD"] = utc_to_btjd(df["Start Time"])
    df["End_BTJD"] = utc_to_btjd(df["End Time"])
    df.loc[~df["valid"], "End_BTJD"] = float("nan")

    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH} ({(~df['valid']).sum()} flagged invalid)")


if __name__ == "__main__":
    main()
