"""Measure flare waiting times in continuous observed exposure, per star.

A raw calendar gap between two flares (T_peak_i - T_peak_{i-1}) is the wrong
clock for a TPP: TESS wasn't observing continuously, so part of that gap is
just "the telescope was pointed elsewhere" (safety of the intensity function
requires this decomposition — see the discussion earlier in this thread on
why observation windows matter). This script instead integrates each star's
observation windows (data/tess_per_star_observation_windows.csv) between
consecutive flare times to get the *exposure time* elapsed — the clock a
Poisson/Hawkes/neural TPP should actually see. This is the standard
time-rescaling transform: under a correctly specified constant-rate model,
these exposure-time waiting times should be i.i.d. Exponential.

Per star, three kinds of interval are emitted (`waiting_time_type`):
  - "initial": from the start of the star's first known observation window
    to its first flare (left boundary of the record).
  - "interevent": between two consecutive flares — the main quantity.
  - "final_censored": from the last flare to the end of the star's last
    known observation window (right-censored — no event followed, but we
    know exposure continued this long).

Two columns make the exposure-vs-calendar gap and its cause visible for every
row, rather than silently collapsing them:
  - `unobserved_days` = calendar_days - exposure_days: how much of the
    interval was outside any known window (usually a real inter-sector gap).
  - `overlaps_corrupted_orbit`: True if the interval overlaps Sector 26 /
    Orbit 60, whose true duration is unknown (corrupted End Time — see
    data/README.md). For these rows `unobserved_days` conflates a real gap
    with a data-quality artifact: the star was very likely being observed
    for at least part of that stretch, so exposure_days is a genuine
    undercount here, not just correct censoring. Treat these rows with
    caution (or exclude them) in anything sensitive to that distinction.
"""
import numpy as np
import pandas as pd

FLARES_PATH = "data/tess_flares_feinstein_tagged.csv"
WINDOWS_PATH = "data/tess_per_star_observation_windows.csv"
OBS_PATH = "data/tess_ffi_observation_times_btjd.csv"
OUT_PATH = "data/tess_waiting_times_exposure.csv"


def exposure_between(t0: float, t1: float, win_starts: np.ndarray, win_ends: np.ndarray) -> float:
    overlap = np.minimum(t1, win_ends) - np.maximum(t0, win_starts)
    return float(np.clip(overlap, 0, None).sum())


def corrupted_orbit_ranges(obs_all: pd.DataFrame) -> list[tuple[float, float]]:
    """[start, next_valid_start) span for each invalid (unknown-duration) window."""
    obs_all = obs_all.sort_values("Start_BTJD").reset_index(drop=True)
    starts = obs_all["Start_BTJD"].to_numpy()
    ranges = []
    for i, row in obs_all.iterrows():
        if not row["valid"]:
            next_start = starts[i + 1] if i + 1 < len(obs_all) else np.inf
            ranges.append((row["Start_BTJD"], next_start))
    return ranges


def overlaps_any(t0: float, t1: float, ranges: list[tuple[float, float]]) -> bool:
    return any(t0 < b and a < t1 for a, b in ranges)


def main() -> None:
    flares = pd.read_csv(FLARES_PATH)
    windows = pd.read_csv(WINDOWS_PATH)
    obs_all = pd.read_csv(OBS_PATH)
    corrupted_ranges = corrupted_orbit_ranges(obs_all)

    rows = []
    for tic, wgrp in windows.groupby("TIC"):
        win_starts = wgrp["Start_BTJD"].to_numpy()
        win_ends = wgrp["End_BTJD"].to_numpy()
        obs_start, obs_end = win_starts.min(), win_ends.max()

        fgrp = flares[flares["TIC"] == tic].sort_values("T_peak")
        peaks = fgrp["T_peak"].to_numpy()
        match_types = fgrp["match_type"].to_numpy()

        boundaries = np.concatenate([[obs_start], peaks, [obs_end]])
        # boundaries has len(peaks)+2 entries -> len(peaks)+1 intervals; label each interval
        # by what closes it: initial (0), interevent (1..n-1), final_censored (n).
        n = len(peaks)
        interval_labels = (["initial"] if n >= 1 else []) + ["interevent"] * max(n - 1, 0) + (["final_censored"] if n >= 1 else [])

        for i in range(len(boundaries) - 1):
            t0, t1 = boundaries[i], boundaries[i + 1]
            exp = exposure_between(t0, t1, win_starts, win_ends)
            cal = t1 - t0
            rows.append(
                {
                    "TIC": tic,
                    "waiting_time_type": interval_labels[i],
                    "t_prev_btjd": t0,
                    "t_curr_btjd": t1,
                    "exposure_days": exp,
                    "calendar_days": cal,
                    "unobserved_days": cal - exp,
                    "overlaps_corrupted_orbit": overlaps_any(t0, t1, corrupted_ranges),
                    "match_type_curr": match_types[i] if i < n else np.nan,
                }
            )

    wait = pd.DataFrame(rows)
    wait.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(wait)} waiting-time rows for {wait['TIC'].nunique()} TICs to {OUT_PATH}")
    print(wait["waiting_time_type"].value_counts().to_string())

    interevent = wait[wait["waiting_time_type"] == "interevent"]
    clean = interevent[~interevent["overlaps_corrupted_orbit"]]
    print(f"\ninterevent rows: {len(interevent)} total, {len(clean)} not touching the corrupted orbit")
    for label, df in [("all interevent", interevent), ("clean interevent", clean)]:
        exp = df["exposure_days"]
        mean, std = exp.mean(), exp.std()
        print(
            f"{label}: n={len(df)}  mean={mean:.4f}d  std={std:.4f}d  "
            f"CV={std / mean:.3f}  (CV=1 -> Poisson-consistent; >1 -> clustered/self-exciting; <1 -> regular)"
        )


if __name__ == "__main__":
    main()
