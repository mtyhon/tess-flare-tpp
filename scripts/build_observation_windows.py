"""Build per-star observation windows for TPP censoring.

Consolidates the flare event catalog (data/tess_flares_feinstein.csv) with the
mission-wide FFI observation-times table (data/tess_ffi_observation_times_btjd.csv)
into, per TIC target: the set of sector/orbit windows during which it was under
observation, and a summary of its event stream against that exposure.

Sector-membership inference (read before using downstream)
------------------------------------------------------------
The FFI observation-times table is mission-wide — it says when TESS was
collecting *any* FFI data, not which sky field (and hence which targets) a
given sector covered. Neither input file carries a per-target sector list, so
a star's observed sectors are *inferred* here as the sectors in which it has
at least one recorded flare. A sector where a star was actually observed but
produced zero flares is invisible to this method and will be missing from its
window set, biasing total exposure (and any fitted background/quiescent rate)
downward. For exact coverage, replace this inference step with a true per-TIC
sector list (e.g. via the `tess-point` package or an astroquery/MAST lookup)
and re-run.

Flare-to-window matching has three tiers, recorded per flare as `match_type`:
  - "exact": T_peak falls inside a valid observation window.
  - "gap_fallback": T_peak falls inside the *invalid* Sector 26 / Orbit 60
    window (corrupted End Time in the source table — see data/README.md).
    Its Start_BTJD is trustworthy, so a time between that start and the next
    valid window's start is attributed to Sector 26, Orbit 60, but that
    orbit's true duration is unknown and it is NOT added to the per-star
    windows/exposure output — only its *sector* membership is used, so
    affected stars aren't dropped, but their exposure is a slight
    undercount. Fix at the source (correct the End Time from MIT's sector
    times page) if precise exposure for these stars matters.
  - "nearest_small_gap": T_peak falls in a real but short (<=1 day) gap
    between two valid windows (e.g. an inter-orbit downlink pause) or between
    a valid boundary and the reporting precision of the flare catalog itself;
    attributed to whichever neighboring window is closer.
  - "unmatched": no window, valid or invalid, is within a day - not expected
    to occur, printed as a warning if it does.

Once a sector is attributed to a star (by any tier), ALL of that sector's
*valid* orbit windows are kept as the star's observation windows there, flares
or not, since a target is either in a sector's field of view for the sector's
full duration or not at all (orbit-to-orbit gaps within a sector are
camera-wide downlink pauses, not target-specific, so they remain real gaps).
"""
import numpy as np
import pandas as pd

FLARES_PATH = "data/tess_flares_feinstein.csv"
OBS_PATH = "data/tess_ffi_observation_times_btjd.csv"
WINDOWS_OUT = "data/tess_per_star_observation_windows.csv"
SUMMARY_OUT = "data/tess_per_star_summary.csv"
FLARES_TAGGED_OUT = "data/tess_flares_feinstein_tagged.csv"

SMALL_GAP_TOLERANCE_DAYS = 1.0


def match_flares_to_windows(times: np.ndarray, obs_all: pd.DataFrame) -> pd.DataFrame:
    """Tiered match of each time to a (Sector, Orbit, match_type)."""
    starts = obs_all["Start_BTJD"].to_numpy()
    ends = obs_all["End_BTJD"].to_numpy()
    valid = obs_all["valid"].to_numpy()
    sectors = obs_all["Sector"].to_numpy()
    orbits = obs_all["Orbit"].to_numpy()
    n = len(obs_all)

    idx = np.clip(np.searchsorted(starts, times, side="right") - 1, 0, n - 1)

    out_sector = np.full(len(times), -1, dtype=int)
    out_orbit = np.full(len(times), -1, dtype=int)
    out_type = np.full(len(times), "unmatched", dtype=object)

    for i, (t, w) in enumerate(zip(times, idx)):
        next_start = starts[w + 1] if w + 1 < n else np.inf

        if valid[w] and t <= ends[w]:
            out_sector[i], out_orbit[i], out_type[i] = sectors[w], orbits[w], "exact"
            continue

        if not valid[w] and t < next_start:
            out_sector[i], out_orbit[i], out_type[i] = sectors[w], orbits[w], "gap_fallback"
            continue

        gap_before = t - ends[w] if valid[w] else t - starts[w]
        gap_after = next_start - t
        if min(gap_before, gap_after) <= SMALL_GAP_TOLERANCE_DAYS:
            j = w if gap_before <= gap_after else w + 1
            out_sector[i], out_orbit[i], out_type[i] = sectors[j], orbits[j], "nearest_small_gap"

    return pd.DataFrame({"Sector": out_sector, "Orbit": out_orbit, "match_type": out_type})


def main() -> None:
    flares = pd.read_csv(FLARES_PATH)
    flares["TIC"] = flares["TIC"].astype(int)

    obs_all = pd.read_csv(OBS_PATH).sort_values("Start_BTJD").reset_index(drop=True)
    obs_valid = obs_all[obs_all["valid"]].reset_index(drop=True)

    match = match_flares_to_windows(flares["T_peak"].to_numpy(), obs_all)
    flares = pd.concat([flares, match], axis=1)

    print("Flare-to-window match_type counts:")
    print(flares["match_type"].value_counts().to_string())
    unmatched = flares[flares["match_type"] == "unmatched"]
    if len(unmatched):
        print(f"\n{len(unmatched)} flares unmatched to any window:")
        print(unmatched[["TIC", "T_peak"]].to_string(index=False))

    flares.to_csv(FLARES_TAGGED_OUT, index=False)

    # Sectors inferred as "observed" for each TIC: any sector attributed by any match tier.
    attributed = flares[flares["Sector"] != -1]
    observed_sectors = attributed[["TIC", "Sector"]].drop_duplicates()

    # Attach every *valid* orbit window belonging to each star's inferred sectors.
    windows = observed_sectors.merge(obs_valid, on="Sector", how="left")
    windows = windows[["TIC", "Sector", "Orbit", "Start_BTJD", "End_BTJD"]].dropna()
    windows = windows.sort_values(["TIC", "Start_BTJD"]).reset_index(drop=True)
    windows.to_csv(WINDOWS_OUT, index=False)
    print(f"\nWrote {len(windows)} per-star window rows for {windows['TIC'].nunique()} TICs to {WINDOWS_OUT}")

    stars_with_no_window_rows = set(observed_sectors["TIC"]) - set(windows["TIC"])
    if stars_with_no_window_rows:
        print(
            f"{len(stars_with_no_window_rows)} TIC(s) have an inferred sector but zero valid orbit "
            f"windows in it (only the corrupted Sector 26/Orbit 60 was attributed): "
            f"{sorted(stars_with_no_window_rows)}"
        )

    exposure = (windows["End_BTJD"] - windows["Start_BTJD"]).groupby(windows["TIC"]).sum()
    exposure.name = "exposure_days"

    per_tic = flares.groupby("TIC").agg(
        n_flares=("T_peak", "size"),
        n_primary=("Primary_or_Secondary", lambda s: (s == "primary").sum()),
        n_secondary=("Primary_or_Secondary", lambda s: (s == "secondary").sum()),
        first_flare_btjd=("T_peak", "min"),
        last_flare_btjd=("T_peak", "max"),
        n_observed_sectors=("Sector", lambda s: pd.Series(s[s != -1]).nunique()),
        has_gap_fallback_sector=("match_type", lambda s: (s == "gap_fallback").any()),
    )
    obs_bounds = windows.groupby("TIC").agg(obs_start_btjd=("Start_BTJD", "min"), obs_end_btjd=("End_BTJD", "max"))

    summary = per_tic.join(exposure).join(obs_bounds)
    summary["flare_rate_per_day"] = summary["n_flares"] / summary["exposure_days"]
    summary = summary.reset_index().sort_values("TIC")
    summary.to_csv(SUMMARY_OUT, index=False)
    print(f"Wrote per-star summary for {len(summary)} TICs to {SUMMARY_OUT}")
    print(f"TICs with zero exposure (no valid window matched at all): {summary['exposure_days'].isna().sum()}")
    print(summary[["n_flares", "n_observed_sectors", "exposure_days", "flare_rate_per_day"]].describe().to_string())


if __name__ == "__main__":
    main()
