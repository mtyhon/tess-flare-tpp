# Data

## `tess_flares_feinstein.csv`

Preliminary catalog of stellar flares detected in TESS light curves (Feinstein et al.),
used as the event-time dataset for the temporal point process (TPP) analysis on this branch.

Columns:
- `TIC` — TESS Input Catalog ID of the host star.
- `T_peak`, `T_start`, `T_end` — flare peak/start/end times (TESS BJD offset).
- `Amplitude` — normalized flux amplitude of the flare.
- `ED` — equivalent duration.
- `Primary_or_Secondary` — whether the flare is a primary event or a secondary
  (sub-)flare superposed on a primary.
- `Num_Points`, `Num_Abv_Threshold` — number of light-curve points spanning the
  flare and number above the detection threshold.
- `Amp_Sigma` — flare amplitude significance in units of the local noise sigma.

19,422 flare events across 2,269 unique TICs — each target's `T_peak` sequence forms
an event stream suitable for point-process modeling (classical Hawkes/self-exciting
processes as a baseline, neural TPP variants as the main line of investigation).

## `tess_ffi_observation_times.csv`

Raw MIT TESS operations table of Full Frame Image sector/orbit data-collection
windows (`Sector`, `Orbit`, `Start Time`, `End Time`), as UTC calendar timestamps.
This defines the true observation support (and gaps — momentum dumps, downlink
gaps, sector-to-sector breaks) that a TPP likelihood needs to censor against; see
the discussion earlier in this thread on why event times alone are insufficient.
One row (Sector 26, Orbit 60) has a corrupted End Time (a `1971-12-31` sentinel,
not a real timestamp); it is preserved as-is for provenance and flagged rather
than silently converted downstream.

## `tess_ffi_observation_times_btjd.csv`

Derived from the table above by `scripts/convert_observation_times_to_btjd.py`,
adding `Start_BTJD` / `End_BTJD` columns and a `valid` flag, so the observation
windows share a time axis with `tess_flares_feinstein.csv`'s `T_peak`/`T_start`/`T_end`
(BTJD = BJD_TDB − 2457000).

The conversion is UTC → TDB (leap seconds + the 32.184 s TT−TAI offset, via
astropy) → Julian date → BTJD. This is **not** a full barycentric light-travel-time
correction — that term depends on each target's sky position (RA/Dec) and can
shift a timestamp by up to ~8.3 minutes, but this table carries no per-target
coordinates to compute it from. Given that sector/orbit windows span days, the
omission is immaterial for matching a flare's BTJD to the sector/orbit that
covers it (the intended use — building per-star observation windows for TPP
censoring). It would matter for sub-minute-precision boundary alignment; redo
that per target with astropy's `light_travel_time_correction` if it's ever needed.
The one row with a corrupted End Time has `End_BTJD = NaN` and `valid = False`.

## `tess_flares_feinstein_tagged.csv`, `tess_per_star_observation_windows.csv`, `tess_per_star_summary.csv`

Built by `scripts/build_observation_windows.py`, which consolidates the flare
catalog with the observation-times table into the per-star observation windows
a TPP likelihood needs for censoring:

- **`tess_flares_feinstein_tagged.csv`** — the flare catalog with `Sector`,
  `Orbit`, and `match_type` columns appended, recording which observation
  window each flare's `T_peak` falls into and how confidently (`exact`,
  `gap_fallback`, or `nearest_small_gap` — see the script docstring for what
  each means; 19,402 / 18 / 2 flares respectively, no flare left unmatched).
- **`tess_per_star_observation_windows.csv`** — one row per (TIC, Sector,
  Orbit) window the star is inferred to have been observed in: `TIC`,
  `Sector`, `Orbit`, `Start_BTJD`, `End_BTJD`.
- **`tess_per_star_summary.csv`** — one row per TIC: flare counts
  (`n_flares`, `n_primary`, `n_secondary`), `n_observed_sectors`,
  `exposure_days` (summed window durations), `obs_start_btjd`/`obs_end_btjd`,
  `first_flare_btjd`/`last_flare_btjd`, `flare_rate_per_day`, and
  `has_gap_fallback_sector` (flags the ~11 TICs whose Sector 26 membership
  came from the corrupted-row fallback rather than an exact match).

**Important limitation — read before fitting anything on `exposure_days`:**
neither input table records which TICs were actually in a given sector's
field of view, so a star's observed sectors are *inferred* as the sectors
containing at least one of its flares. A sector where the star was observed
but produced zero flares is invisible to this method, so `exposure_days` is a
systematic *undercount* of true exposure, and any background/quiescent rate
fit against it will be biased high. This is fine for a first pass, but before
drawing conclusions from flare rates, replace the inference step with a real
per-TIC sector list (e.g. via the `tess-point` package or an MAST/astroquery
lookup) and re-run. Separately, Sector 26/Orbit 60's exposure is entirely
excluded (not estimated) because its End Time is corrupted in the source
table — fix that at the source if precise exposure matters for the affected
stars.

## `tess_waiting_times_exposure.csv`

Built by `scripts/compute_exposure_waiting_times.py`. For each star, flare
waiting times measured on the *exposure clock* — the observation windows
integrated between consecutive flares — rather than raw calendar time, so
that time TESS wasn't looking at the star doesn't count as part of the wait.
This is the time-rescaling transform a TPP fit should be evaluated against:
under a correctly specified constant-rate model, `exposure_days` for
`interevent` rows should come out i.i.d. Exponential.

Columns: `TIC`, `waiting_time_type` (`initial` — window start to first
flare; `interevent` — between consecutive flares, the main quantity;
`final_censored` — last flare to window end), `t_prev_btjd`/`t_curr_btjd`,
`exposure_days` (the rescaled wait), `calendar_days` (the naive wait, for
comparison), `unobserved_days` (their difference — how much of the interval
was outside any known window), and `overlaps_corrupted_orbit` (True if the
interval touches Sector 26/Orbit 60's unknown-duration span — for these
rows `unobserved_days` conflates a real gap with the data-quality artifact,
so `exposure_days` is a genuine undercount rather than correct censoring;
18 flares / ~40 interevent rows are affected — filter these out for
anything sensitive to that distinction).

21,691 rows (17,153 interevent + 2,269 initial + 2,269 final_censored, one
each per TIC). Pooled interevent stats: mean 5.36 d, CV (std/mean) 1.29
(1.31 excluding rows touching the corrupted orbit) — a coefficient of
variation above 1 is consistent with clustered/self-exciting flaring rather
than a homogeneous Poisson process, which is the expected direction for
stellar flares and motivates the Hawkes/neural-TPP line of investigation
over a simple renewal model.

## Homogeneous Poisson fit and goodness of fit (`scripts/fit_homogeneous_poisson.py`)

Fits λ(t) = μ against the exposure-time waiting times above and tests it via
the time-rescaling theorem: zᵢ = μ · exposure_daysᵢ over every interval that
closes at an event (`initial` + `interevent` rows, excluding the corrupted-orbit
ones) should be i.i.d. Exp(1) if the model is correct. Outputs in `outputs/`:
`homogeneous_poisson_gof_metrics.txt` (numbers below), `homogeneous_poisson_time_rescaling.png`
(survival function, two Q-Q plots, lag-1 independence scatter), and
`homogeneous_poisson_residuals.csv` (the zᵢ themselves).

Three fits are reported, deliberately, because they answer different
questions:
- **Global** (one μ for the whole catalog, μ̂ = 0.150 flares/day): **rejected**
  (KS D = 0.138, p ≈ 0). This is the literal `λ(t) = μ` model.
- **Naive calendar-time counterexample** — μ fit and zᵢ computed from
  `calendar_days` instead of `exposure_days` (i.e. exactly the `np.diff(T_peak)`
  approach this pipeline was built to avoid): rejected far more badly
  (KS D = 0.314, var(z) ≈ 20.9 instead of 1). Included only to make the cost
  of skipping the exposure correction concrete, not as a candidate model.
- **Per-star** (μ_TIC = N_TIC/E_TIC fit separately per star with ≥3 clean
  events, residuals pooled after fitting): **also rejected** (KS D = 0.052,
  p = 1.5e-41), despite absorbing away each star's own average rate. This is
  the more diagnostic result — a global rejection alone could just mean
  "different stars flare at different rates," which is unsurprising and not
  evidence of clustering; rejecting *after* removing that effect means the
  non-Poisson structure is within individual stars' own timelines.

The lag-1 correlation between successive uᵢ = 1 − e^{−zᵢ} (same star, no
intervening excluded interval) is +0.22 — positive and not small given
n ≈ 16,140 pairs, i.e. a short wait tends to be followed by another short
wait. Together with the per-star rejection and the excess of short zᵢ
visible in the survival-function plot (empirical survival falls below the
Exp(1) line through the low-to-mid range before both curves' tails drop
off), this is consistent with self-exciting/clustered flaring — the
motivating case for moving to a Hawkes or neural TPP rather than a
homogeneous Poisson baseline. Treat "consistent with" as exactly that,
not confirmation: the pooled tests mix stars with very different amounts of
data, and a formal model comparison (e.g. against a fitted Hawkes model's
own time-rescaled residuals) is the next step before calling this settled.

## Hawkes fit and goodness of fit (`scripts/fit_hawkes_process.py`)

Fits an exponential-kernel Hawkes process, λ(t) = μ + α·Σ_{tⱼ<t} e^{−β(t−tⱼ)},
with one shared (μ, α, β) across all stars (each star's flares are an
independent realization — no cross-star excitation). The log-likelihood
integrates the compensator only over each star's known observation windows
(`tess_per_star_observation_windows.csv`) — never over unobserved time —
while every recorded flare, regardless of match-tier, still contributes to
the excitation sum, since its occurrence is known even where the surrounding
window isn't (mirrors how the corrupted Sector 26/Orbit 60 window is handled
throughout this pipeline). A built-in sanity check confirms the implementation:
the Hawkes log-likelihood at α→0 reproduces the Poisson log-likelihood
exactly (checked to 4 decimal places). The same time-rescaling residual
diagnostics as the Poisson fit are then run on zᵢ = ∫ λ(u) du over the same
event-closing intervals, now integrating the fitted self-exciting intensity
instead of a constant. Outputs in `outputs/`: `hawkes_gof_metrics.txt`,
`hawkes_time_rescaling.png` (Hawkes vs. Poisson survival function, both
Q-Q plots, lag-1 independence scatter), `hawkes_residuals.csv`.

**Fit:** μ̂ = 0.083 flares/day, α̂ = 0.051 flares/day, β̂ = 0.080/day
(excitation decay timescale 1/β ≈ 12.5 days), branching ratio α/β ≈ 0.63
(stationary — each flare triggers 0.63 "child" flares on average). Against
the Poisson baseline (fit at the same total exposure): likelihood-ratio
statistic 4791 on 2 extra degrees of freedom, p ≈ 0; ΔAIC ≈ −4787. This is
about as decisive as a model comparison gets — self-excitation is real and
strongly preferred over a constant rate.

**Goodness of fit — improved, but not resolved:**
- KS test on zᵢ: D = 0.064, p = 7.9e-67 — **still rejected**, but the KS
  statistic roughly halved from the Poisson global fit's D = 0.138.
- Lag-1 residual correlation: +0.028, down from +0.22 for Poisson — the
  Hawkes kernel absorbs the great majority of the serial dependence that a
  constant rate completely missed, though a small, still-detectable
  correlation remains at n ≈ 16,140 pairs.
- The survival-function and Q-Q plots show the *direction* of the remaining
  misfit has flipped relative to Poisson: rather than an excess of short
  waits, the Hawkes residuals now show a deficit of long waits relative to
  Exp(1) (the empirical survival curve falls *below* the Poisson one at
  large z, and the Q-Q plot bows below the diagonal). Read together with
  the long fitted decay timescale (~12.5 days — closer to active-region
  lifetime than to the hours-scale sympathetic-flaring timescale usually
  invoked for direct triggering), this pattern is consistent with a single
  shared exponential kernel being the wrong functional form (a power-law/
  Omori-type decay is the standard alternative in triggered-event modeling,
  e.g. seismology's ETAS), and/or with excitation strength varying by star
  rather than being shared globally — exactly the global-vs-per-star
  distinction that mattered for the Poisson background rate. Both are
  natural next refinements, ahead of moving to a neural TPP.
