"""Fit a homogeneous Poisson process lambda(t) = mu to the flare catalog and
check goodness of fit via the waiting-time survival function and a
time-rescaling test.

Exposure, not calendar time
----------------------------
Waiting times are read from data/tess_waiting_times_exposure.csv, which
already integrates each star's observation windows between consecutive
flares (scripts/compute_exposure_waiting_times.py) rather than taking
np.diff(T_peak) directly. A raw np.diff would silently fold multi-month
inter-sector gaps into "the star went quiet for months," inflating waiting
times with unobserved time and corrupting both the rate estimate and every
downstream goodness-of-fit test. Rows flagged `overlaps_corrupted_orbit`
(exposure of unknown reliability around the corrupted Sector 26/Orbit 60
window; see data/README.md) are excluded throughout.

Rescaling
---------
For a homogeneous Poisson process, the compensator is Lambda(t) = mu * t
(exposure-time t), so the time-rescaled spacings

    z_i = integral_{t_{i-1}}^{t_i} lambda(u) du = mu * exposure_days_i

should be i.i.d. Exp(1) if the model is correctly specified (Ogata 1988).
z_i is computed for every interval that *closes at an event* - the `initial`
(window start -> first flare) and `interevent` (flare -> flare) rows of the
waiting-time table - using each interval's exposure_days. `final_censored`
rows (last flare -> window end) are right-censored and excluded from the
z_i ~ Exp(1) test, consistent with standard time-rescaling practice.

Two fits are reported:
  - "global": a single mu shared by the whole catalog - what
    lambda(t) = mu, without a star index, literally means.
  - "per-star": mu_TIC = N_TIC / E_TIC fit separately for each star with
    >= MIN_EVENTS_FOR_PER_STAR_FIT clean events, residuals pooled after
    fitting. This isolates whether flaring is non-Poisson *within* a star's
    own timeline from the (much larger, and unsurprising) fact that
    different stars simply flare at different average rates - the global
    fit conflates the two, so a global rejection alone does not establish
    self-excitation, only that "one rate for every star" is wrong. Reported
    as the more diagnostic test for that reason.

A naive-calendar counterexample (mu fit and z_i computed from calendar_days
instead of exposure_days - i.e. what you'd get from np.diff(T_peak) with no
window correction) is also computed and reported side by side, to make
concrete why it's wrong rather than merely asserting it.
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WAIT_PATH = "data/tess_waiting_times_exposure.csv"
OUT_FIG = "outputs/homogeneous_poisson_time_rescaling.png"
OUT_TXT = "outputs/homogeneous_poisson_gof_metrics.txt"
OUT_RESID = "outputs/homogeneous_poisson_residuals.csv"

MIN_EVENTS_FOR_PER_STAR_FIT = 3
ALPHA = 0.05


def fit_mu(events: pd.DataFrame, all_intervals: pd.DataFrame) -> float:
    """MLE mu = N / E: N events over ALL clean exposure (including the
    right-censored final_censored tail, which carries zero-event information)."""
    return len(events) / all_intervals["exposure_days"].sum()


def qq_theoretical_quantiles(n: int) -> np.ndarray:
    ranks = np.arange(1, n + 1)
    return -np.log(1 - (ranks - 0.5) / n)


def main() -> None:
    wait = pd.read_csv(WAIT_PATH)
    wait["row_in_tic"] = wait.groupby("TIC").cumcount()
    clean = wait[~wait["overlaps_corrupted_orbit"]].copy()
    closing = clean[clean["waiting_time_type"].isin(["initial", "interevent"])].copy()

    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("=== Homogeneous Poisson fit: lambda(t) = mu ===\n")

    # ---------------- Global fit ----------------
    mu_global = fit_mu(closing, clean)
    closing["z_global"] = mu_global * closing["exposure_days"]
    z_global = closing["z_global"].to_numpy()

    ks_global = stats.kstest(z_global, "expon")
    log("--- Global fit (single mu for the whole catalog) ---")
    log(f"N events (clean)      = {len(closing)}")
    log(f"Total exposure (days) = {clean['exposure_days'].sum():.2f}")
    log(f"mu_hat                = {mu_global:.5f} flares/day  (mean wait 1/mu = {1/mu_global:.3f} d)")
    log(f"mean(z), var(z)       = {z_global.mean():.4f}, {z_global.var():.4f}  (Exp(1): both = 1)")
    log(f"KS test vs Exp(1)     : D = {ks_global.statistic:.4f}, p = {ks_global.pvalue:.3e}")
    verdict = "REJECTED" if ks_global.pvalue < ALPHA else "not rejected"
    log(f"  -> homogeneous Poisson (global mu) is {verdict} at alpha={ALPHA}")

    # ---------------- Naive calendar-time counterexample ----------------
    E_calendar = clean["calendar_days"].sum()
    mu_naive = len(closing) / E_calendar
    z_naive = mu_naive * closing["calendar_days"].to_numpy()
    ks_naive = stats.kstest(z_naive, "expon")
    log("\n--- Counterexample: naive calendar-time rescaling (np.diff(T_peak), no window correction) ---")
    log(f"mu_hat (naive)        = {mu_naive:.5f} flares/day")
    log(f"mean(z), var(z)       = {z_naive.mean():.4f}, {z_naive.var():.4f}")
    log(f"KS test vs Exp(1)     : D = {ks_naive.statistic:.4f}, p = {ks_naive.pvalue:.3e}")
    log("  -> included only to show the effect of skipping exposure correction; not a candidate model")

    # ---------------- Per-star fit, pooled residuals ----------------
    z_per_star_parts = []
    n_stars_used = 0
    for tic, g in clean.groupby("TIC"):
        g_closing = g[g["waiting_time_type"].isin(["initial", "interevent"])]
        if len(g_closing) < MIN_EVENTS_FOR_PER_STAR_FIT:
            continue
        mu_star = fit_mu(g_closing, g)
        z_per_star_parts.append(mu_star * g_closing["exposure_days"].to_numpy())
        n_stars_used += 1
    z_per_star = np.concatenate(z_per_star_parts)
    ks_per_star = stats.kstest(z_per_star, "expon")
    log(f"\n--- Per-star fit (mu_TIC = N_TIC / E_TIC, stars with >= {MIN_EVENTS_FOR_PER_STAR_FIT} clean events), residuals pooled ---")
    log(f"Stars used            = {n_stars_used}")
    log(f"N residuals pooled    = {len(z_per_star)}")
    log(f"mean(z), var(z)       = {z_per_star.mean():.4f}, {z_per_star.var():.4f}")
    log(f"KS test vs Exp(1)     : D = {ks_per_star.statistic:.4f}, p = {ks_per_star.pvalue:.3e}")
    verdict = "REJECTED" if ks_per_star.pvalue < ALPHA else "not rejected"
    log(f"  -> homogeneous Poisson (per-star mu, pooled residuals) is {verdict} at alpha={ALPHA}")
    log(
        "  -> this isolates within-star temporal structure from between-star rate heterogeneity; "
        "a rejection here is the stronger evidence for self-excitation / clustering."
    )

    # ---------------- Independence check: lag-1 correlation of adjacent z's ----------------
    closing_sorted = closing.sort_values(["TIC", "row_in_tic"]).reset_index(drop=True)
    prev_tic = closing_sorted["TIC"].shift(1)
    prev_row = closing_sorted["row_in_tic"].shift(1)
    prev_z = closing_sorted["z_global"].shift(1)
    adjacent = (closing_sorted["TIC"] == prev_tic) & (closing_sorted["row_in_tic"] - prev_row == 1)
    u_curr = 1 - np.exp(-closing_sorted.loc[adjacent, "z_global"].to_numpy())
    u_prev = 1 - np.exp(-prev_z[adjacent].to_numpy())
    lag1_corr = np.corrcoef(u_prev, u_curr)[0, 1]
    log(f"\n--- Independence check (global-mu residuals) ---")
    log(f"Adjacent same-star pairs = {adjacent.sum()}")
    log(f"Lag-1 Pearson corr of u_i=1-exp(-z_i) = {lag1_corr:.4f}  (0 expected under independence)")

    # ---------------- Save residuals ----------------
    closing[["TIC", "waiting_time_type", "t_prev_btjd", "t_curr_btjd", "exposure_days", "z_global"]].to_csv(
        OUT_RESID, index=False
    )
    log(f"\nWrote {len(closing)} time-rescaled residuals to {OUT_RESID}")

    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote summary to {OUT_TXT}")

    # ---------------- Figure ----------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    def survival(z):
        zs = np.sort(z)
        s = 1 - np.arange(1, len(zs) + 1) / len(zs)
        return zs, s

    ax = axes[0, 0]
    zs, s = survival(z_global)
    ax.step(zs, s, where="post", label="empirical (exposure-based)", color="#2b6cb0")
    zn, sn = survival(z_naive)
    ax.step(zn, sn, where="post", label="empirical (naive calendar)", color="#c53030", alpha=0.7)
    xlim = zs.max() * 1.15
    zz = np.linspace(0, xlim, 200)
    ax.plot(zz, np.exp(-zz), "k--", label="theoretical Exp(1)")
    ax.set_yscale("log")
    ax.set_xlim(0, xlim)
    ax.set_xlabel("z")
    ax.set_ylabel("survival S(z) = P(Z > z)")
    ax.set_title("Waiting-time survival function\n(x-axis clipped to the exposure-based range; naive extends to z~80)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    n = len(z_global)
    theo_q = qq_theoretical_quantiles(n)
    ax.scatter(theo_q, np.sort(z_global), s=6, alpha=0.4, color="#2b6cb0")
    lim = max(theo_q.max(), z_global.max())
    ax.plot([0, lim], [0, lim], "k--")
    ax.set_xlabel("theoretical Exp(1) quantile")
    ax.set_ylabel("empirical z quantile (exposure-based)")
    ax.set_title(f"Q-Q plot, exposure-based\nKS D={ks_global.statistic:.3f}, p={ks_global.pvalue:.2e}")

    ax = axes[1, 0]
    n = len(z_naive)
    theo_q = qq_theoretical_quantiles(n)
    ax.scatter(theo_q, np.sort(z_naive), s=6, alpha=0.4, color="#c53030")
    lim = max(theo_q.max(), z_naive.max())
    ax.plot([0, lim], [0, lim], "k--")
    ax.set_xlabel("theoretical Exp(1) quantile")
    ax.set_ylabel("empirical z quantile (naive calendar)")
    ax.set_title(f"Q-Q plot, naive calendar-time\nKS D={ks_naive.statistic:.3f}, p={ks_naive.pvalue:.2e}")

    ax = axes[1, 1]
    ax.scatter(u_prev, u_curr, s=6, alpha=0.3, color="#2b6cb0")
    ax.set_xlabel(r"$u_i = 1-e^{-z_i}$")
    ax.set_ylabel(r"$u_{i+1}$")
    ax.set_title(f"Lag-1 independence check\ncorr={lag1_corr:.3f} (adjacent same-star pairs)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    print(f"Wrote figure to {OUT_FIG}")


if __name__ == "__main__":
    main()
