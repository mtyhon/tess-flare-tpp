"""Fit a Hawkes process to the flare catalog and run the same time-rescaling
residual diagnostics as scripts/fit_homogeneous_poisson.py, to check whether
allowing self-excitation resolves the homogeneous Poisson's goodness-of-fit
failure.

Model
-----
Exponential-kernel Hawkes process, fit jointly (one shared mu, alpha, beta)
across all stars, each star's flare sequence treated as an independent
realization (no cross-star excitation - these are physically different
objects):

    lambda_TIC(t) = mu + alpha * sum_{t_j < t, t_j in TIC} exp(-beta (t - t_j))

alpha/beta is the branching ratio: the expected number of "child" flares
directly triggered by one flare. 1/beta is the excitation timescale.

Censoring, handled the same way as the Poisson baseline
---------------------------------------------------------
The log-likelihood is

    sum_i log lambda(t_i^-)  -  integral_{observed windows} lambda(u) du

using data/tess_per_star_observation_windows.csv for the *compensator*
integral (i.e. never integrating over unobserved time, same principle as
the exposure-time waiting times), while *all* recorded flares - regardless
of which match_type tier they came from - contribute to the excitation sum,
since a flare's own time is known even where the surrounding window isn't.
This mirrors the Poisson script's treatment of the corrupted Sector
26/Orbit 60 window: its duration is excluded from the compensator (unknown),
but flares known to have occurred there still excite later intensity.

Residual diagnostics
---------------------
For each interval that closes at an event (the same `initial`/`interevent`
rows used by the Poisson script, built by scripts/compute_exposure_waiting_times.py,
again excluding rows flagged `overlaps_corrupted_orbit`), the time-rescaled
residual is

    z_i = integral over the *observed* sub-segments of [t_{i-1}, t_i] of
          lambda(u) du

computed with the fitted (mu, alpha, beta) and that star's flare history up
to t_{i-1} (no other flares of the star fall strictly inside the interval,
by construction). Under a correctly specified model, z_i ~ i.i.d. Exp(1) -
exactly the same test as before, now against a self-exciting intensity
instead of a constant one. The same KS test, survival function, Q-Q plot,
and lag-1 independence check are produced, plotted alongside the Poisson
residuals for direct comparison.

Scope note: this fits one shared kernel for the whole catalog (paralleling
the Poisson script's "global" fit, the one that was decisively rejected),
not a per-star baseline. A per-star baseline mu_TIC with a shared (alpha,
beta) - paralleling the Poisson script's per-star fit, which isolated
within-star structure from between-star rate heterogeneity - is the natural
next refinement and is not attempted here.
"""
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FLARES_PATH = "data/tess_flares_feinstein_tagged.csv"
WINDOWS_PATH = "data/tess_per_star_observation_windows.csv"
WAIT_PATH = "data/tess_waiting_times_exposure.csv"
POISSON_RESID_PATH = "outputs/homogeneous_poisson_residuals.csv"

OUT_FIG = "outputs/hawkes_time_rescaling.png"
OUT_TXT = "outputs/hawkes_gof_metrics.txt"
OUT_RESID = "outputs/hawkes_residuals.csv"

ALPHA_LEVEL = 0.05


def load_star_data():
    flares = pd.read_csv(FLARES_PATH)
    windows = pd.read_csv(WINDOWS_PATH)
    events = {tic: np.sort(g["T_peak"].to_numpy()) for tic, g in flares.groupby("TIC")}
    wins = {
        tic: list(zip(g["Start_BTJD"].to_numpy(), g["End_BTJD"].to_numpy()))
        for tic, g in windows.groupby("TIC")
    }
    return events, wins


def excitation_integral(a: float, b: float, hist: np.ndarray, alpha: float, beta: float) -> float:
    """integral_a^b alpha * sum_{t_j in hist, t_j < b} exp(-beta (u - t_j)) du"""
    active = hist[hist < b]
    if len(active) == 0:
        return 0.0
    dt_a = np.clip(a - active, 0, None)
    dt_b = b - active
    return float((alpha / beta) * np.sum(np.exp(-beta * dt_a) - np.exp(-beta * dt_b)))


def compensator(win_list, events_arr, mu, alpha, beta) -> float:
    total = 0.0
    for a, b in win_list:
        total += mu * (b - a)
        total += excitation_integral(a, b, events_arr, alpha, beta)
    return total


def event_log_intensities(events_arr: np.ndarray, mu: float, alpha: float, beta: float) -> np.ndarray:
    """log lambda(t_i^-) for each event, via the O(n) exponential-kernel recursion."""
    n = len(events_arr)
    if n == 0:
        return np.array([])
    A = np.zeros(n)
    for i in range(1, n):
        dt = events_arr[i] - events_arr[i - 1]
        A[i] = np.exp(-beta * dt) * (1 + A[i - 1])
    lam = mu + alpha * A
    return np.log(lam)


def neg_log_likelihood(log_params, events: dict, wins: dict) -> float:
    mu, alpha, beta = np.exp(log_params)
    ll = 0.0
    for tic, ev in events.items():
        ll += event_log_intensities(ev, mu, alpha, beta).sum()
        ll -= compensator(wins.get(tic, []), ev, mu, alpha, beta)
    return -ll


def main():
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    events, wins = load_star_data()
    n_total_events = sum(len(v) for v in events.values())
    log(f"Loaded {n_total_events} flares across {len(events)} TICs, {sum(len(v) for v in wins.values())} windows.\n")

    # Sanity check: neg_log_likelihood at alpha->0 should match the pure-Poisson
    # log-likelihood N*log(mu) - mu*E for the SAME mu (using only windows present
    # in `wins`, i.e. the same "clean" exposure as the Poisson script's global fit).
    mu_check = 0.15
    E_total = sum(b - a for w in wins.values() for a, b in w)
    ll_poisson_direct = n_total_events * np.log(mu_check) - mu_check * E_total
    ll_hawkes_alpha0 = -neg_log_likelihood(np.log([mu_check, 1e-12, 1.0]), events, wins)
    log(f"Sanity check (alpha->0 should match pure-Poisson log-likelihood):")
    log(f"  direct Poisson ll = {ll_poisson_direct:.3f}")
    log(f"  Hawkes(alpha~0) ll = {ll_hawkes_alpha0:.3f}")
    log(f"  difference = {abs(ll_poisson_direct - ll_hawkes_alpha0):.4f} (should be ~0)\n")

    # ---------------- Fit ----------------
    x0 = np.log([0.08, 0.3, 1.0])  # mu, alpha, beta initial guess
    t0 = time.time()
    res = minimize(neg_log_likelihood, x0, args=(events, wins), method="L-BFGS-B")
    fit_time = time.time() - t0
    mu_hat, alpha_hat, beta_hat = np.exp(res.x)
    branching_ratio = alpha_hat / beta_hat

    log("--- Hawkes fit (shared mu, alpha, beta across all stars) ---")
    log(f"Converged: {res.success}, {res.nit} iterations, {fit_time:.1f}s")
    log(f"mu_hat    = {mu_hat:.5f} flares/day (background rate)")
    log(f"alpha_hat = {alpha_hat:.5f} flares/day (excitation jump size)")
    log(f"beta_hat  = {beta_hat:.5f} /day  (excitation decay rate; timescale 1/beta = {1/beta_hat:.4f} d = {24/beta_hat:.2f} h)")
    log(f"branching ratio alpha/beta = {branching_ratio:.4f}  ({'STATIONARY' if branching_ratio < 1 else 'NON-STATIONARY - alpha/beta >= 1'})")

    ll_hawkes = -res.fun
    ll_poisson_opt = ll_poisson_direct  # mu_check=0.15 ~= the Poisson script's own MLE (0.14962)
    lr_stat = 2 * (ll_hawkes - ll_poisson_opt)
    lr_pvalue = stats.chi2.sf(lr_stat, df=2)  # Hawkes has 2 more free parameters (alpha, beta)
    aic_poisson = -2 * ll_poisson_opt + 2 * 1
    aic_hawkes = -2 * ll_hawkes + 2 * 3
    log(f"log-likelihood = {ll_hawkes:.2f}  (vs Poisson-only: {ll_poisson_opt:.2f})")
    log(f"Likelihood-ratio test vs Poisson (2 extra params): stat = {lr_stat:.2f}, p = {lr_pvalue:.3e}")
    log(f"AIC: Poisson = {aic_poisson:.1f}, Hawkes = {aic_hawkes:.1f}  (delta = {aic_hawkes - aic_poisson:.1f}, more negative favors Hawkes)")

    # ---------------- Residuals (time-rescaled, Hawkes intensity) ----------------
    wait = pd.read_csv(WAIT_PATH)
    wait["row_in_tic"] = wait.groupby("TIC").cumcount()
    clean = wait[~wait["overlaps_corrupted_orbit"]].copy()
    closing = clean[clean["waiting_time_type"].isin(["initial", "interevent"])].copy()

    z_hawkes = np.empty(len(closing))
    for idx, row in enumerate(closing.itertuples()):
        tic = row.TIC
        t_prev, t_curr = row.t_prev_btjd, row.t_curr_btjd
        ev = events[tic]
        hist = ev[ev <= t_prev]
        win_list = wins.get(tic, [])
        z = 0.0
        for a, b in win_list:
            lo, hi = max(a, t_prev), min(b, t_curr)
            if hi > lo:
                z += mu_hat * (hi - lo)
                z += excitation_integral(lo, hi, hist, alpha_hat, beta_hat)
        z_hawkes[idx] = z
    closing["z_hawkes"] = z_hawkes

    ks_hawkes = stats.kstest(z_hawkes, "expon")
    log(f"\n--- Time-rescaling test (Hawkes) ---")
    log(f"N residuals          = {len(z_hawkes)}")
    log(f"mean(z), var(z)      = {z_hawkes.mean():.4f}, {z_hawkes.var():.4f}  (Exp(1): both = 1)")
    log(f"KS test vs Exp(1)    : D = {ks_hawkes.statistic:.4f}, p = {ks_hawkes.pvalue:.3e}")
    verdict = "REJECTED" if ks_hawkes.pvalue < ALPHA_LEVEL else "not rejected"
    log(f"  -> Hawkes process is {verdict} at alpha={ALPHA_LEVEL}")

    # Independence check
    closing_sorted = closing.sort_values(["TIC", "row_in_tic"]).reset_index(drop=True)
    prev_tic = closing_sorted["TIC"].shift(1)
    prev_row = closing_sorted["row_in_tic"].shift(1)
    prev_z = closing_sorted["z_hawkes"].shift(1)
    adjacent = (closing_sorted["TIC"] == prev_tic) & (closing_sorted["row_in_tic"] - prev_row == 1)
    u_curr = 1 - np.exp(-closing_sorted.loc[adjacent, "z_hawkes"].to_numpy())
    u_prev = 1 - np.exp(-prev_z[adjacent].to_numpy())
    lag1_corr = np.corrcoef(u_prev, u_curr)[0, 1]
    log(f"\nAdjacent same-star pairs = {adjacent.sum()}")
    log(f"Lag-1 Pearson corr of u_i=1-exp(-z_i) = {lag1_corr:.4f}  (0 expected under independence; Poisson baseline was +0.224)")

    closing[["TIC", "waiting_time_type", "t_prev_btjd", "t_curr_btjd", "z_hawkes"]].to_csv(OUT_RESID, index=False)
    log(f"\nWrote {len(closing)} Hawkes time-rescaled residuals to {OUT_RESID}")

    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote summary to {OUT_TXT}")

    # ---------------- Figure: Hawkes vs Poisson residuals, side by side ----------------
    poisson_resid = pd.read_csv(POISSON_RESID_PATH)
    z_poisson = poisson_resid["z_global"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    def survival(z):
        zs = np.sort(z)
        s = 1 - np.arange(1, len(zs) + 1) / len(zs)
        return zs, s

    def qq_theoretical_quantiles(n):
        ranks = np.arange(1, n + 1)
        return -np.log(1 - (ranks - 0.5) / n)

    ax = axes[0, 0]
    zs, s = survival(z_hawkes)
    ax.step(zs, s, where="post", label="Hawkes", color="#2f855a")
    zp, sp = survival(z_poisson)
    ax.step(zp, sp, where="post", label="homogeneous Poisson", color="#c53030", alpha=0.7)
    xlim = max(zs.max(), zp.max()) * 1.1
    zz = np.linspace(0, xlim, 200)
    ax.plot(zz, np.exp(-zz), "k--", label="theoretical Exp(1)")
    ax.set_yscale("log")
    ax.set_xlim(0, xlim)
    ax.set_xlabel("z")
    ax.set_ylabel("survival S(z) = P(Z > z)")
    ax.set_title("Waiting-time survival function")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    n = len(z_hawkes)
    theo_q = qq_theoretical_quantiles(n)
    ax.scatter(theo_q, np.sort(z_hawkes), s=6, alpha=0.4, color="#2f855a")
    lim = max(theo_q.max(), z_hawkes.max())
    ax.plot([0, lim], [0, lim], "k--")
    ax.set_xlabel("theoretical Exp(1) quantile")
    ax.set_ylabel("empirical z quantile (Hawkes)")
    ax.set_title(f"Q-Q plot, Hawkes\nKS D={ks_hawkes.statistic:.3f}, p={ks_hawkes.pvalue:.2e}")

    ax = axes[1, 0]
    n = len(z_poisson)
    theo_q = qq_theoretical_quantiles(n)
    ax.scatter(theo_q, np.sort(z_poisson), s=6, alpha=0.4, color="#c53030")
    lim = max(theo_q.max(), z_poisson.max())
    ax.plot([0, lim], [0, lim], "k--")
    ax.set_xlabel("theoretical Exp(1) quantile")
    ax.set_ylabel("empirical z quantile (Poisson, for reference)")
    ax.set_title("Q-Q plot, homogeneous Poisson (reference)")

    ax = axes[1, 1]
    ax.scatter(u_prev, u_curr, s=6, alpha=0.3, color="#2f855a")
    ax.set_xlabel(r"$u_i = 1-e^{-z_i}$")
    ax.set_ylabel(r"$u_{i+1}$")
    ax.set_title(f"Lag-1 independence check (Hawkes)\ncorr={lag1_corr:.3f} (adjacent same-star pairs)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    print(f"Wrote figure to {OUT_FIG}")


if __name__ == "__main__":
    main()
