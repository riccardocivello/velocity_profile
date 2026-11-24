#my_stats.py
from scipy import stats
from scipy.optimize import curve_fit
import numpy as np
import warnings
from config import SlipResult, LinearFit, MATERIALS


def ci_halfwidth(samples, axis=0, confidence=0.95):
    """Half-width of the confidence interval of the mean over independent samples.

    Uses the Student t distribution and the sample standard deviation (ddof=1).
    Assumes the entries along `axis` are independent (here: separate trajectories).
    """
    n = samples.shape[axis]
    sem = samples.std(axis=axis, ddof=1) / np.sqrt(n)
    return stats.t.ppf(0.5 + confidence / 2, df=n - 1) * sem

def linear(x, a, b):
    """Linear function"""
    return a*x + b

def mean_velocity(v, bin_idx, skip_frames):
    """Mean velocity of one bin over batches and frames after the transient."""
    return np.nanmean(v[:, bin_idx, skip_frames:])

def frames_from_times(times_ns, params):
    """Convert times in ns to frame indices."""
    return [round(t / params.frame_interval_ns) for t in times_ns]

def window_average(v, start, width):
    """Average over the frames start ... start+width-1.

    Parameters
    ----------
    v : ndarray, shape (n_batches, n_bins, n_frames)
    start, width : int
        First frame and number of frames of the window.

    Returns
    -------
    ndarray, shape (n_batches, n_bins)

    Raises
    ------
    ValueError
        If the window extends beyond the last frame.
    """
    if start + width > v.shape[-1]:
        raise ValueError(
            f"Window [{start}, {start + width}) exceeds the {v.shape[-1]} frames available"
        )
    return v[..., start:start + width].mean(axis=-1)


def standard_error(samples, axis=0):
    """Standard error of the mean over independent samples (ddof=1)."""
    return samples.std(axis=axis, ddof=1) / np.sqrt(samples.shape[axis])

def time_window_average(v, t0_ns, t1_ns, params):
    """Average of v over the frames between t0_ns and t1_ns.

    v : ndarray, shape (n_batches, n_bins, n_frames)
    Returns ndarray, shape (n_batches, n_bins).
    """
    start = round(t0_ns / params.frame_interval_ns)
    stop = round(t1_ns / params.frame_interval_ns)
    return window_average(v, start, stop - start)


def steady_profiles(v, t0_ns, t1_ns, params, materials=MATERIALS):
    """Window-averaged profile of each material, summarised over batches.

    Returns dict: material -> (mean, sem), each of shape (n_bins,).
    """
    out = {}
    for m in materials:
        win = time_window_average(v[m], t0_ns, t1_ns, params)   # (batches, bins)
        out[m] = (win.mean(axis=0), standard_error(win, axis=0))
    return out

def t_factor(n, confidence=0.95):
    """Student-t multiplier that turns a standard error into a confidence half-width."""
    return stats.t.ppf(0.5 + confidence / 2, df=n - 1)

def plot_steady_profiles(ax, z, profiles, method, n_batches, style, no_errorbars):
    """One profile per material; error bars are 95% confidence intervals.

    Returns (handles, labels) so the caller can build the legend.
    """
    mult = t_factor(n_batches)
    handles, labels = [], []
    for m, (mean, sem) in profiles.items():
        name, color = style[m]
        if m in no_errorbars:
            h, = ax.plot(z, mean, color=color, alpha=0.8)
        else:
            h = ax.errorbar(z, mean, yerr=mult * sem, color=color,
                            capsize=3, alpha=0.8).lines[0]
        handles.append(h)
        labels.append(f"{name} ({method})")
    return handles, labels

def my_linear_fit(x, y, y_err, confidence=0.95):
    """Weighted linear fit of a velocity profile.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Positions across the channel, in Angstrom.
    y : ndarray, shape (n,)
        Velocity at each position, in m/s.
    y_err : ndarray, shape (n,)
        Standard error of each y value (absolute uncertainty, not rescaled).
    confidence : float, optional
        Level of the confidence band of the fitted line. Default 0.95.

    Returns
    -------
    LinearFit
        Slope (m/s per Angstrom), intercept (m/s), covariance matrix, fitted
        values, confidence band and reduced chi-square.

    Raises
    ------
    ValueError
        If fewer than 3 points are given or any y_err is not positive and finite.
    """
    x, y, y_err = np.asarray(x), np.asarray(y), np.asarray(y_err)
    if len(x) < 3:
        raise ValueError("At least 3 points are needed for a fit with errors")
    if not np.all(np.isfinite(y_err) & (y_err > 0)):
        raise ValueError("y_err must be positive and finite (empty bins must be removed)")

    (a, b), pcov = curve_fit(linear, x, y, sigma=y_err, absolute_sigma=True)

    y_fit = linear(x, a, b)
    sigma_y = np.sqrt(x**2 * pcov[0, 0] + pcov[1, 1] + 2 * x * pcov[0, 1])

    dof = len(x) - 2
    mult = stats.t.ppf(0.5 + confidence / 2, df=dof)
    chi2_red = np.sum(((y - y_fit) / y_err) ** 2) / dof

    return LinearFit(a, b, pcov, x, y_fit, sigma_y,
                     y_fit - mult * sigma_y, y_fit + mult * sigma_y, chi2_red)

 
def slip_length_extrapolation(a, b, pcov, v_wall, interface,
                              n_samples=20000, seed=0):
    """Slip length from a linear fit v(z) = a*z + b, by Monte-Carlo sampling.

    The fit parameters are drawn from a multivariate normal distribution with
    the fitted covariance. For each draw, x0 is the position where the profile
    reaches -v_wall, and the slip length is interface - x0.

    The slip length is extrapolated at the bottom wall of the system of whihc 
    velocity is assumed to be negatibe.

    Parameters
    ----------
    a, b : float
        Slope (m/s per Angstrom) and intercept (m/s) of the fit.
    pcov : ndarray, shape (2, 2)
        Covariance matrix of (a, b).
    v_wall : float
        Wall speed in m/s (the profile is extrapolated to -v_wall).
    interface : float
        Position of the interface in Angstrom.
    n_samples, seed : int
        Size and random seed of the Monte-Carlo sample.

    Returns
    -------
    SlipResult
    """
    rng = np.random.default_rng(seed)
    a_s, b_s = rng.multivariate_normal([a, b], pcov, size=n_samples).T

    if np.mean(np.sign(a_s) != np.sign(a)) > 1e-3:
        warnings.warn("The fitted slope is compatible with zero: the slip "
                      "length distribution has heavy tails and its mean and "
                      "standard deviation are unreliable.")

    x0_s = (-v_wall - b_s) / a_s
    lb_s = interface - x0_s

    return SlipResult(
        slip_length=lb_s.mean(),
        std=lb_s.std(ddof=1),
        ci95=tuple(np.percentile(lb_s, [2.5, 97.5])),
        x0=x0_s.mean(),
    )

