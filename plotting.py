#plotting.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from my_stats import ci_halfwidth, window_average, time_window_average, steady_profiles, plot_steady_profiles, slip_length_extrapolation, my_linear_fit
from config import STYLE, NO_ERRORBARS


def plot_bin_trajectories(time, v, ax=None):
    """Velocity of one bin versus time, for every batch, plus the ensemble mean.

    Parameters
    ----------
    time : ndarray, shape (n_frames,)
        Time in ns.
    v : ndarray, shape (n_batches, n_frames)
        Velocity of a single bin in m/s, one row per batch.
    ax : matplotlib Axes, optional
        Axes to draw on. A new figure is created if omitted.

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(constrained_layout=True)
    ax.plot(time, v.T, alpha=0.6)
    ax.plot(time, np.nanmean(v, axis=0), color="black", linewidth=2, label="average")
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("v (m/s)")
    ax.grid()
    return ax


def plot_mean_with_band(time, mean, err, ax=None, ylabel="v (m/s)"):
    """Plot a mean curve with a shaded confidence band.

    mean, err : ndarray, shape (n_frames,)
        err is the half-width of the confidence interval.
    """
    if ax is None:
        _, ax = plt.subplots(constrained_layout=True)
    ax.fill_between(time, mean - err, mean + err, alpha=0.2)
    ax.plot(time, mean, color="black", linewidth=1)
    ax.set_xlabel("time (ns)")
    ax.set_ylabel(ylabel)
    ax.grid()
    return ax


def draw_channel_inset(ax, regions, n_bins, marker_bins, marker_colors):
    """Draw a small schematic of the channel with a dot at each plotted bin.

    Region boundaries are drawn from the first to the end of the last bin
    of each region, in units of the box length (0 to 1).
    """
    inset = inset_axes(ax, width="40%", height="10%", loc="upper left",
                       bbox_to_anchor=(0, -0.001, 1, 1),
                       bbox_transform=ax.transAxes, borderpad=0)

    def edges(region):
        return region[0] / n_bins, (region[-1] + 1) / n_bins

    segments = [
        ((regions.ofm[-1] + 1) / n_bins, regions.top_wall[0] / n_bins, "black"),
        (*edges(regions.top_wall), "firebrick"),
        (*edges(regions.ofm), "deepskyblue"),
        (*edges(regions.bottom_wall), "peru"),
    ]
    for x0, x1, color in segments:
        inset.hlines(0.5, x0, x1, color=color, linewidth=2)

    for b, color in zip(marker_bins, marker_colors):
        inset.plot((b + 0.5) / n_bins, 0.5, "o", color=color, markersize=5)

    inset.set_xticks([])
    inset.set_yticks([])
    return inset


def plot_bin_evolution(ax, time_ps, v_ttcf, bins, z_centers, cmap="winter", v_dav=None, error_every=None):
    """Ensemble-averaged TTCF velocity versus time for selected bins.

    Parameters
    ----------
    v_ttcf : ndarray, shape (n_batches, n_bins, n_frames), m/s
    bins : list of int
        Bin indices to plot.
    z_centers : ndarray
        Bin centres in Angstrom (used for the legend).
    v_dav : ndarray, optional
        Direct-average velocity, same shape, drawn as thin blue lines.
    error_every : int, optional
        If given, draw 95% confidence error bars every `error_every` frames.

    Returns
    -------
    colors : ndarray
        One colour per bin (reuse them for the inset markers).
    """
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(bins)))
    for b, color in zip(bins, colors):
        mean = np.nanmean(v_ttcf[:, b, :], axis=0)
        ax.plot(time_ps, mean, color=color, label=f"{z_centers[b]:.0f}")

        if error_every:
            sel = slice(None, None, error_every)
            err = ci_halfwidth(v_ttcf[:, b, sel])
            ax.errorbar(time_ps[sel], mean[sel], yerr=err, color=color, capsize=2, linestyle="none")

        if v_dav is not None:
            ax.plot(time_ps, np.nanmean(v_dav[:, b, :], axis=0), color="tab:blue", linewidth=0.7, alpha=0.6)
    return colors


def plot_wall_average(ax, time_ps, v, region, color, label):
    """Dashed line: velocity averaged over batches and over the wall bins."""
    mean = np.nanmean(v[:, list(region), :], axis=(0, 1))
    ax.plot(time_ps, mean, linestyle="dashed", color=color,
            linewidth=2, label=label)


def style_velocity_axis(ax, ylim, fontsize=12):
    """Common labels, limits and legend placement for the velocity panels."""
    ax.set_xlabel("time (ps)", fontsize=fontsize)
    ax.set_ylabel("velocity (m/s)", fontsize=fontsize)
    ax.set_ylim(*ylim)
    ax.grid()
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.legend(fontsize=10, loc="center right",
              bbox_to_anchor=(1.24, 0.5), borderaxespad=0.1)
    ax.yaxis.set_label_coords(-0.13, 0.5)


def plot_profile_grid(v_ttcf, v_dav, z, start_frames, width_frames,
                      frame_interval_ps, materials,
                      ncols=5, figsize=(14, 8)):
    """Grid of velocity profiles, one panel per time window.

    Each panel shows, for every material, the TTCF profile with a 95%
    confidence band over batches (coloured) and the direct average (black).

    Parameters
    ----------
    v_ttcf, v_dav : dict[str, ndarray]
        Velocities, shape (n_batches, n_bins, n_frames), in m/s.
    z : ndarray, shape (n_bins,)
        Bin centres in Angstrom.
    start_frames : list of int
        First frame of each window (one panel each).
    width_frames : int
        Window length in frames.
    frame_interval_ps : float
        Time between saved frames, used for the panel titles.
    materials : tuple[str]
        Tuple containing the material identifiers (keywords) used to select
        the corresponding data entries for plotting.
    """
    nrows = -(-len(start_frames) // ncols)          # ceiling division
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize,
                            sharex=True, sharey=True, constrained_layout=True)
    axs = np.atleast_1d(axs).ravel()

    for ax, start in zip(axs, start_frames):
        for m in materials:
            win_ttcf = window_average(v_ttcf[m], start, width_frames)   # (batches, bins)
            win_dav = window_average(v_dav[m], start, width_frames)
            ax.errorbar(z, win_ttcf.mean(axis=0), yerr=ci_halfwidth(win_ttcf),
                        label=m, capsize=3, alpha=0.8)
            ax.plot(z, win_dav.mean(axis=0), color="black", alpha=0.8)

        t0 = start * frame_interval_ps
        t1 = (start + width_frames) * frame_interval_ps
        ax.set_title(f"[{t0:.1f}, {t1:.1f}] ps")
        ax.grid()

    for ax in axs[len(start_frames):]:               # hide unused panels
        ax.set_visible(False)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center")
    fig.supxlabel("z (Å)")
    fig.supylabel("velocity (m/s)")
    return fig, axs


def plot_fit_band(ax, x, y_fit, y_lower, y_upper):
    h, = ax.plot(x, y_fit, color="red")
    ax.fill_between(x, y_lower, y_upper, alpha=0.4, color="red")
    return h, "linear fit"

def steady_state_figure(v, window_ns, params, method, fit_bins, interface, v_dav_pb=None, ylim=None):
    """Steady-state velocity profile with a linear fit of the liquid region.

    Parameters
    ----------
    v : dict[str, ndarray]
        Velocities, shape (n_batches, n_bins, n_frames), in m/s.
    window_ns : (float, float)
        Start and end of the averaging window.
    method : str
        "TTCF" or "DAV", used in the legend.
    fit_bins : range, slice or ndarray
        Bins included in the fit.
    interface : float
        Position (A) passed to the slip-length extrapolation.
    v_dav_pb : ndarray, optional
        If given, the DAV PB profile is overlaid as a thin line.

    Returns
    -------
    fig : matplotlib Figure
    slip_length : whatever slip_length_extrapolation returns
    """
    t0, t1 = window_ns
    z = params.bin_centers
    n_batches = v["pb"].shape[0]

    profiles = steady_profiles(v, t0, t1, params)
    fig, ax = plt.subplots(constrained_layout=True)
    handles, labels = plot_steady_profiles(ax, z, profiles, method, n_batches, STYLE, NO_ERRORBARS)

    if v_dav_pb is not None:
        mean_dav = time_window_average(v_dav_pb, t0, t1, params).mean(axis=0)
        h, = ax.plot(z, mean_dav, color="tab:blue", alpha=0.6, linewidth=0.7)
        handles.append(h)
        labels.append("Polybutene (DAV)")

    # linear fit on the PB bins, skipping empty (NaN) bins
    x = z[fit_bins]
    y, y_err = (arr[fit_bins] for arr in profiles["pb"])
    ok = np.isfinite(y) & np.isfinite(y_err)
    x, y, y_err = x[ok], y[ok], y_err[ok]

    fit = my_linear_fit(x, y, y_err)
    result = slip_length_extrapolation(fit.slope, fit.intercept, fit.pcov, params.v_wall, interface)

    h, lab = plot_fit_band(ax, fit.x, fit.y_fit, fit.lower, fit.upper)
    handles.append(h)
    labels.append(lab)
    # ax.axvline(x[0], linestyle="dashed", color="black", alpha=0.8)
    # ax.axvline(x[-1], linestyle="dashed", color="black", alpha=0.8)

    ax.set_xlabel("z (Å)", fontsize=12)
    ax.set_ylabel("v (m/s)", fontsize=12)
    ax.set_title(f"Time average over [{t0 * 1e3:.0f}, {t1 * 1e3:.0f}] ps")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid()
    ax.legend(handles, labels)
    return fig, fit, result