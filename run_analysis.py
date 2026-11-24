import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import argparse
from pathlib import Path

from constants import MOM_FACTOR, INTERFACE_Z_A
from config import SimulationParams, Regions, Columns, MATERIALS, FILE_NAMES, TIME_AXIS, V_WALL, T_WALL, BOX
from io_utils import load_batches, describe_data, get_n_frames
from ttcf import ttcf_conversion_factors, mass_per_bin, velocity_profile, velocity_profile_ttcf
from my_stats import ci_halfwidth, frames_from_times, mean_velocity
from plotting import plot_bin_trajectories, plot_mean_with_band, draw_channel_inset, plot_bin_evolution, plot_wall_average, style_velocity_axis, plot_profile_grid, steady_state_figure

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dir",
    type=Path,
    required=True,
    help="Path to the simulation results folder"
)
parser.add_argument(
    "--first",
    type=int,
    default=1,
    help="Index of the first average folder to process (default: 1)"
)

parser.add_argument(
    "--last",
    type=int,
    default=8,
    help="Index of the last average folder to process (default: 8)"
)

args = parser.parse_args()
root = args.data_dir.expanduser()

if args.first < 1 or args.last < args.first:
    parser.error("--first must be >= 1 and --last must be >= --first")

first = args.first
last = args.last

my_run = SimulationParams(v_wall=V_WALL, T_wall=T_WALL, box=BOX) 

data = load_batches(root, first, last, names=FILE_NAMES)

Nfolders = last - first + 1

describe_data(data)

n_timesteps = get_n_frames(data, TIME_AXIS)

time_ns = np.arange(n_timesteps)*my_run.frame_interval_ns

regions = Regions()
regions.show()

idxs = Columns()

#===========================#
# PREPARING VELCOITY PROFILES FROM TTCF

om_factor, vel_ttcf_factor, stress_ttcf_factor = ttcf_conversion_factors(my_run)

mass = {m: mass_per_bin(data[m], my_run, idxs) for m in MATERIALS}
px = {m: data[m][..., idxs.momentum] * MOM_FACTOR for m in MATERIALS}
px_om = {m: data[m][..., idxs.momentum_ttcf] * MOM_FACTOR * om_factor for m in MATERIALS}

dt_fs = my_run.dump_every * my_run.timestep_fs
vx = {m: velocity_profile(px[m], mass[m]) for m in MATERIALS}
vx_ttcf = {m: velocity_profile_ttcf(px[m], px_om[m], mass[m], dt_fs) for m in MATERIALS}


pb = regions.pb
v_ttcf_sum = np.nansum(vx_ttcf["pb"][:, pb], axis=1)                # (batches, frames)
v_ttcf_sum_abs = np.nansum(np.abs(vx_ttcf["pb"][:, pb]), axis=1)
v_dav_sum = np.nansum(vx["pb"][:, pb], axis=1)

err_ttcf = ci_halfwidth(v_ttcf_sum)       # shape (frames,)
err_ttcf_abs = ci_halfwidth(v_ttcf_sum_abs)
err_dav = ci_halfwidth(v_dav_sum)

#===========================#
# PLOT VELCOITY PROFILES FROM TTCF

# CHECK TRAJECTORIES
phase, bin_plot = "pb", 80

fig, ax = plt.subplots(constrained_layout=True)
plot_bin_trajectories(time_ns, vx_ttcf[phase][:, bin_plot, :], ax=ax)
# fig.savefig(out_dir / f"check_traj_bin{bin_plot}.pdf")

# steady-state check: v_ttcf_sum_abs has shape (n_batches, n_frames)
mean_abs = v_ttcf_sum_abs.mean(axis=0)
fig, ax = plt.subplots(constrained_layout=True)
plot_mean_with_band(time_ns, mean_abs, err_ttcf_abs, ax=ax, ylabel="summed |v| over PB bins (m/s)")


# PLOT BIN VELOCITY EVOLUTION

skip = 200   # frames discarded as transient
for label, arr, b in [("TTCF ofm", vx_ttcf["ofm"], 10),
                      ("DAV  ofm", vx["ofm"], 10),
                      ("DAV  hematite", vx["hematite"], 9),
                      ("DAV  hematite", vx["hematite"], 10)]:
    print(f"{label:15s} bin {b:3d}: {mean_velocity(arr, b, skip):.4g} m/s")

time_ps = time_ns * 1000
z = my_run.bin_centers
v0 = my_run.v_wall

fig, (ax_ofm, ax_pb) = plt.subplots(2, 1, figsize=(7, 6.5), constrained_layout=True)

# Panel 1: OFM bins
bins_ofm = [10, 15, 20]
colors = plot_bin_evolution(ax_ofm, time_ps, vx_ttcf["ofm"], bins_ofm, z, cmap="winter")
plot_wall_average(ax_ofm, time_ps, vx_ttcf["hematite"], regions.bottom_wall, "peru", "bot wall")
style_velocity_axis(ax_ofm, ylim=(-1.8 * v0, 0.5 * v0))
draw_channel_inset(ax_ofm, regions, my_run.n_bins, bins_ofm, colors)

# Panel 2: PB bins, with direct-average curves and error bars
bins_pb = [24, 39, 54, 69, 84]
colors = plot_bin_evolution(ax_pb, time_ps, vx_ttcf["pb"], bins_pb, z, cmap="summer", v_dav=None, error_every=1000)
plot_wall_average(ax_pb, time_ps, vx_ttcf["hematite"], regions.bottom_wall, "peru", "bot wall")
plot_wall_average(ax_pb, time_ps, vx_ttcf["hematite"], regions.top_wall, "firebrick", "top wall")
style_velocity_axis(ax_pb, ylim=(-1.8 * v0, 1.8 * v0))
draw_channel_inset(ax_pb, regions, my_run.n_bins, bins_pb, colors)

# fig.savefig(out_dir / "v_bin_evolution.pdf")

# PLOT VELOCITY GRID

shift_ns = 0.01          # offset of the first window
spacing_ns = 0.0005      # distance between window starts
window_ns = 0.0003       # window length
n_panels = 20

times_ns = shift_ns + spacing_ns * np.arange(1, n_panels + 1)
starts = frames_from_times(times_ns, my_run)
width = round(window_ns / my_run.frame_interval_ns)

fig, axs = plot_profile_grid(vx_ttcf, vx, my_run.bin_centers, starts, width, frame_interval_ps=my_run.frame_interval_ns * 1000, materials=MATERIALS)

# fig.savefig(out_dir / "v_profile_grid.pdf")

# PLOT STEADY STATE AND LINEAR FIT

interface = INTERFACE_Z_A - my_run.bin_width / 2   
v0 = my_run.v_wall

jobs = [("TTCF", vx_ttcf, (0.010, 0.200)),
        ("DAV",  vx,      (0.010, 0.200)),
        ("DAV",  vx,      (0.100, 0.200)),
        ("DAV",  vx,      (0.150, 0.200))] # (method, data, (start_time, end_time)) : start and end time are provided in nanoseconds

for method, v, window in jobs:
    fig, fit, result = steady_state_figure(
        v, window, my_run, method, regions.pb, interface,
        v_dav_pb=vx["pb"] if method == "TTCF" else None,
        ylim=(-1.7 * v0, 1.7 * v0),
    )

    print(f"{method} {window} ns: slope = {fit.slope:.3e} +/- {fit.slope_err:.1e}, chi2_red = {fit.chi2_red:.2f}")
    print(result)

plt.show()
