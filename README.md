# velocity_profile

Post-processing of TTCF (transient-time correlation function) shear simulations.

Starting from time- and space-resolved averages written by LAMMPS runs and pre-processed in python, the code computes velocity profiles with and without the TTCF , ensemble averages with confidence intervals, a linear fit of the liquid velocity profile, and the resulting slip length.

Note: the usual ensemle average is referred here as direct average (DAV)

## Contents

- [Requirements](#requirements)
- [Input data format](#input-data-format)
- [Usage](#usage)
- [Outputs](#outputs)
- [Conventions and units](#conventions-and-units)
- [Repository layout](#repository-layout)
- [Status](#status)
- [License](#license)
- [Author](#author)

## Requirements

- Python 3.9 or newer
- numpy, scipy, matplotlib

```bash
pip install -r requirements.txt
```

## Input data format

The code reads NumPy (.npy) files generated during simulation post-processing. Each batch folder (average#) contains statistical averages computed over a block of trajectories.

```
<data_folder>/
├── average1/
│   ├── stress.npy
│   ├── pb.npy
│   ├── ofm.npy
│   └── hematite.npy
├── average2/
│   └── ...
└── averageN/
```

All batches must have the same shapes across folders. The number of frames and bins is read from the files, not set in the code.

### `stress.npy`

Shape `(n_frames, 8)`. The first four columns hold three interfacial forces and the potential energy; the last four are their TTCF counterparts, in the same order.

| Columns | Content |
|---|---|
| 0-3 | force on upper wall, force on bottom wall, PB-OFM interface force, potential energy |
| 4-7 | TTCF versions of columns 0-3 |

### `pb.npy`, `ofm.npy`, `hematite.npy`

Shape `(n_bins, n_frames, 6)`: profiles across the channel (bins along z) for each saved frame.

| Column | Content |
|---|---|
| 0 | number density (1/cm³) |
| 1 | mass density (g/cm³) |
| 2 | kinetic energy |
| 3 | x-momentum |
| 4 | TTCF kinetic energy |
| 5 | TTCF x-momentum |

## Usage

```bash
python run_analysis.py --data-dir /path/to/data_folder --first 1 --last 8
```

The data folder is passed as an argument. Wall velocity, wall temperature, box size, time axis (index of the time coordinate in the data matrixes), file names, materials for profile cisualization and plot styles are defined in `config.py`. 

## Outputs

- Plot of the TTCF velocity from a slected bin for each batch to check the trajectories have independnt behaviors. This can help to spot certain type of 
    correlations. For example, if using the same random seeds for the Langevin thermostats in each trajectory, the displayed velocity may show the same 
    fluctuations for different trajectories.
- Plot of the sum over bins of the absolute value of the fluid velocity confidence intervals.
- Velocity evolution in selected bins versus time, with TTCF (DAV to be added)
- Grid of velocity profiles at different time windows
- Steady-state velocity profiles for each material, with a linear fit of the polymer region. This is time-averaged over a defined time interval. 
    The code must be updated for different time intervals. Velocity profiles are given for the TTCF (average over 10 to 200 ps), and DAV (averge over 10 to 200, 100 to 200 and 150 to 200 ps)
- Linear fits performed over the velocity profiles
- Slip length from the extrapolation of the fit, with a Monte Carlo estimate of its uncertainty

Error bars and bands are 95% confidence intervals over independent trajectories (Student t distribution).

## Conventions and units

- Velocities in m/s, lengths in Å, time axes in ns (figures in ps)
- Mass density in g/cm³
- Unit conversions are collected in `constants.py`
- Empty bins (no atoms of a material) are set to NaN and are not drawn

## Repository layout

| File | Purpose |
|---|---|
| `constants.py` | physical constants and unit conversions |
| `config.py` | run parameters, region definitions, column indices, plot styles |
| `io_utils.py` | loading and checking the `.npy` data |
| `ttcf.py` | TTCF conversion factors and velocity profiles |
| `my_stats.py` | ensemble statistics, linear fit, slip length |
| `plotting.py` | figure functions |
| `run_analysis.py` | entry point |

## Status

Under development. No example data are included yet.

## License

MIT 

## Author

C. Riccardo Civello
