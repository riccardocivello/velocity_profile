import numpy as np
from scipy.integrate import cumulative_trapezoid

from constants import FS2S, K_B, G_MOL2KG, A_FS2M_S, KCAL_MOL_A2N, G2KG

from config import SimulationParams

def ttcf_conversion_factors(params: SimulationParams) -> tuple[float, float, float]:
    om_factor = params.v_wall * KCAL_MOL_A2N / (params.T_wall * K_B)
    vel_ttcf_factor = FS2S * om_factor * (G_MOL2KG * A_FS2M_S)
    stress_ttcf_factor = FS2S * om_factor * KCAL_MOL_A2N
    return om_factor, vel_ttcf_factor, stress_ttcf_factor

def mass_per_bin(raw: np.ndarray, params, cols) -> np.ndarray:
    """Mass in each bin and frame, in kg.

    raw : shape (n_batches, n_bins, n_frames, n_cols), density in g/cm3.
    """
    return raw[..., cols.density] * (params.bin_volume_cm3 * G2KG)

def velocity_profile(momentum, mass, min_mass=1e-25):
    """v = p / m in each bin and frame (m/s); NaN in bins that are empty.

    momentum, mass : shape (n_batches, n_bins, n_frames), SI units.
    min_mass : bins whose time-averaged mass is below this (kg) are skipped.
    """
    occupied = mass.mean(axis=-1, keepdims=True) > min_mass   # (batches, bins, 1)
    vel = np.full(momentum.shape, np.nan)
    np.divide(momentum, mass, out=vel, where=occupied)
    return vel

def velocity_profile_ttcf(momentum, momentum_om, mass, dt_fs, min_mass=1e-25):
    """TTCF velocity: (p(0) - integral of the correlation) / m, in m/s."""
    integral = cumulative_trapezoid(momentum_om, dx=dt_fs, axis=-1, initial=0) * FS2S
    return velocity_profile(momentum[..., :1] - integral, mass, min_mass)
