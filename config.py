# config.py
import numpy as np
from dataclasses import dataclass #, field, replace
from constants import A2NM

DTYPE = np.float64 # set float32 if loading data are heavy

FILE_NAMES = ("stress", "pb", "ofm", "hematite")
TIME_AXIS = {"stress": 1, "pb": 2, "ofm": 2, "hematite": 2}
MATERIALS = ("pb", "ofm", "hematite")


STYLE = {
    "pb": ("Polybutene", "tab:orange"),
    "ofm": ("F-SAM", "tab:blue"),
    "hematite": ("Hematite", "tab:green"),
}
NO_ERRORBARS = {"hematite"}

V_WALL = 0.00662495377662124
T_WALL = 318.77
BOX = [55.418, 55.088, 120]

@dataclass(frozen=True) # kw_only=True, only for newer version of Python
class SimulationParams:
    v_wall: float            # m/s
    T_wall: float            # K
    box: tuple[float, float, float]   # Dx, Dy, Dz in Angstrom
    n_bins: int = 100
    n_pb_atoms: int = 28462
    dump_every: int = 10     # steps between saved frames
    timestep_ns: float = 1e-6 
    timestep_fs: float = 1

    @property
    def bin_width(self) -> float:
        return self.box[2] / self.n_bins

    @property
    def bin_volume_A3(self) -> float:
        return self.box[0] * self.box[1] * self.bin_width

    @property
    def bin_volume_cm3(self) -> float:
        return self.bin_volume_A3*(1e-8)**3

    @property
    def bin_centers(self) -> np.ndarray:
        return (np.arange(self.n_bins) + 0.5) * self.bin_width # in A

    @property
    def frame_interval_ns(self) -> float:
        return self.dump_every * self.timestep_ns

    @property
    def frame_interval_fs(self) -> float:
        return self.dump_every * self.timestep_fs


@dataclass(frozen=True)
class Regions:
    """Bin index ranges for each material region."""
    top_wall: range = range(87, 98)
    bottom_wall: range = range(0, 11)
    ofm: range = range(9, 23)
    pb: range = range(23, 87)
    # thermostatted subsets
    top_wall_thermo: range = range(87,95)
    bottom_wall_thermo: range = range(3,11)

    def show(self) -> None:
        """Print every region, one per line."""
        for name, value in vars(self).items():
            print(f"{name:20s} {value}")


@dataclass(frozen=True)
class Columns:
    density: int = 1
    kinetic_energy: int = 2
    momentum: int = 3
    momentum_ttcf: int = 5

@dataclass(frozen=True)
class LinearFit:
    """Result of a weighted linear fit y = slope*x + intercept."""
    slope: float
    intercept: float
    pcov: np.ndarray          # 2x2 covariance matrix of (slope, intercept)
    x: np.ndarray
    y_fit: np.ndarray
    sigma_y: np.ndarray       # standard error of the fitted line at each x
    lower: np.ndarray         # lower edge of the confidence band
    upper: np.ndarray         # upper edge of the confidence band
    chi2_red: float           # reduced chi-square of the fit

    @property
    def slope_err(self) -> float:
        return float(np.sqrt(self.pcov[0, 0]))

    @property
    def intercept_err(self) -> float:
        return float(np.sqrt(self.pcov[1, 1]))
    
@dataclass(frozen=True)
class SlipResult:
    """Slip length and related quantities, all lengths in Angstrom."""
    slip_length: float      # mean of the Monte-Carlo distribution
    std: float              # standard deviation of the slip length
    ci95: tuple             # (2.5%, 97.5%) percentiles
    x0: float               # mean position where v = -v_wall

    def __str__(self) -> str:
        return (f"x0 = {self.x0 * A2NM:.3f} nm\n"
                f"slip length = {self.slip_length * A2NM:.3f} nm\n"
                f"sigma (MC) = {self.std * A2NM:.3f} nm\n"
                f"95% CI = [{self.ci95[0] * A2NM:.3f}, {self.ci95[1] * A2NM:.3f}] nm")


# p = SimulationParams(v_wall=20,T_wall=30,box=[20, 40, 80])
# print(p)
# p2 = replace(p,v_wall = 10)
# print(p2)
