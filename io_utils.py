#io_utils.py
import numpy as np
from config import DTYPE


def load_batches(root, first, last, names):
    """Load averaged .npy files from folders average<first>..average<last>.

    Returns
    -------
    dict[str, ndarray]
        One array per name, with a leading batch axis:
        shape (n_batches, *shape_of_single_file).
    """
    n_batches = last - first + 1
    data = {}
    for name in names:
        first_file = np.load(root / f"average{first}" / f"{name}.npy").astype(DTYPE)
        data[name] = np.empty((n_batches, *first_file.shape), dtype=first_file.dtype)
        data[name][0] = first_file

        for i, batch in enumerate(range(first + 1, last + 1), start=1):
            arr = np.load(root / f"average{batch}" / f"{name}.npy").astype(DTYPE)
            if arr.shape != first_file.shape:
                raise ValueError(
                    f"{name}.npy in average{batch} has shape {arr.shape}, "
                    f"expected {first_file.shape}"
                )
            data[name][i] = arr
    return data

def describe_data(data: dict[str, np.ndarray]) -> None:
    """Print shape, dtype, memory use and NaN count of each array."""
    for name, arr in data.items():
        size_mb = arr.nbytes / 1e6
        n_nan = np.isnan(arr).sum()
        print(f"{name:10s} shape={arr.shape}  dtype={arr.dtype}  "
              f"{size_mb:.1f} MB  NaNs={n_nan}")

def get_n_frames(data: dict[str, np.ndarray], TIME_AXIS) -> int:
    """Number of frames per batch, checked for consistency across arrays.

    Raises
    ------
    ValueError
        If the arrays do not all have the same number of frames.
    """
    n_frames = {name: arr.shape[TIME_AXIS[name]] for name, arr in data.items()}
    if len(set(n_frames.values())) != 1:
        raise ValueError(f"Inconsistent number of frames: {n_frames}")
    
    return next(iter(n_frames.values()))






