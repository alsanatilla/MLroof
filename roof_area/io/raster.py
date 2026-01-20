from __future__ import annotations

from typing import Tuple

import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.windows import Window, from_bounds


Bounds = Tuple[float, float, float, float]


def open_raster(path: str) -> rasterio.io.DatasetReader:
    """Open a raster dataset and return the rasterio dataset reader."""
    return rasterio.open(path)


def get_pixel_size_m(dataset: rasterio.io.DatasetReader) -> Tuple[float, float]:
    """Return the absolute pixel size (x, y) in meters for a dataset."""
    transform = dataset.transform
    return abs(transform.a), abs(transform.e)


def read_window(
    dataset: rasterio.io.DatasetReader,
    bounds_crs: Bounds,
) -> Tuple[Window, NDArray[np.generic]]:
    """Read a window from the dataset given bounds in the dataset CRS."""
    window = from_bounds(*bounds_crs, transform=dataset.transform)
    data = dataset.read(window=window)
    return window, data
