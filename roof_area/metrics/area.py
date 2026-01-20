from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from pyproj import CRS


def mask_area_m2(
    mask: NDArray[np.bool_] | NDArray[np.integer],
    pixel_size_x: float | Tuple[float, float],
    pixel_size_y: float | None = None,
) -> float:
    """Compute the masked area in square meters."""
    if pixel_size_y is None:
        if not isinstance(pixel_size_x, tuple):
            raise ValueError("pixel_size_y must be provided when pixel_size_x is not a tuple.")
        pixel_size_x, pixel_size_y = pixel_size_x

    pixel_count = int(np.count_nonzero(mask))
    return float(pixel_count) * float(pixel_size_x) * float(pixel_size_y)


def ensure_metric_crs(
    crs: CRS | str | None,
    *,
    target_crs: CRS | str = "EPSG:3857",
) -> CRS:
    """Return a metric CRS, reprojecting to a target CRS if needed."""
    if crs is None:
        raise ValueError("CRS is required to compute metric areas.")

    parsed = CRS.from_user_input(crs)
    if _is_metric_crs(parsed):
        return parsed

    return CRS.from_user_input(target_crs)


def _is_metric_crs(crs: CRS) -> bool:
    if not crs.is_projected:
        return False
    for axis in crs.axis_info or []:
        unit_name = (axis.unit_name or "").lower()
        if "metre" in unit_name or "meter" in unit_name:
            return True
    return False
