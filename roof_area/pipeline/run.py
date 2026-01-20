from __future__ import annotations

from typing import Tuple

from pyproj import CRS

from roof_area.io.vector import reproject_bounds


Bounds = Tuple[float, float, float, float]


def reproject_aoi_to_raster_crs(
    aoi_bounds: Bounds,
    aoi_crs: CRS | str,
    raster_crs: CRS | str,
) -> Bounds:
    """Reproject AOI bounds into the raster CRS."""
    if CRS.from_user_input(aoi_crs) == CRS.from_user_input(raster_crs):
        return aoi_bounds

    return reproject_bounds(aoi_bounds, aoi_crs, raster_crs)
