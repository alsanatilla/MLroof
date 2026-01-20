from __future__ import annotations

from typing import Tuple

import geopandas as gpd
import pandas as pd
from pyproj import CRS

from roof_area.io.vector import reproject_bounds
from roof_area.metrics.area import ensure_metric_crs


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


def aggregate_area(
    data: pd.DataFrame | gpd.GeoDataFrame,
    *,
    area_column: str = "area_m2",
    target_crs: CRS | str = "EPSG:3857",
) -> pd.DataFrame:
    """Aggregate areas by building_id when available, otherwise tile_id."""
    if area_column not in data.columns:
        if not isinstance(data, gpd.GeoDataFrame):
            raise KeyError(f"Missing '{area_column}' in data for aggregation.")
        metric_crs = ensure_metric_crs(data.crs, target_crs=target_crs)
        metric_data = data.to_crs(metric_crs)
        data = metric_data.assign(**{area_column: metric_data.geometry.area})

    if "building_id" in data.columns and data["building_id"].notna().any():
        group_column = "building_id"
    elif "tile_id" in data.columns:
        group_column = "tile_id"
    else:
        raise KeyError("Aggregation requires a 'building_id' or 'tile_id' column.")

    return (
        data.groupby(group_column, dropna=False, as_index=False)[area_column]
        .sum()
        .reset_index(drop=True)
    )
