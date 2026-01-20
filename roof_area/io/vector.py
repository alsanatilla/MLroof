from __future__ import annotations

from typing import Tuple

from pyproj import CRS, Transformer


Bounds = Tuple[float, float, float, float]


def reproject_bounds(bounds: Bounds, src_crs: CRS | str, dst_crs: CRS | str) -> Bounds:
    """Reproject bounding box coordinates from src_crs to dst_crs."""
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    minx, miny, maxx, maxy = bounds

    xs = [minx, maxx, minx, maxx]
    ys = [miny, miny, maxy, maxy]
    txs, tys = transformer.transform(xs, ys)

    return min(txs), min(tys), max(txs), max(tys)
