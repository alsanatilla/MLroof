import pytest

pyproj = pytest.importorskip("pyproj")
from pyproj import CRS, Transformer

from roof_area.io.vector import reproject_bounds
from roof_area.pipeline.run import reproject_aoi_to_raster_crs


def test_reproject_bounds_matches_pyproj():
    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_epsg(3857)
    bounds = (-5.0, 50.0, -4.0, 51.0)

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    xs = [bounds[0], bounds[2], bounds[0], bounds[2]]
    ys = [bounds[1], bounds[1], bounds[3], bounds[3]]
    txs, tys = transformer.transform(xs, ys)
    expected = (min(txs), min(tys), max(txs), max(tys))

    assert reproject_bounds(bounds, src_crs, dst_crs) == expected


def test_reproject_aoi_short_circuit():
    bounds = (0.0, 0.0, 1.0, 1.0)
    crs = CRS.from_epsg(4326)

    assert reproject_aoi_to_raster_crs(bounds, crs, crs) == bounds
