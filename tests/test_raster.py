import pytest

np = pytest.importorskip("numpy")
rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin

from roof_area.io.raster import get_pixel_size_m, open_raster, read_window


def _write_test_raster(path, width=10, height=10):
    transform = from_origin(0, 10, 1, 1)
    data = np.arange(width * height, dtype=np.uint8).reshape((height, width))
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        transform=transform,
        crs="EPSG:4326",
    ) as dst:
        dst.write(data, 1)
    return data, transform


def test_open_raster(tmp_path):
    raster_path = tmp_path / "test.tif"
    _write_test_raster(raster_path)

    with open_raster(str(raster_path)) as dataset:
        assert dataset.width == 10
        assert dataset.height == 10


def test_get_pixel_size_m(tmp_path):
    raster_path = tmp_path / "test.tif"
    _, transform = _write_test_raster(raster_path)

    with rasterio.open(raster_path) as dataset:
        pixel_size = get_pixel_size_m(dataset)

    assert pixel_size == (abs(transform.a), abs(transform.e))


def test_read_window(tmp_path):
    raster_path = tmp_path / "test.tif"
    data, _ = _write_test_raster(raster_path)

    with rasterio.open(raster_path) as dataset:
        window, window_data = read_window(dataset, (2, 2, 5, 5))

    assert window.width == 3
    assert window.height == 3
    assert window_data.shape == (1, 3, 3)
    np.testing.assert_array_equal(window_data[0], data[5:8, 2:5])
