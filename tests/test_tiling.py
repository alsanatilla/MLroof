import pytest

np = pytest.importorskip("numpy")
rasterio = pytest.importorskip("rasterio")
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from roof_area.preprocess.tiling import iter_windows


def _make_dataset(width=5, height=4):
    data = np.arange(width * height, dtype=np.uint8).reshape((height, width))
    transform = from_origin(0, height, 1, 1)

    memfile = MemoryFile()
    dataset = memfile.open(
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        transform=transform,
        crs="EPSG:4326",
    )
    dataset.write(data, 1)
    return memfile, dataset


def test_iter_windows_tile_size_and_overlap():
    memfile, dataset = _make_dataset()
    try:
        windows = list(iter_windows(dataset, tile_size=3, overlap=1))
    finally:
        dataset.close()
        memfile.close()

    assert windows[0].width == 3
    assert windows[0].height == 3
    assert windows[-1].width == 2
    assert windows[-1].height == 1


def test_iter_windows_overlap_validation():
    memfile, dataset = _make_dataset()
    try:
        try:
            list(iter_windows(dataset, tile_size=3, overlap=3))
        except ValueError as exc:
            assert "overlap" in str(exc)
        else:
            raise AssertionError("Expected ValueError for excessive overlap")
    finally:
        dataset.close()
        memfile.close()
