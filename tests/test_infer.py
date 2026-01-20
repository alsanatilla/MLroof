import numpy as np
import pytest
import rasterio
import geopandas
from shapely import geometry as shapely_geometry

from rasterio.transform import from_origin

from roof_area.model.infer import InferenceError, run_inference


@pytest.fixture()
def sample_raster(tmp_path):
    path = tmp_path / "image.tif"
    data = np.zeros((1, 10, 10), dtype=np.uint8)
    data[0, 3:7, 3:7] = 255
    transform = from_origin(0, 10, 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype=data.dtype,
        crs="EPSG:3857",
        transform=transform,
    ) as dataset:
        dataset.write(data)
    return path


@pytest.fixture()
def sample_footprints(tmp_path):
    path = tmp_path / "footprints.geojson"
    geometry = [shapely_geometry.box(0, 0, 5, 10)]
    gdf = geopandas.GeoDataFrame({"id": [1]}, geometry=geometry, crs="EPSG:3857")
    gdf.to_file(path, driver="GeoJSON")
    return path


def test_baseline_inference_writes_mask(sample_raster, sample_footprints, tmp_path):
    output_path = tmp_path / "mask.tif"
    result = run_inference(
        raster_path=str(sample_raster),
        footprints_path=str(sample_footprints),
        output_path=str(output_path),
        model_path=None,
        threshold=0.0,
    )

    assert result == str(output_path)
    with rasterio.open(output_path) as dataset:
        output = dataset.read(1)
        assert output.shape == (10, 10)
        assert output.dtype == np.uint8
        assert set(np.unique(output)).issubset({0, 1})

        geometry = [shapely_geometry.box(0, 0, 5, 10)]
        outside = rasterio.features.geometry_mask(
            geometry,
            out_shape=output.shape,
            transform=dataset.transform,
            invert=False,
        )
        assert np.all(output[outside] == 0)


def test_baseline_requires_footprints(sample_raster):
    with pytest.raises(InferenceError):
        run_inference(
            raster_path=str(sample_raster),
            footprints_path=None,
            output_path=None,
            model_path=None,
            threshold=0.5,
        )


def test_model_path_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        run_inference(
            raster_path="dummy.tif",
            footprints_path=None,
            output_path=None,
            model_path="model.pt",
            threshold=0.5,
        )
