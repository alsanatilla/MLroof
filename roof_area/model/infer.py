from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import Polygon


class InferenceError(RuntimeError):
    """Raised when inference cannot be executed."""


def run_inference(
    *,
    raster_path: str,
    footprints_path: str | None,
    output_path: str | None,
    model_path: str | None,
    threshold: float,
    logger: logging.Logger | None = None,
) -> str:
    """Run inference using either a baseline or a model-defined pipeline."""
    if logger is None:
        logger = logging.getLogger(__name__)

    if model_path:
        return _run_model_inference(
            raster_path=raster_path,
            footprints_path=footprints_path,
            output_path=output_path,
            model_path=model_path,
            threshold=threshold,
            logger=logger,
        )

    return _run_baseline_inference(
        raster_path=raster_path,
        footprints_path=footprints_path,
        output_path=output_path,
        threshold=threshold,
        logger=logger,
    )


def _run_model_inference(
    *,
    raster_path: str,
    footprints_path: str | None,
    output_path: str | None,
    model_path: str,
    threshold: float,
    logger: logging.Logger,
) -> str:
    """Placeholder for ML-model-based inference."""
    raise NotImplementedError(
        "ML model inference is not implemented yet. "
        "Omit --model to use the heuristic baseline or implement a UNet/DeepLab runner."
    )


def _run_baseline_inference(
    *,
    raster_path: str,
    footprints_path: str | None,
    output_path: str | None,
    threshold: float,
    logger: logging.Logger,
) -> str:
    if not footprints_path:
        raise InferenceError(
            "No building footprints provided. Provide --footprints to run the baseline "
            "or train a model and pass --model."
        )

    output_path = output_path or _default_output_path(raster_path)

    with rasterio.open(raster_path) as dataset:
        image = dataset.read()
        grayscale = _to_grayscale(image)
        gradient_mask = _gradient_threshold_mask(grayscale, threshold)
        footprint_mask = _mask_by_footprints(
            dataset=dataset,
            footprints_path=footprints_path,
            base_mask=gradient_mask,
            logger=logger,
        )

        profile = dataset.profile.copy()
        profile.update(dtype=rasterio.uint8, count=1)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(footprint_mask.astype(rasterio.uint8), 1)

    logger.info("Baseline inference saved mask to %s", output_path)
    return output_path


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a multi-band image to grayscale for gradient analysis."""
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.shape[0] == 1:
        return image[0].astype(np.float32)

    channels = image[:3].astype(np.float32)
    grayscale = np.mean(channels, axis=0)
    return grayscale


def _gradient_threshold_mask(image: np.ndarray, threshold: float) -> np.ndarray:
    """Compute a gradient-based mask from grayscale imagery."""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)

    if np.max(magnitude) > 0:
        magnitude = (magnitude / np.max(magnitude)) * 255.0

    threshold_value = np.clip(threshold, 0.0, 1.0) * 255.0
    _, mask = cv2.threshold(magnitude, threshold_value, 255.0, cv2.THRESH_BINARY)
    return mask.astype(bool)


def _mask_by_footprints(
    *,
    dataset: rasterio.io.DatasetReader,
    footprints_path: str,
    base_mask: np.ndarray,
    logger: logging.Logger,
) -> np.ndarray:
    gdf = gpd.read_file(footprints_path)
    if gdf.empty:
        raise InferenceError(
            "No building footprints found in the provided file. "
            "Provide footprints or train a model and pass --model."
        )

    if gdf.crs is None:
        raise InferenceError("Building footprints are missing a CRS definition.")

    gdf = gdf.to_crs(dataset.crs)
    output_mask = np.zeros(base_mask.shape, dtype=bool)

    for geom in _iter_polygons(gdf.geometry):
        geometry_mask = features.geometry_mask(
            [geom],
            out_shape=base_mask.shape,
            transform=dataset.transform,
            invert=True,
        )
        output_mask |= base_mask & geometry_mask

    logger.info("Applied baseline mask to %d building footprints", len(gdf))
    return output_mask


def _iter_polygons(geometries: Iterable[object]) -> Iterable[Polygon]:
    for geometry in geometries:
        if geometry is None:
            continue
        if geometry.geom_type == "Polygon":
            yield geometry
        elif geometry.geom_type == "MultiPolygon":
            for polygon in geometry.geoms:
                yield polygon


def _default_output_path(raster_path: str) -> str:
    base = Path(raster_path)
    return str(base.with_suffix("")) + "_roof_mask.tif"
