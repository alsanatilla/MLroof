from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio import features
from shapely.geometry import Polygon


@dataclass(frozen=True)
class EvalMetrics:
    iou: float
    dice: float
    abs_area_error: float
    rel_area_error: float
    pred_area_m2: float
    gt_area_m2: float


def rasterize_ground_truth(
    footprints_path: str,
    reference_dataset: rasterio.io.DatasetReader,
) -> NDArray[np.bool_]:
    gdf = gpd.read_file(footprints_path)
    if gdf.empty:
        return np.zeros(reference_dataset.shape, dtype=bool)

    if gdf.crs is None:
        raise ValueError("Ground truth vector is missing a CRS definition.")

    gdf = gdf.to_crs(reference_dataset.crs)
    geometries = list(_iter_polygons(gdf.geometry))
    if not geometries:
        return np.zeros(reference_dataset.shape, dtype=bool)

    mask = features.rasterize(
        [(geom, 1) for geom in geometries],
        out_shape=reference_dataset.shape,
        transform=reference_dataset.transform,
        fill=0,
        dtype="uint8",
    )
    return mask.astype(bool)


def compute_metrics(
    pred_mask: NDArray[np.bool_],
    gt_mask: NDArray[np.bool_],
    *,
    pixel_area_m2: float = 1.0,
) -> EvalMetrics:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        iou = 1.0
    else:
        iou = float(intersection) / float(union)

    denom = pred.sum() + gt.sum()
    if denom == 0:
        dice = 1.0
    else:
        dice = float(2 * intersection) / float(denom)

    pred_area_m2 = float(pred.sum()) * pixel_area_m2
    gt_area_m2 = float(gt.sum()) * pixel_area_m2
    abs_area_error = abs(pred_area_m2 - gt_area_m2)
    if gt_area_m2 == 0:
        rel_area_error = 0.0 if pred_area_m2 == 0 else float("inf")
    else:
        rel_area_error = abs_area_error / gt_area_m2

    return EvalMetrics(
        iou=iou,
        dice=dice,
        abs_area_error=abs_area_error,
        rel_area_error=rel_area_error,
        pred_area_m2=pred_area_m2,
        gt_area_m2=gt_area_m2,
    )


def evaluate_masks(
    *,
    pred_mask_path: str,
    ground_truth_path: str,
) -> EvalMetrics:
    with rasterio.open(pred_mask_path) as dataset:
        pred_mask = dataset.read(1).astype(bool)
        pixel_area_m2 = abs(dataset.transform.a * dataset.transform.e)
        gt_mask = rasterize_ground_truth(ground_truth_path, dataset)

    return compute_metrics(pred_mask, gt_mask, pixel_area_m2=pixel_area_m2)


def format_report(metrics: EvalMetrics) -> str:
    rel_area = _format_metric(metrics.rel_area_error)
    return "\n".join(
        [
            "# Roof Area Evaluation Report",
            "",
            "## Metrics",
            f"- IoU: {metrics.iou:.4f}",
            f"- Dice: {metrics.dice:.4f}",
            f"- Predicted area (m²): {metrics.pred_area_m2:.4f}",
            f"- Ground truth area (m²): {metrics.gt_area_m2:.4f}",
            f"- Absolute area error (m²): {metrics.abs_area_error:.4f}",
            f"- Relative area error: {rel_area}",
            "",
        ]
    )


def write_report(metrics: EvalMetrics, report_path: str) -> str:
    report_text = format_report(metrics)
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_text, encoding="utf-8")
    return str(path)


def default_report_path(pred_mask_path: str) -> str:
    base = Path(pred_mask_path)
    return str(base.with_suffix("")) + "_eval_report.md"


def _format_metric(value: float) -> str:
    if np.isinf(value):
        return "inf"
    return f"{value:.4f}"


def _iter_polygons(geometries: Iterable[object]) -> Iterable[Polygon]:
    for geometry in geometries:
        if geometry is None:
            continue
        if geometry.geom_type == "Polygon":
            yield geometry
        elif geometry.geom_type == "MultiPolygon":
            for polygon in geometry.geoms:
                yield polygon
