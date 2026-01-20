"""Command line interface for roof area workflows."""

from __future__ import annotations

import argparse
from typing import Sequence

from roof_area.config import RoofAreaSettings
from roof_area.logging import configure_logging
from roof_area.metrics.eval import default_report_path, evaluate_masks, write_report
from roof_area.model.infer import run_inference


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--threshold", type=float, help="Probability threshold")
    parser.add_argument("--tile-size", type=int, help="Tile size in pixels")
    parser.add_argument("--overlap", type=int, help="Tile overlap in pixels")
    parser.add_argument("--min-area-m2", type=float, help="Minimum roof area in m²")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--log-level", type=str, help="Logging level")


def _build_settings(args: argparse.Namespace) -> RoofAreaSettings:
    data = {k: v for k, v in vars(args).items() if v is not None}
    return RoofAreaSettings(**data)


def _infer_command(args: argparse.Namespace) -> int:
    settings = _build_settings(args)
    logger = configure_logging(settings.log_level, "roof_area.infer")
    logger.info("Running inference with settings: %s", settings.model_dump())
    if not settings.raster_path:
        raise ValueError("Missing --raster for inference.")

    run_inference(
        raster_path=settings.raster_path,
        footprints_path=settings.footprints_path,
        output_path=settings.output_path,
        model_path=settings.model_path,
        threshold=settings.threshold,
        logger=logger,
    )
    return 0


def _eval_command(args: argparse.Namespace) -> int:
    settings = _build_settings(args)
    logger = configure_logging(settings.log_level, "roof_area.eval")
    logger.info("Running evaluation with settings: %s", settings.model_dump())
    if not settings.pred_mask_path:
        raise ValueError("Missing --pred-mask for evaluation.")
    if not settings.ground_truth_path:
        raise ValueError("Missing --ground-truth for evaluation.")

    metrics = evaluate_masks(
        pred_mask_path=settings.pred_mask_path,
        ground_truth_path=settings.ground_truth_path,
    )
    report_path = settings.report_path or default_report_path(
        settings.pred_mask_path
    )
    write_report(metrics, report_path)
    logger.info("Evaluation report saved to %s", report_path)
    return 0


def _train_command(args: argparse.Namespace) -> int:
    settings = _build_settings(args)
    logger = configure_logging(settings.log_level, "roof_area.train")
    logger.info("Running training with settings: %s", settings.model_dump())
    logger.info("TODO: implement training pipeline")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="roof-area", description="Roof area pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer_parser = subparsers.add_parser("infer", help="Run inference")
    _add_common_args(infer_parser)
    infer_parser.add_argument("--raster", dest="raster_path", type=str, help="Input raster path")
    infer_parser.add_argument(
        "--footprints",
        dest="footprints_path",
        type=str,
        help="Building footprint vector path",
    )
    infer_parser.add_argument(
        "--output",
        dest="output_path",
        type=str,
        help="Output mask path (GeoTIFF)",
    )
    infer_parser.add_argument(
        "--model",
        dest="model_path",
        type=str,
        help="Optional model path to enable ML inference",
    )
    infer_parser.set_defaults(func=_infer_command)

    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    _add_common_args(eval_parser)
    eval_parser.add_argument(
        "--pred-mask",
        dest="pred_mask_path",
        type=str,
        help="Predicted mask raster path",
    )
    eval_parser.add_argument(
        "--ground-truth",
        dest="ground_truth_path",
        type=str,
        help="Ground truth vector path",
    )
    eval_parser.add_argument(
        "--report",
        dest="report_path",
        type=str,
        help="Output report markdown path",
    )
    eval_parser.set_defaults(func=_eval_command)

    train_parser = subparsers.add_parser("train", help="Run training")
    _add_common_args(train_parser)
    train_parser.set_defaults(func=_train_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
