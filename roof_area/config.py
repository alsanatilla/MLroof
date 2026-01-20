"""Configuration for roof area inference."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RoofAreaSettings(BaseSettings):
    """Application settings with environment overrides."""

    model_config = SettingsConfigDict(env_prefix="ROOF_AREA_", case_sensitive=False)

    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Probability threshold")
    tile_size: int = Field(512, ge=64, description="Tile size in pixels")
    overlap: int = Field(32, ge=0, description="Tile overlap in pixels")
    min_area_m2: float = Field(5.0, ge=0.0, description="Minimum roof area in m²")
    seed: int = Field(42, description="Random seed")
    log_level: str = Field("INFO", description="Logging level")
    raster_path: str | None = Field(None, description="Path to input raster")
    footprints_path: str | None = Field(
        None, description="Path to building footprint vector data"
    )
    output_path: str | None = Field(None, description="Path to output mask raster")
    model_path: str | None = Field(
        None, description="Optional path to a trained ML model"
    )
    pred_mask_path: str | None = Field(None, description="Path to predicted mask")
    ground_truth_path: str | None = Field(
        None, description="Path to ground truth vector data"
    )
    report_path: str | None = Field(None, description="Path to evaluation report")
