from __future__ import annotations

from typing import Generator, Tuple

import rasterio
from rasterio.windows import Window


TileSize = Tuple[int, int]


def _normalize_tile_size(tile_size: int | TileSize) -> TileSize:
    if isinstance(tile_size, int):
        return tile_size, tile_size
    return tile_size


def iter_windows(
    dataset: rasterio.io.DatasetReader,
    tile_size: int | TileSize,
    overlap: int = 0,
) -> Generator[Window, None, None]:
    """Iterate over raster windows without loading the full raster into memory."""
    tile_width, tile_height = _normalize_tile_size(tile_size)
    if overlap >= tile_width or overlap >= tile_height:
        raise ValueError("overlap must be smaller than tile dimensions")

    step_x = tile_width - overlap
    step_y = tile_height - overlap

    for y in range(0, dataset.height, step_y):
        for x in range(0, dataset.width, step_x):
            width = min(tile_width, dataset.width - x)
            height = min(tile_height, dataset.height - y)
            yield Window(col_off=x, row_off=y, width=width, height=height)
