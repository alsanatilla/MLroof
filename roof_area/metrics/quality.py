from __future__ import annotations

import numpy as np


def shadow_score(
    image: np.ndarray,
    mask: np.ndarray,
    darkness_threshold: float = 0.3,
) -> float:
    """Compute the fraction of dark pixels within a roof mask."""
    if image.ndim == 3:
        image = image.mean(axis=-1)

    mask_bool = mask.astype(bool)
    masked_pixels = image[mask_bool]
    if masked_pixels.size == 0:
        return 0.0

    dark_pixels = masked_pixels <= darkness_threshold
    return float(dark_pixels.mean())


def edge_confidence(probability: np.ndarray, mask: np.ndarray) -> float:
    """Compute the mean probability along the mask edge."""
    mask_bool = mask.astype(bool)
    if mask_bool.size == 0:
        return 0.0

    up = np.pad(mask_bool[:-1, :], ((1, 0), (0, 0)), constant_values=False)
    down = np.pad(mask_bool[1:, :], ((0, 1), (0, 0)), constant_values=False)
    left = np.pad(mask_bool[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    right = np.pad(mask_bool[:, 1:], ((0, 0), (0, 1)), constant_values=False)

    interior = mask_bool & up & down & left & right
    edge = mask_bool & ~interior
    if not edge.any():
        return 0.0

    return float(probability[edge].mean())


def quality_flag_shadow(score: float, threshold: float = 0.4) -> bool:
    """Flag roofs with a high shadow ratio."""
    return score >= threshold
