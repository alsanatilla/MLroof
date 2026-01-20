import numpy as np

from roof_area.metrics.quality import shadow_score


def test_shadow_score_dark_pixels_higher():
    mask = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    image_low_shadow = np.array(
        [
            [0.9, 0.9, 0.9, 0.9],
            [0.9, 0.1, 0.9, 0.9],
            [0.9, 0.9, 0.9, 0.9],
            [0.9, 0.9, 0.9, 0.9],
        ]
    )
    image_high_shadow = np.array(
        [
            [0.9, 0.9, 0.9, 0.9],
            [0.9, 0.1, 0.1, 0.9],
            [0.9, 0.1, 0.1, 0.9],
            [0.9, 0.9, 0.9, 0.9],
        ]
    )

    low_score = shadow_score(image_low_shadow, mask, darkness_threshold=0.2)
    high_score = shadow_score(image_high_shadow, mask, darkness_threshold=0.2)

    assert high_score > low_score
