import numpy as np

from roof_area.metrics.quality import edge_confidence


def test_edge_confidence_means_edge_probs():
    mask = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    probs = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.3, 0.4, 0.0],
            [0.0, 0.5, 0.9, 0.6, 0.0],
            [0.0, 0.7, 0.8, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    expected_edge_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1]
    expected_mean = float(np.mean(expected_edge_values))

    assert edge_confidence(probs, mask) == expected_mean
