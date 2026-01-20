from roof_area.metrics.quality import quality_flag_shadow


def test_quality_flag_shadow_threshold():
    assert quality_flag_shadow(0.6, threshold=0.5) is True
    assert quality_flag_shadow(0.4, threshold=0.5) is False
