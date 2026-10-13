import numpy as np
import pytest

from runtime.interpolation import AdjustableLinearInterpolator


def test_requested_multiplier_waits_for_the_segment_in_flight() -> None:
    interpolator = AdjustableLinearInterpolator(multiplier=2)
    interpolator.add(np.array([0.0]))
    interpolator.get(0)
    assert interpolator.needs_new_action()
    interpolator.add(np.array([4.0]))

    interpolator.request_multiplier(4)
    assert not interpolator.needs_new_action()
    assert interpolator.multiplier == 2
    np.testing.assert_allclose(interpolator.get(0), [2.0])
    np.testing.assert_allclose(interpolator.get(1), [4.0])

    assert interpolator.needs_new_action()
    assert interpolator.multiplier == 4


@pytest.mark.parametrize("multiplier", [0, 6])
def test_request_multiplier_rejects_out_of_range_values(multiplier: int) -> None:
    interpolator = AdjustableLinearInterpolator()

    with pytest.raises(ValueError, match="multiplier"):
        interpolator.request_multiplier(multiplier)
    assert interpolator.requested_multiplier == 1
