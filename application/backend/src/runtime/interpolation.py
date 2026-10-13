from __future__ import annotations

from typing import override

from physicalai.runtime import LinearInterpolator

MIN_INTERPOLATION_MULTIPLIER = 1
MAX_INTERPOLATION_MULTIPLIER = 5


class AdjustableLinearInterpolator(LinearInterpolator):
    """``LinearInterpolator`` whose multiplier can change while ``RobotRuntime`` runs.

    A requested multiplier takes effect once the segment in flight completes, so
    a cycle is never re-split half-way. ``RobotRuntime`` derives each tick's
    period from ``multiplier`` right after asking ``needs_new_action()``.
    """

    def __init__(self, multiplier: int = MIN_INTERPOLATION_MULTIPLIER) -> None:
        super().__init__(multiplier=multiplier)
        self._requested = multiplier

    @property
    def requested_multiplier(self) -> int:
        return self._requested

    def request_multiplier(self, multiplier: int) -> None:
        if not MIN_INTERPOLATION_MULTIPLIER <= multiplier <= MAX_INTERPOLATION_MULTIPLIER:
            raise ValueError(
                f"multiplier must be between {MIN_INTERPOLATION_MULTIPLIER} and "
                f"{MAX_INTERPOLATION_MULTIPLIER}, got {multiplier}"
            )
        self._requested = multiplier

    @override
    def needs_new_action(self) -> bool:
        done = super().needs_new_action()
        if done:
            self.multiplier = self._requested
        return done
