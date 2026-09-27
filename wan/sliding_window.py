"""Per-inference temporal windows, measured in transformer latent frames."""

from dataclasses import dataclass


DEFAULT_WINDOW_LENGTH = 31
DEFAULT_WINDOW_STRIDE = 16


@dataclass(frozen=True)
class SlidingWindowConfig:
    window_length: int = DEFAULT_WINDOW_LENGTH
    window_stride: int = DEFAULT_WINDOW_STRIDE

    def __post_init__(self):
        for name in ("window_length", "window_stride"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer in internal latent frames.")
        if self.window_stride > self.window_length:
            raise ValueError("window_stride must not exceed window_length (this would leave frames uncovered).")

    def windows(self, temporal_length):
        """Keep a fixed stride, shorten the tail, and cover every frame."""
        if isinstance(temporal_length, bool) or not isinstance(temporal_length, int) or temporal_length <= 0:
            raise ValueError("The temporal length must be a positive integer.")
        for start in range(0, temporal_length, self.window_stride):
            end = min(start + self.window_length, temporal_length)
            yield start, end
            if end == temporal_length:
                break


def make_sliding_window_config(
    sliding_window=False,
    window_length=DEFAULT_WINDOW_LENGTH,
    window_stride=DEFAULT_WINDOW_STRIDE,
):
    if not isinstance(sliding_window, bool):
        raise ValueError("sliding_window must be a boolean.")
    return SlidingWindowConfig(window_length, window_stride) if sliding_window else None
