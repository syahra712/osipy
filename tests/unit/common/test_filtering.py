"""Unit tests for osipy.common.signal.filtering module."""

import numpy as np
import pytest

from osipy.common.signal.filtering import temporal_filter

ndimage = pytest.importorskip("scipy.ndimage")


@pytest.fixture
def signal() -> np.ndarray:
    """Random 4D data with a short time axis."""
    rng = np.random.default_rng(0)
    return rng.random((3, 2, 2, 20))


class TestTemporalFilter:
    """Tests for temporal_filter against scipy.ndimage."""

    @pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
    def test_gaussian_matches_scipy(self, signal: np.ndarray, sigma: float) -> None:
        """Gaussian filter matches scipy and is not shifted in time."""
        result = temporal_filter(signal, filter_type="gaussian", sigma=sigma)
        expected = ndimage.gaussian_filter1d(signal, sigma, axis=-1, mode="nearest")
        np.testing.assert_allclose(result, expected, atol=1e-10)

    @pytest.mark.parametrize("size", [2, 3, 4, 5])
    def test_uniform_matches_scipy(self, signal: np.ndarray, size: int) -> None:
        """Moving average matches scipy and is not shifted in time."""
        result = temporal_filter(signal, filter_type="moving_average", window_size=size)
        expected = ndimage.uniform_filter1d(signal, size, axis=-1, mode="nearest")
        np.testing.assert_allclose(result, expected, atol=1e-10)

    @pytest.mark.parametrize("size", [3, 5])
    def test_median_matches_scipy(self, signal: np.ndarray, size: int) -> None:
        """Median filter matches scipy."""
        result = temporal_filter(signal, filter_type="median", window_size=size)
        expected = ndimage.median_filter(signal, size=(1, 1, 1, size), mode="nearest")
        np.testing.assert_allclose(result, expected, atol=1e-10)

    @pytest.mark.parametrize(
        ("filter_type", "kwargs"),
        [("gaussian", {"sigma": 1.0}), ("moving_average", {"window_size": 3})],
    )
    def test_impulse_stays_centred(self, filter_type: str, kwargs: dict) -> None:
        """A centred impulse stays centred after smoothing."""
        data = np.zeros((1, 1, 1, 15))
        data[..., 7] = 1.0
        result = temporal_filter(data, filter_type=filter_type, **kwargs)
        assert result.shape == data.shape
        # Symmetric about the impulse, so not shifted either way
        np.testing.assert_allclose(result, result[..., ::-1], atol=1e-12)
