"""Tests for datetime coordinate support."""

from datetime import date

import numpy as np

import nimblend

DT = "datetime64[D]"


class TestDatetimeCoords:
    """Test datetime64 coordinate handling."""

    def test_creation_datetime64(self):
        """Array can be created with datetime64 coordinates."""
        dates = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype=DT)
        arr = nimblend.Array(np.array([1.0, 2.0, 3.0]), {"time": dates})
        assert arr.shape == (3,)
        assert arr.coords["time"].dtype == np.dtype(DT)

    def test_sel_datetime64(self):
        """sel() works with datetime64 keys."""
        dates = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype=DT)
        arr = nimblend.Array(np.array([1.0, 2.0, 3.0]), {"time": dates})
        result = arr.sel({"time": np.datetime64("2024-01-02")})
        assert result == 2.0

    def test_sel_string(self):
        """sel() works with string keys for datetime coords."""
        dates = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype=DT)
        arr = nimblend.Array(np.array([1.0, 2.0, 3.0]), {"time": dates})
        result = arr.sel({"time": "2024-01-02"})
        assert result == 2.0

    def test_sel_python_date(self):
        """sel() works with python date keys."""
        dates = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype=DT)
        arr = nimblend.Array(np.array([1.0, 2.0, 3.0]), {"time": dates})
        result = arr.sel({"time": date(2024, 1, 2)})
        assert result == 2.0

    def test_sel_multiple_strings(self):
        """sel() works with multiple string keys."""
        dates = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype=DT)
        arr = nimblend.Array(np.array([1.0, 2.0, 3.0]), {"time": dates})
        result = arr.sel({"time": ["2024-01-01", "2024-01-03"]})
        np.testing.assert_array_equal(result.data, [1.0, 3.0])

    def test_outer_join_datetime(self):
        """Outer join works correctly with datetime coords."""
        dates1 = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype=DT)
        dates2 = np.array(["2024-01-02", "2024-01-03", "2024-01-04"], dtype=DT)

        arr1 = nimblend.Array(np.array([1.0, 2.0, 3.0]), {"time": dates1})
        arr2 = nimblend.Array(np.array([20.0, 30.0, 40.0]), {"time": dates2})

        result = arr1 + arr2
        assert result.shape == (4,)
        np.testing.assert_array_equal(result.data, [1.0, 22.0, 33.0, 40.0])
        assert len(result.coords["time"]) == 4

    def test_datetime_ns_precision(self):
        """Works with nanosecond precision datetime64."""
        dates = np.array(
            ["2024-01-01T12:00:00", "2024-01-01T12:00:01", "2024-01-01T12:00:02"],
            dtype="datetime64[ns]",
        )
        arr = nimblend.Array(np.array([1.0, 2.0, 3.0]), {"time": dates})
        result = arr.sel({"time": "2024-01-01T12:00:01"})
        assert result == 2.0
