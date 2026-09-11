"""Arguments validated before any entry is read."""

import re

import numpy as np
import pytest

from nimblend import DenseArray, from_dense, kernel

LABELS = {"x": np.array(["a", "b"]), "y": np.array([10, 20, 30])}
VALUES = np.arange(6, dtype=np.float64).reshape(2, 3)


def arrays():
    return (
        DenseArray.from_dense(VALUES, LABELS, "empty"),
        from_dense(VALUES, LABELS),
    )


def test_an_unknown_duplicate_policy_raises_with_no_repeated_entry():
    idx = np.array([[0, 1], [0, 1]], dtype=np.int32)
    data = np.array([1.0, 2.0])
    message = re.escape("on_duplicate is 'sum' or 'raise'; got 'first'")
    with pytest.raises(ValueError, match=message):
        kernel.canonicalize(idx, data, (2, 2), on_duplicate="first")
    # an empty block is validated too
    empty = np.empty((2, 0), dtype=np.int32)
    with pytest.raises(ValueError, match=message):
        kernel.canonicalize(empty, np.empty(0), (2, 2), on_duplicate="first")


@pytest.mark.parametrize("array", arrays(), ids=["dense", "sparse"])
def test_an_unknown_shift_mode_raises_with_no_shift_given(array):
    with pytest.raises(
        ValueError, match=re.escape("mode is 'drop' or 'wrap'; got 'clip'")
    ):
        array.shift({}, mode="clip")


@pytest.mark.parametrize("array", arrays(), ids=["dense", "sparse"])
def test_a_shift_along_a_dimension_the_array_lacks_raises(array):
    message = re.escape(
        "shift dimension(s) ['q'] are not in the array over ('x', 'y'); pass "
        "dimensions of the array"
    )
    with pytest.raises(ValueError, match=message):
        array.shift({"q": 1})


@pytest.mark.parametrize("array", arrays(), ids=["dense", "sparse"])
def test_a_transpose_that_repeats_a_dimension_raises(array):
    message = re.escape(
        "transpose requires each dimension of ('x', 'y') once; got ('x', 'y', 'x')"
    )
    with pytest.raises(ValueError, match=message):
        array.transpose("x", "y", "x")
