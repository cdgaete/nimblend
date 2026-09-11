"""Arguments validated before any entry is read."""

import re

import numpy as np
import pytest

from nimblend import DenseArray, Domain, StoredCoord, from_dense, from_long, kernel

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


def missing(what):
    return re.escape(
        f"{what} dimension(s) ['q'] are not in the array over ('x', 'y'); pass "
        "dimensions of the array"
    )


@pytest.mark.parametrize("array", arrays(), ids=["dense", "sparse"])
def test_a_dimension_the_array_lacks_raises_with_the_frame(array):
    with pytest.raises(ValueError, match=missing("sel")):
        array.sel({"q": 1})
    with pytest.raises(ValueError, match=missing("sum")):
        array.sum("q")
    with pytest.raises(ValueError, match=missing("domain")):
        array.domain(("q",))
    with pytest.raises(ValueError, match=missing("coordinates")):
        array.coordinates(("q",))
    with pytest.raises(ValueError, match=missing("rename")):
        array.rename({"q": "z"})
    with pytest.raises(ValueError, match=missing("shift")):
        array.roll({"q": 1})


def test_a_group_over_a_dimension_the_array_lacks_raises():
    with pytest.raises(ValueError, match=missing("group")):
        arrays()[1].group(("q",), into="g")


def repeated(dims):
    return re.escape(
        f"dimension(s) ['x'] appear more than once in {dims}; pass each dimension once"
    )


def test_a_repeated_dimension_raises_at_construction():
    coords = {"x": StoredCoord(LABELS["x"])}
    with pytest.raises(ValueError, match=repeated(("x", "x"))):
        DenseArray(np.zeros((2, 2)), coords, ("x", "x"))
    with pytest.raises(ValueError, match=repeated(("x", "x"))):
        from_long(("x", "x"), coords, {"x": np.array(["a"])}, np.array([1.0]))
    with pytest.raises(ValueError, match=repeated(("x", "x"))):
        Domain(np.array([0], dtype=np.int64), ("x", "x"), coords, (2, 2))


@pytest.mark.parametrize("array", arrays(), ids=["dense", "sparse"])
def test_a_repeated_dimension_raises_in_expand_and_domain(array):
    q = {"q": StoredCoord(np.array([1, 2]))}
    with pytest.raises(ValueError, match=repeated(("q", "q")).replace("'x'", "'q'")):
        array.expand(("q", "q"), q)
    with pytest.raises(ValueError, match=repeated(("x", "x"))):
        array.domain(("x", "x"))


@pytest.mark.parametrize("array", arrays(), ids=["dense", "sparse"])
def test_a_transpose_over_a_name_that_is_not_a_dimension_raises(array):
    message = re.escape(
        "transpose requires each dimension of ('x', 'y') once; got ('x', 1)"
    )
    with pytest.raises(ValueError, match=message):
        array.transpose("x", 1)
