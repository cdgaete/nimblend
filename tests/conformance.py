"""The Array contract, run against an implementation."""

import numpy as np
import pytest

from nimblend import protocol
from nimblend.coords import StoredCoord
from nimblend.domain import Domain


def missing_members(cls):
    """Protocol members `cls` does not define."""
    return sorted(m for m in protocol.Array.__protocol_attrs__ if not hasattr(cls, m))


def check_array_contract(make):
    """Run the shared contract against a factory building a 2-D array.

    `make(values, labels, absence)` returns an implementation holding
    `values` (a dense ndarray) under `labels` (a dict of dimension to label
    array).
    """
    labels = {"x": np.array(["a", "b"]), "y": np.array([10, 20, 30])}
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    arr = make(values, labels, "empty")

    assert isinstance(arr, protocol.Array)
    assert arr.dims == ("x", "y")
    assert arr.shape == (2, 3)
    assert arr.absence in protocol.ABSENCE
    assert np.array_equal(arr.to_dense(), values)

    assert np.array_equal(arr.sel({"x": "b"}).to_dense(), values[1])
    assert np.array_equal(arr.sum("x").to_dense(), values.sum(axis=0))
    assert arr.sum() == values.sum()
    assert np.array_equal(arr.transpose("y", "x").to_dense(), values.T)
    assert arr.rename({"x": "z"}).dims == ("z", "y")

    assert arr.domain(("x",)).dims == ("x",)
    assert arr.domain(("x",)).size == 2
    assert arr.coordinates(("x",)).tolist() == [[0, 0, 0, 1, 1, 1]]
    assert arr.restrict(arr.domain()) is not None
    assert arr.restrict(arr.domain()).nnz == arr.domain().size
    widened = arr.expand(("z",), {"z": StoredCoord(np.array([1, 2]))})
    assert widened.dims == ("x", "y", "z")
    assert widened.to_dense().shape == (2, 3, 2)

    assert arr.values().size == arr.domain().size
    assert sorted(arr.values().tolist()) == sorted(values.ravel().tolist())

    # reducing every dimension in turn ends at an array over none, which is
    # one entry carrying the total; `sum()` with no dimension is the number
    emptied = arr.sum("x").sum("y")
    assert emptied.dims == ()
    assert emptied.shape == ()
    assert emptied.nnz == 1
    assert np.array_equal(emptied.to_dense(), values.sum())
    assert emptied.values().tolist() == [values.sum()]

    grouped = arr.group(("x",), into="g")
    assert grouped.dims == ("g", "y")
    assert np.array_equal(grouped.to_dense(), values)

    # a domain states which grouped coordinates survive; an entry at one it
    # does not carry is not emitted
    kept = Domain.from_labels(
        ("x",), {"x": StoredCoord(labels["x"])}, {"x": np.array(["b"])}
    )
    narrowed = arr.group(("x",), into="g", domain=kept)
    assert narrowed.dims == ("g", "y")
    assert np.array_equal(narrowed.to_dense(), values[1:])

    # an offset numbers the result into an extent wider than its members span
    shifted = arr.group(("x",), into="g", offset=5)
    assert shifted.coordinates()[0].tolist() == [5, 5, 5, 6, 6, 6]

    # a shift of nothing moves nothing out of the frame, so it drops nothing
    assert np.array_equal(arr.shift({"x": 0}).to_dense(), values)
    assert arr.shift({"x": 0}).nnz == arr.nnz
    assert np.array_equal(arr.roll({"x": 0}).to_dense(), values)

    assert np.array_equal(arr.shift({"x": 1}).to_dense(), [[0.0, 0.0, 0.0], values[0]])
    assert np.array_equal(arr.roll({"x": 1}).to_dense(), values[[1, 0]])

    with pytest.raises(ValueError, match="mode is 'drop' or 'wrap'"):
        arr.shift({"x": 1}, mode="sideways")

    # a repeated label would ask one position to occupy two
    with pytest.raises(ValueError, match="is named twice for dimension"):
        arr.conform(["x", "y"], {"x": np.array(["a", "a"]), "y": labels["y"]})

    with pytest.raises(ValueError, match="leading prefix"):
        arr.group(("y",), into="g")
    with pytest.raises(ValueError, match="already carried"):
        arr.group(("x",), into="y")
    with pytest.raises(ValueError, match="not negative"):
        arr.group(("x",), into="g", offset=-1)

    check_frame_contract(arr)
    check_arithmetic_contract(arr, make(values, labels, "empty"))


def check_frame_contract(arr):
    """What an array answers about its own frame, and what it refuses there."""
    assert set(arr.coords) == set(arr.dims)
    assert tuple(len(arr.coords[d]) for d in arr.dims) == arr.shape
    assert arr.nnz == arr.domain().size

    # no dimensions given reverses them, so a two-dimensional array flips
    assert arr.transpose().dims == tuple(reversed(arr.dims))
    assert np.array_equal(arr.transpose().to_dense(fill=0.0), arr.to_dense(fill=0.0).T)

    assert arr.rename({"x": "z"}).dims == ("z", "y")
    # two dimensions renamed onto one name would ask a single coordinate to
    # stand for both, so it is refused rather than carried
    with pytest.raises(ValueError, match="onto one name"):
        arr.rename({"x": "y"})

    assert arr.as_empty().absence == "empty"
    assert arr.as_unknown().absence == "unknown"
    assert arr.as_empty().as_unknown().absence == "unknown"
    assert arr.as_unknown().as_empty().nnz == arr.nnz


def check_arithmetic_contract(arr, other):
    """The operators, over one frame. A wider frame is each implementation's own."""
    dense = arr.to_dense(fill=0.0)
    for got, wanted in (
        (arr + other, dense * 2),
        (other + arr, dense * 2),
        (arr - other, dense * 0),
        (-arr, -dense),
        (arr * other, dense * dense),
        (arr * 3, dense * 3),
        (3 * arr, dense * 3),
        (arr + 3, dense + 3),
        (3 + arr, dense + 3),
        (arr - 3, dense - 3),
        (arr / 2, dense / 2),
    ):
        assert np.allclose(got.to_dense(fill=0.0), wanted), got

    # an absent denominator is a coordinate the numerator reaches and the
    # denominator does not: a coverage refusal, not an arithmetic one
    one_row = Domain.from_labels(("x",), {"x": arr.coords["x"]}, {"x": np.array(["b"])})
    holed = other.restrict(one_row)
    assert holed.nnz < arr.nnz
    with pytest.raises(ValueError, match="denominator is absent"):
        arr / holed
