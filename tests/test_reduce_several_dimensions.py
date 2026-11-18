"""A reduction over several dimensions, on both implementations."""

import numpy as np
import pytest

from nimblend import DenseArray, SparseArray, StoredCoord

LABELS = {
    "x": np.array(["a", "b", "c"]),
    "y": np.array([10, 20, 30, 40]),
    "z": np.array([1.5, 2.5]),
}
SHAPE = (3, 4, 2)
OPS = ("sum", "mean", "min", "max")
IDENTITY = {"sum": 0.0, "mean": 0.0, "min": np.inf, "max": -np.inf}


def pair(seed, density, absence):
    """Return one logical array as a SparseArray and a DenseArray, with its cells."""
    rng = np.random.default_rng(seed)
    values = np.round(rng.random(SHAPE) * 10, 3)
    mask = rng.random(SHAPE) < density
    mask[0, 0, 0] = density > 0
    index = np.stack(np.nonzero(mask)).astype(np.int32)
    coords = {name: StoredCoord(labels) for name, labels in LABELS.items()}
    sparse = SparseArray.from_canonical(
        index, values[mask], coords, ("x", "y", "z"), absence
    )
    if absence == "unknown":
        tagged = np.where(mask, values, np.nan)
        dense = DenseArray.from_dense(tagged, LABELS, "unknown")
    else:
        dense = DenseArray.from_dense(values, LABELS, "empty", mask)
    return {"sparse": sparse, "dense": dense}, values, mask


def over_present(values, mask, axes, op):
    """Return the reduction of the present cells over `axes`, NaN where none is."""
    present = mask.any(axis=axes)
    if op == "mean":
        total = np.where(mask, values, 0.0).sum(axis=axes)
        reduced = total / np.maximum(mask.sum(axis=axes), 1)
    else:
        reduced = getattr(np, op)(np.where(mask, values, IDENTITY[op]), axis=axes)
    return np.where(present, reduced, np.nan)


CASES = [
    (kind, seed, density, absence)
    for kind in ("sparse", "dense")
    for seed in (0, 1)
    for density in (1.0, 0.5)
    for absence in ("empty", "unknown")
]


def policy(absence):
    return {"skip": True} if absence == "unknown" else {}


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("kind,seed,density,absence", CASES)
def test_a_reduction_over_two_dimensions_reads_the_present_cells(
    op, kind, seed, density, absence
):
    arrays, values, mask = pair(seed, density, absence)
    got = getattr(arrays[kind], op)(("x", "z"), **policy(absence))
    assert got.dims == ("y",)
    want = over_present(values, mask, (0, 2), op)
    assert np.allclose(got.to_dense(fill=np.nan), want, equal_nan=True)


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("kind,seed,density,absence", CASES)
def test_the_dimensions_are_reduced_in_any_order_given(
    op, kind, seed, density, absence
):
    arrays, _, _ = pair(seed, density, absence)
    ahead = getattr(arrays[kind], op)(("x", "z"), **policy(absence))
    back = getattr(arrays[kind], op)(("z", "x"), **policy(absence))
    assert np.array_equal(
        ahead.to_dense(fill=np.nan), back.to_dense(fill=np.nan), equal_nan=True
    )


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("kind", ["sparse", "dense"])
def test_one_dimension_in_a_tuple_equals_the_dimension_alone(op, kind):
    arrays, _, _ = pair(0, 0.5, "empty")
    alone = getattr(arrays[kind], op)("y")
    listed = getattr(arrays[kind], op)(("y",))
    assert listed.dims == alone.dims
    assert np.array_equal(
        listed.to_dense(fill=np.nan), alone.to_dense(fill=np.nan), equal_nan=True
    )


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("kind", ["sparse", "dense"])
def test_a_fill_counts_at_each_absent_cell_of_every_reduced_dimension(op, kind):
    arrays, values, mask = pair(1, 0.5, "empty")
    got = getattr(arrays[kind], op)(("x", "z"), fill=1.0)
    want = getattr(np, op)(np.where(mask, values, 1.0), axis=(0, 2))
    assert np.allclose(got.to_dense(), want)
    assert got.nnz == SHAPE[1]


@pytest.mark.parametrize("kind", ["sparse", "dense"])
def test_reducing_every_dimension_in_a_tuple_leaves_an_array_over_none(kind):
    arrays, values, mask = pair(0, 0.5, "empty")
    got = arrays[kind].sum(("x", "y", "z"))
    assert got.dims == ()
    assert got.shape == ()
    assert np.isclose(got.to_dense(), values[mask].sum())


@pytest.mark.parametrize("kind", ["sparse", "dense"])
def test_a_dimension_the_array_does_not_have_is_refused(kind):
    arrays, _, _ = pair(0, 0.5, "empty")
    with pytest.raises(ValueError, match="'w'"):
        arrays[kind].sum(("x", "w"))


@pytest.mark.parametrize("kind", ["sparse", "dense"])
def test_a_dimension_given_twice_is_refused(kind):
    arrays, _, _ = pair(0, 0.5, "empty")
    with pytest.raises(ValueError, match="more than once"):
        arrays[kind].sum(("x", "x"))


@pytest.mark.parametrize("kind", ["sparse", "dense"])
def test_no_dimension_is_refused(kind):
    arrays, _, _ = pair(0, 0.5, "empty")
    with pytest.raises(ValueError, match="no dimension"):
        arrays[kind].sum(())


def test_a_sparse_sum_over_two_dimensions_sorts_once(monkeypatch):
    arrays, _, _ = pair(0, 0.5, "empty")
    calls = []
    sort = np.argsort

    def counted(*args, **kwargs):
        calls.append(1)
        return sort(*args, **kwargs)

    monkeypatch.setattr(np, "argsort", counted)
    arrays["sparse"].sum(("x", "y"))
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ["sparse", "dense"])
@pytest.mark.parametrize("given", [5, b"x", None.__class__])
def test_a_dimension_that_is_not_a_name_of_the_array_is_refused(kind, given):
    # a value that is not a sequence of names is read as one dimension
    arrays, _, _ = pair(0, 0.5, "empty")
    with pytest.raises(ValueError, match="are not in the array"):
        arrays[kind].sum(given)
