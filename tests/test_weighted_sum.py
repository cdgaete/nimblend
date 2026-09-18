import numpy as np
import pytest

from nimblend import DenseArray, SparseArray, StoredCoord

LABELS = {
    "x": np.array(["a", "b", "c"]),
    "y": np.array([10, 20, 30, 40]),
    "z": np.array([1, 2]),
}


def random_sparse(seed, density, absence="empty"):
    rng = np.random.default_rng(seed)
    shape = tuple(labels.size for labels in LABELS.values())
    values = np.round(rng.standard_normal(shape) * 10, 3)
    mask = rng.random(shape) < density
    coords = {name: StoredCoord(labels) for name, labels in LABELS.items()}
    index = np.stack(np.nonzero(mask)).astype(np.int32)
    return SparseArray(index, values[mask], coords, tuple(LABELS), absence)


def weight_array(dim, weights):
    return SparseArray.from_dense(weights, {dim: LABELS[dim]})


def as_dense(arr):
    present = np.zeros(arr.shape, dtype=bool)
    present[tuple(arr.index)] = True
    return DenseArray(arr.to_dense(fill=0.0), arr.coords, arr.dims, "empty", present)


CASES = [
    (seed, density, dim)
    for seed in (0, 1)
    for density in (1.0, 0.4, 0.0)
    for dim in ("x", "y", "z")
]


@pytest.mark.parametrize("seed,density,dim", CASES)
def test_weighted_sum_equals_the_sum_of_the_weighted_product(seed, density, dim):
    arr = random_sparse(seed, density)
    weights = np.random.default_rng(seed + 10).standard_normal(len(LABELS[dim]))
    got = arr.weighted_sum(dim, weights)
    wanted = (arr * weight_array(dim, weights)).sum(dim)
    assert got.dims == wanted.dims
    assert got.coords == wanted.coords
    assert np.array_equal(got.coordinates(), wanted.coordinates())
    assert np.allclose(got.values(), wanted.values())


@pytest.mark.parametrize("seed,density,dim", CASES)
def test_the_dense_weighted_sum_matches_the_sparse_one(seed, density, dim):
    arr = random_sparse(seed, density)
    weights = np.random.default_rng(seed + 10).standard_normal(len(LABELS[dim]))
    sparse = arr.weighted_sum(dim, weights)
    dense = as_dense(arr).weighted_sum(dim, weights)
    assert dense.dims == sparse.dims
    assert np.array_equal(dense.coordinates(), sparse.coordinates())
    assert np.allclose(dense.values(), sparse.values())


def test_weighted_sum_of_an_empty_array_is_empty():
    got = random_sparse(0, 0.0).weighted_sum("y", np.ones(4))
    assert got.dims == ("x", "z")
    assert got.nnz == 0


def test_weighted_sum_keeps_the_absence():
    arr = random_sparse(0, 0.5, "unknown")
    assert arr.weighted_sum("x", np.ones(3), skip=True).absence == "unknown"


@pytest.mark.parametrize("make", [lambda a: a, as_dense], ids=["sparse", "dense"])
def test_weighted_sum_raises_for_weights_of_another_shape(make):
    arr = make(random_sparse(0, 1.0))
    with pytest.raises(ValueError, match="dimension 'y' has extent 4"):
        arr.weighted_sum("y", np.ones(3))


@pytest.mark.parametrize("make", [lambda a: a, as_dense], ids=["sparse", "dense"])
def test_weighted_sum_raises_for_a_dimension_the_array_does_not_have(make):
    arr = make(random_sparse(0, 1.0))
    with pytest.raises(ValueError, match="are not in the array"):
        arr.weighted_sum("w", np.ones(3))


@pytest.mark.parametrize(
    "make", [lambda a: a, lambda a: as_dense(a).as_unknown()], ids=["sparse", "dense"]
)
def test_weighted_sum_under_unknown_requires_skip(make):
    arr = make(random_sparse(0, 0.5, "unknown"))
    with pytest.raises(ValueError, match="pass skip=True"):
        arr.weighted_sum("x", np.ones(3))


def test_weighted_sum_raises_for_a_skip_other_than_true():
    with pytest.raises(ValueError, match="skip is True or None"):
        random_sparse(0, 1.0).weighted_sum("x", np.ones(3), skip=False)
