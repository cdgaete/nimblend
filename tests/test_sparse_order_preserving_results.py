import numpy as np
import pytest

from nimblend import is_canonical, kernel
from nimblend.coords import StoredCoord
from nimblend.sparse import SparseArray

SHAPE = {"r": 5, "c": 4, "t": 6}


def held(seed=0, dims=("r", "c", "t")):
    """Return a random array over `dims` with about a third of its cells."""
    rng = np.random.default_rng(seed)
    shape = tuple(SHAPE[d] for d in dims)
    keep = np.flatnonzero(rng.random(int(np.prod(shape))) < 0.35)
    index = np.stack(np.unravel_index(keep, shape)).astype(np.int32)
    coords = {d: StoredCoord(np.arange(SHAPE[d])) for d in dims}
    return SparseArray(index, rng.standard_normal(keep.size), coords, dims)


# each operation returns its entries in the order of the entries it reads
RESULTS = {
    "times a number": lambda a: a * 2.0,
    "plus a number": lambda a: a + 1.0,
    "a number minus the array": lambda a: 1.0 - a,
    "divided by a number": lambda a: a / 4.0,
    "a number divided by the array": lambda a: 1.0 / a,
    "negated": lambda a: -a,
    "raised to a number": lambda a: a**2,
    "summed over the middle dimension": lambda a: a.sum("c"),
    "summed over the first dimension": lambda a: a.sum("r"),
    "minimum over the first dimension": lambda a: a.min("r"),
    "maximum over the last dimension": lambda a: a.max("t"),
    "mean over the middle dimension": lambda a: a.mean("c"),
    "shifted with entries dropped": lambda a: a.shift({"c": 1}),
    "selected at one label": lambda a: a.sel({"c": 2}),
    "as empty": lambda a: a.as_empty(),
    "as unknown": lambda a: a.as_unknown(),
}


def _refuse_canonicalize(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the result was canonicalized a second time")

    monkeypatch.setattr(kernel, "canonicalize", refuse)


@pytest.mark.parametrize("name", RESULTS)
def test_an_order_preserving_result_is_not_canonicalized_again(name, monkeypatch):
    array = held()
    _refuse_canonicalize(monkeypatch)
    RESULTS[name](array)


def test_a_product_over_one_shared_dimension_is_not_canonicalized_again(
    monkeypatch,
):
    left, right = held(7, ("r", "c")), held(8, ("c", "t"))
    _refuse_canonicalize(monkeypatch)
    assert (left * right).dims == ("r", "c", "t")


@pytest.mark.parametrize("name", RESULTS)
@pytest.mark.parametrize("seed", range(4))
def test_an_order_preserving_result_is_canonical(name, seed):
    result = RESULTS[name](held(seed))
    assert is_canonical(result.index, result.shape)


@pytest.mark.parametrize("seed", range(4))
def test_a_product_over_one_shared_dimension_is_canonical(seed):
    product = held(seed, ("r", "c")) * held(seed + 10, ("c", "t"))
    assert is_canonical(product.index, product.shape)


@pytest.mark.parametrize("seed", range(4))
def test_a_product_whose_shared_dimension_is_last_in_both_is_canonical(seed):
    product = held(seed, ("r", "c")) * held(seed + 10, ("t", "c"))
    assert product.dims == ("r", "c", "t")
    assert is_canonical(product.index, product.shape)


def test_a_product_with_a_number_shares_the_index_of_the_array():
    array = held()
    assert np.shares_memory((array * 2.0).index, array.index)


def test_a_shift_that_wraps_is_canonical():
    # a wrapped entry moves before the entries it followed
    array = held()
    wrapped = array.shift({"c": 1}, mode="wrap")
    assert is_canonical(wrapped.index, wrapped.shape)
    assert wrapped.nnz == array.nnz


def test_a_product_with_an_extended_precision_number_has_float64_values():
    # numpy returns longdouble for a float64 array times a longdouble scalar
    product = held() * np.longdouble(2.0)
    assert product.data.dtype == np.float64
