import numpy as np

from nimblend import kernel
from nimblend.domain import Domain
from nimblend.sparse import SparseArray


def block(values, labels):
    """An array holding only the nonzero cells of `values`."""
    arr = SparseArray.from_dense(np.asarray(values, dtype=np.float64), labels)
    keep = arr.data != 0.0
    return SparseArray.from_canonical(
        arr.index[:, keep], arr.data[keep], arr.coords, arr.dims
    )


def xy():
    return block(
        [[1.0, 0.0, 2.0], [0.0, 3.0, 4.0]],
        {"x": np.array(["a", "b"]), "y": np.array([10, 20, 30])},
    )


def test_restrict_keeps_only_the_entries_the_domain_carries():
    arr = xy()
    # a domain over x carrying row "a" alone; its cells hold 1.0 and 2.0
    only_a = Domain(np.array([0]), ("x",), {"x": arr.coords["x"]}, (2,))
    got = arr.restrict(only_a)
    assert got.nnz == 2
    assert got.data.tolist() == [1.0, 2.0]


def test_restrict_drops_an_entry_outside_the_domain():
    arr = xy()
    first = SparseArray.from_canonical(
        arr.index[:, :1], arr.data[:1], arr.coords, arr.dims
    )
    got = arr.restrict(first.domain())
    assert got.nnz == 1
    assert list(got.data) == [1.0]


def test_restrict_returns_the_same_array_when_nothing_drops():
    arr = xy()
    assert arr.restrict(arr.domain()) is arr


def test_restrict_leaves_the_result_canonical():
    arr = xy()
    two = SparseArray.from_canonical(
        arr.index[:, [0, 3]], arr.data[[0, 3]], arr.coords, arr.dims
    )
    got = arr.restrict(two.domain())
    assert kernel.is_canonical(got.index, got.shape)


def test_restrict_keeps_the_arrays_absence():
    arr = block(
        [[1.0, 2.0]], {"x": np.array(["a"]), "y": np.array([10, 20])}
    ).as_unknown()
    first = SparseArray.from_canonical(
        arr.index[:, :1], arr.data[:1], arr.coords, arr.dims, "unknown"
    )
    assert arr.restrict(first.domain()).absence == "unknown"


def test_restricting_to_an_empty_domain_leaves_no_entry():
    arr = xy()
    empty = SparseArray.from_canonical(
        arr.index[:, :0], arr.data[:0], arr.coords, arr.dims
    )
    assert arr.restrict(empty.domain()).nnz == 0
