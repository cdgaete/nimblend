import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import splu

import nimblend as nb
from nimblend import kernel

LABELS = np.array(list("abcde"))


def rows_contiguous(index):
    return all(index[d].flags.c_contiguous for d in range(index.shape[0]))


def coords():
    return {d: nb.StoredCoord(LABELS) for d in ("r", "k", "j")}


def unsorted(c):
    labels = {"r": np.array(list("eacbd")), "k": np.array(list("adbec"))}
    return nb.from_long(("r", "k"), c, labels, np.arange(1.0, 6.0))


def operations():
    c = coords()
    a = unsorted(c)
    diagonal = nb.from_long(("r", "k"), c, {"r": LABELS, "k": LABELS}, np.ones(5))
    narrow = nb.from_long(
        ("k",), c, {"k": np.array(list("ace"))}, np.array([2.0, 3.0, 4.0])
    )
    full = nb.from_long(("k",), c, {"k": LABELS}, np.arange(1.0, 6.0))
    overlap = nb.from_long(
        ("k", "j"),
        c,
        {"k": np.array(list("abc")), "j": np.array(list("cab"))},
        np.array([1.0, 2.0, 3.0]),
    )
    repeated = nb.from_long(
        ("r", "k"),
        c,
        {"r": np.array(list("eaae")), "k": np.array(list("baab"))},
        np.array([1.0, 2.0, 3.0, 4.0]),
        on_duplicate="sum",
    )
    domain = nb.Domain.from_labels(("r",), c, {"r": np.array(list("abc"))})
    reversed_labels = {"r": LABELS[::-1], "k": LABELS}
    return {
        "from_long on unsorted rows": a,
        "from_long on sorted rows": diagonal,
        "from_long with on_duplicate='sum'": repeated,
        "sum_arrays": nb.sum_arrays([a, diagonal]),
        "addition": a + diagonal,
        "product with a scalar": 2 * a,
        "product with a narrower array": a * narrow,
        "product with an overlapping array": a * overlap,
        "quotient by a narrower array": a / full,
        "sum over a dimension": a.sum("k"),
        "restrict": a.restrict(domain),
        "conform": a.conform(("r", "k"), reversed_labels),
        "group": a.group(("r", "k"), "rk"),
        "shift": a.shift({"k": 1}),
        "transpose": a.transpose("k", "r"),
    }


@pytest.mark.parametrize("name", list(operations()))
def test_every_operation_returns_an_index_with_contiguous_rows(name):
    assert rows_contiguous(operations()[name].index)


def test_an_array_grouped_into_a_buffer_has_contiguous_rows_without_a_copy():
    # a reservation is a column slice of the buffer: each row is contiguous,
    # the two-dimensional slice is not, and the entries are written in place
    buffer = nb.EntryBuffer(2, 10)
    grouped = unsorted(coords()).group(("r", "k"), "rk", reserve=buffer.reserve)
    assert rows_contiguous(grouped.index)
    assert np.shares_memory(grouped.index, buffer.index)


def unsorted_block():
    idx = np.array([[2, 0, 2, 1], [1, 0, 1, 2]], dtype=np.int32)
    return idx, np.array([1.0, 2.0, 3.0, 4.0])


def test_canonicalize_returns_c_contiguous_rows_after_a_sort():
    idx, data = unsorted_block()
    out_idx, _ = kernel.canonicalize(idx, data, (3, 3), on_duplicate="sum")
    assert out_idx.flags.c_contiguous


def test_canonicalize_returns_c_contiguous_rows_after_a_sum_of_sorted_keys():
    idx = np.array([[0, 0, 1], [0, 0, 2]], dtype=np.int32)
    out_idx, _ = kernel.canonicalize(idx, np.ones(3), (3, 3), on_duplicate="sum")
    assert out_idx.flags.c_contiguous


def test_merge_sum_returns_c_contiguous_rows():
    idx, data = unsorted_block()
    first = kernel.canonicalize(idx, data, (3, 3))
    out_idx, _ = kernel.merge_sum(
        [first[0], first[0]], [first[1], first[1]], (3, 3), "union"
    )
    assert out_idx.flags.c_contiguous


def test_reduce_axis_returns_c_contiguous_rows_after_a_sort():
    idx = np.array([[0, 1, 0], [2, 0, 0], [1, 1, 0]], dtype=np.int32)
    out_idx, _ = kernel.reduce_axis(idx, np.ones(3), 0, (2, 3, 2))
    assert out_idx.flags.c_contiguous


def test_compress_returns_c_contiguous_rows():
    idx, data = unsorted_block()
    out_idx, _ = kernel.compress(idx, data, np.array([True, False, True, True]))
    assert out_idx.flags.c_contiguous


def test_multiply_lookup_returns_c_contiguous_rows():
    idx, data = unsorted_block()
    take = np.array([0, -1, 1, 0])
    out_idx, _ = kernel.multiply_lookup(idx, data, np.array([2.0, 3.0]), take)
    assert out_idx.flags.c_contiguous


def test_to_csr_returns_contiguous_views_for_an_array_built_from_unsorted_rows():
    arr = unsorted(coords())
    indices, values, indptr = arr.to_csr()
    assert indices.flags.c_contiguous
    assert values.flags.c_contiguous
    assert indptr.flags.c_contiguous
    assert np.shares_memory(indices, arr.index)
    assert np.shares_memory(values, arr.data)


def test_scipy_factorizes_the_transpose_of_an_array_with_summed_rows():
    # the transpose of a CSR matrix is a CSC matrix over the same three
    # arrays, passed to SuperLU without a copy
    rng = np.random.default_rng(7)
    n = 40
    labels = np.arange(n)
    c = {"r": nb.StoredCoord(labels), "k": nb.StoredCoord(labels)}
    off_rows = rng.integers(0, n, 200)
    off_cols = rng.integers(0, n, 200)
    rows = np.concatenate([labels, off_rows, labels])
    cols = np.concatenate([labels, off_cols, labels])
    values = np.concatenate(
        [np.full(n, 10.0), rng.uniform(-1.0, 0.0, 200), np.full(n, -1.0)]
    )
    order = rng.permutation(rows.size)
    arr = nb.from_long(
        ("r", "k"),
        c,
        {"r": rows[order], "k": cols[order]},
        values[order],
        on_duplicate="sum",
    )
    indices, data, indptr = arr.to_csr()
    matrix = sp.csr_matrix((data, indices, indptr), shape=arr.shape)
    b = rng.uniform(size=n)
    solved = splu(matrix.T).solve(b)
    np.testing.assert_allclose(solved, np.linalg.solve(arr.to_dense().T, b), rtol=1e-12)
