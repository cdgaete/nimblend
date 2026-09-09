import numpy as np

from nimblend.sparse import SparseArray


def block(values, labels):
    """An array holding only the nonzero cells of `values`."""
    arr = SparseArray.from_dense(np.asarray(values, dtype=np.float64), labels)
    keep = arr.data != 0.0
    return SparseArray.from_canonical(
        arr.index[:, keep], arr.data[keep], arr.coords, arr.dims
    )


def ijc():
    values = np.zeros((2, 3, 2))
    values[0, 0, 1] = 5.0
    values[0, 2, 0] = 7.0
    values[1, 1, 1] = 9.0
    return block(
        values,
        {
            "i": np.array(["p", "q"]),
            "j": np.array([10, 20, 30]),
            "c": np.array([0, 1]),
        },
    )


def test_domain_over_a_leading_prefix_answers_the_distinct_coordinates():
    # entries sit at (i, j) of (0,0), (0,2) and (1,1)
    got = ijc().domain(("i", "j"))
    assert got.dims == ("i", "j")
    assert got.shape == (2, 3)
    assert got.coordinates().tolist() == [[0, 0, 1], [0, 2, 1]]


def test_a_shared_coordinate_is_one_member_not_two():
    values = np.zeros((2, 3, 2))
    values[0, 0, 0] = 1.0
    values[0, 0, 1] = 2.0
    arr = block(
        values,
        {"i": np.array(["p", "q"]), "j": np.array([10, 20, 30]), "c": np.array([0, 1])},
    )
    assert arr.nnz == 2
    assert arr.domain(("i", "j")).size == 1


def test_domain_over_every_dimension_answers_one_member_per_entry():
    arr = ijc()
    assert arr.domain().size == arr.nnz
    assert arr.domain().dims == ("i", "j", "c")


def test_domain_over_a_trailing_dimension_answers_sorted_members():
    got = ijc().domain(("c",))
    assert got.dims == ("c",)
    assert list(got.codes) == [0, 1]


def test_a_domain_carries_the_arrays_own_coordinates():
    got = ijc().domain(("i", "j"))
    assert list(got.labels()["i"]) == ["p", "p", "q"]
    assert list(got.labels()["j"]) == [10, 30, 20]


def test_coordinates_answer_each_entrys_multi_index():
    assert ijc().coordinates(("i", "j")).tolist() == [[0, 0, 1], [0, 2, 1]]


def test_coordinates_are_a_copy_that_does_not_write_through():
    arr = ijc()
    got = arr.coordinates(("i", "j"))
    got[0, 0] = 7
    assert arr.index[0, 0] == 0
