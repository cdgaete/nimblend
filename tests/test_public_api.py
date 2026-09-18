import re

import numpy as np
import pytest

import nimblend as nb


def test_the_module_exports_the_contract_and_the_implementation():
    assert set(nb.__all__) == {
        "Array",
        "DenseArray",
        "Domain",
        "EntryBuffer",
        "SparseArray",
        "combined_dims",
        "from_long",
        "from_dense",
        "is_canonical",
        "StoredCoord",
        "ProductCoord",
        "SubsetCoord",
        "__version__",
    }


def test_the_buffer_layers_own_arithmetic_is_not_on_the_public_module():
    # `ravel` and `unravel` convert an index matrix to and from the keys the
    # kernel sorts on. They mean nothing at the array layer, and publishing
    # them puts the layer below `Array` back within a consumer's reach.
    # `is_canonical` stays: it is the question `from_canonical` asks a caller
    # to answer, about buffers that caller built itself.
    assert not hasattr(nb, "ravel")
    assert not hasattr(nb, "unravel")
    assert nb.is_canonical is not None


def test_is_canonical_on_the_public_module_is_not_the_kernel_function():
    from nimblend import kernel

    assert nb.is_canonical is not kernel.is_canonical


def test_combined_dims_names_a_frame_without_an_array_to_read_it_from():
    assert nb.combined_dims(("P", "Q"), ("Q", "R")) == ("P", "Q", "R")
    with pytest.raises(ValueError, match="share no dimension"):
        nb.combined_dims(("P",), ("Q",))


def coords_tr():
    return {
        "t": nb.StoredCoord(np.array([2030, 2040])),
        "r": nb.StoredCoord(np.array(["DE", "FR"])),
    }


def test_from_long_reads_a_label_column_per_dimension_and_a_value_column():
    # the shape a columnar store hands over: one label column per dimension
    arr = nb.from_long(
        ("t", "r"),
        coords_tr(),
        {"t": np.array([2030, 2040, 2040]), "r": np.array(["DE", "DE", "FR"])},
        np.array([5.0, 6.0, 7.0]),
    )
    assert arr.dims == ("t", "r")
    assert arr.nnz == 3
    assert arr.to_dense()[1, 1] == 7.0


def test_from_long_carries_the_coordinates_it_is_given():
    # the coordinate a caller already holds, not a second one built from
    # labels: a dimension is named once and its lookup is built once
    held = coords_tr()
    arr = nb.from_long(
        ("t", "r"),
        held,
        {"t": np.array([2030]), "r": np.array(["FR"])},
        np.array([1.0]),
    )
    assert arr.coords["t"] is held["t"]
    assert arr.coords["r"] is held["r"]


def test_from_long_refuses_a_value_column_of_the_wrong_length():
    with pytest.raises(ValueError, match="length"):
        nb.from_long(
            ("t",),
            {"t": nb.StoredCoord(np.array([2030, 2040]))},
            {"t": np.array([2030, 2040])},
            np.array([1.0]),
        )


def test_from_long_refuses_a_dimension_carrying_no_coordinate():
    with pytest.raises(ValueError, match="no coordinate"):
        nb.from_long(
            ("t", "r"),
            {"t": nb.StoredCoord(np.array([2030]))},
            {"t": np.array([2030]), "r": np.array(["DE"])},
            np.array([1.0]),
        )


def test_from_long_refuses_a_dimension_carrying_no_label_column():
    with pytest.raises(ValueError, match="no label column"):
        nb.from_long(("t", "r"), coords_tr(), {"t": np.array([2030])}, np.array([1.0]))


def test_from_long_refuses_a_label_the_coordinate_does_not_carry():
    with pytest.raises(KeyError, match="is not in the coordinate"):
        nb.from_long(
            ("t",),
            {"t": nb.StoredCoord(np.array([2030, 2040]))},
            {"t": np.array([2050])},
            np.array([1.0]),
        )


def test_from_long_reads_an_index_matrix_for_a_generated_coordinate():
    # two labels over (2, 3): the multi-indices (0, 2) and (1, 0)
    arr = nb.from_long(
        ("k",),
        {"k": nb.ProductCoord((2, 3))},
        {"k": np.array([[0, 1], [2, 0]])},
        np.array([1.0, 2.0]),
    )
    assert arr.coordinates().tolist() == [[2, 3]]
    assert arr.values().tolist() == [1.0, 2.0]


def test_from_long_counts_the_labels_of_a_generated_coordinate():
    message = (
        "label columns have lengths {'k': 2} and the value column has length 4; "
        "pass columns of equal length"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        nb.from_long(
            ("k",),
            {"k": nb.ProductCoord((2, 3))},
            {"k": np.array([[0, 1], [2, 0]])},
            np.array([1.0, 2.0, 3.0, 4.0]),
        )


def test_from_long_and_from_labels_report_the_same_label_counts():
    coords = {"k": nb.ProductCoord((2, 3)), "t": nb.StoredCoord(np.arange(3))}
    labels = {"k": np.array([[0, 1], [2, 0]]), "t": np.arange(3)}
    with pytest.raises(ValueError, match=re.escape("{'k': 2, 't': 3}")):
        nb.from_long(("k", "t"), coords, labels, np.ones(2))
    with pytest.raises(ValueError, match=re.escape("{'k': 2, 't': 3}")):
        nb.Domain.from_labels(("k", "t"), coords, labels)


def test_from_long_over_no_dimension_holds_one_value():
    arr = nb.from_long((), {}, {}, np.array([5.0]))
    assert arr.dims == ()
    assert arr.values().tolist() == [5.0]


def test_from_dense_round_trips():
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    labels = {"x": np.array(["a", "b"]), "y": np.array([1, 2, 3])}
    assert np.array_equal(nb.from_dense(values, labels).to_dense(), values)


def test_a_sparse_array_is_an_array():
    arr = nb.from_dense(np.zeros((1, 1)), {"x": np.array(["a"]), "y": np.array([1])})
    assert isinstance(arr, nb.Array)


def test_is_canonical_answers_whether_buffers_are_in_canonical_order():
    ordered = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int32)
    assert nb.is_canonical(ordered, (2, 2))
    assert not nb.is_canonical(ordered[:, ::-1].copy(), (2, 2))


def test_is_canonical_reads_an_integer_index_matrix_given_as_a_list():
    assert nb.is_canonical([[0, 0, 1], [0, 1, 0]], (2, 2))
    assert not nb.is_canonical([[1, 0], [0, 0]], (2, 2))


def test_is_canonical_raises_for_an_index_that_is_not_two_dimensional():
    message = "index has 1 dimension(s); pass a 2-D index matrix"
    with pytest.raises(ValueError, match=re.escape(message)):
        nb.is_canonical(np.array([0, 1], dtype=np.int32), (2,))


def test_is_canonical_raises_for_an_index_that_is_not_integer():
    message = "index has dtype float64; pass an integer index matrix"
    with pytest.raises(TypeError, match=re.escape(message)):
        nb.is_canonical(np.array([[0.0, 1.0]]), (2,))


def test_is_canonical_raises_for_one_row_per_extent_missing():
    message = "index has 1 row(s) and shape (2, 2) has 2 extent(s)"
    with pytest.raises(ValueError, match=re.escape(message)):
        nb.is_canonical(np.array([[0, 1]], dtype=np.int32), (2, 2))


def test_is_canonical_raises_for_a_position_outside_its_extent():
    # (0, 5) over (2, 3) ravels to the key of (1, 2)
    message = (
        "row 1 of the index has positions from 0 to 5 and extent 3; pass "
        "positions from 0 to 2"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        nb.is_canonical(np.array([[0, 1], [0, 5]], dtype=np.int32), (2, 3))
    with pytest.raises(ValueError, match="positions from -1 to 0"):
        nb.is_canonical(np.array([[-1, 0]], dtype=np.int32), (2,))


def test_is_canonical_reads_an_empty_index():
    assert nb.is_canonical(np.empty((2, 0), dtype=np.int32), (2, 2))


def test_is_canonical_is_what_from_canonical_asks_a_caller_to_check():
    index = np.array([[0, 1], [1, 0]], dtype=np.int32)
    assert nb.is_canonical(index, (2, 2))
    arr = nb.SparseArray.from_canonical(
        index,
        np.array([3.0, 4.0]),
        {
            "x": nb.StoredCoord(np.array(["a", "b"])),
            "y": nb.StoredCoord(np.array([1, 2])),
        },
        ("x", "y"),
    )
    assert arr.nnz == 2


def test_a_domain_answers_the_frame_a_consumer_reads():
    # `dims`, `shape`, `coords` and `size` are the frame, and are contract
    # rather than representation. `codes` is not among them: it is the raw
    # ravelled members of the layer below, as an array's `.index` is its raw
    # buffer, and every question a consumer asks of it has a reader above it.
    coords = {
        "t": nb.StoredCoord(np.array([2030, 2040])),
        "r": nb.StoredCoord(np.array(["DE", "FR"])),
    }
    domain = nb.Domain.full(("t", "r"), coords)

    assert domain.dims == ("t", "r")
    assert domain.shape == (2, 2)
    assert domain.coords == coords
    assert domain.size == 4


def test_a_consumer_numbers_a_domains_members_without_reading_its_codes():
    # the reader above `codes`: `as_coord` numbers the members, and the
    # coordinate it answers places an entry the same way the array `identity`
    # builds does. A consumer never hands raw codes to `SubsetCoord` itself.
    coords = {"t": nb.StoredCoord(np.array([2030, 2040, 2050]))}
    domain = nb.Domain.full(("t",), coords)

    numbering = domain.as_coord(10)
    assert len(numbering) == 3
    assert numbering.to_position(domain.coordinates()).tolist() == [10, 11, 12]

    paired = domain.identity("k", nb.ProductCoord((20,)), start=10)
    assert paired.coordinates()[1].tolist() == [10, 11, 12]


def test_a_domain_answers_with_an_array_over_its_own_members():
    # the reader above `codes` for a caller holding one value per member: no
    # index is built by the caller, and none is read back
    coords = {"t": nb.StoredCoord(np.array([2030, 2040, 2050]))}
    domain = nb.Domain.full(("t",), coords)
    arr = domain.array(np.array([1.0, 2.0, 3.0]))
    assert arr.dims == ("t",)
    assert arr.values().tolist() == [1.0, 2.0, 3.0]


def test_the_package_ships_its_type_marker():
    from pathlib import Path

    assert (Path(nb.__file__).parent / "py.typed").is_file()


def test_no_module_of_the_package_imports_scipy():
    from pathlib import Path

    root = Path(nb.__file__).parent
    offenders = [str(p) for p in root.rglob("*.py") if "scipy" in p.read_text()]
    assert offenders == [], offenders
