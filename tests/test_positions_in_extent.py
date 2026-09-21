import re

import numpy as np
import pytest

import nimblend as nb

X = nb.StoredCoord(np.array(["a", "b"]))
Y = nb.StoredCoord(np.array(["p", "q", "r"]))


def test_a_coordinate_takes_no_start():
    # positions along a dimension run from 0 to its extent less one; a
    # numbering inside a larger dimension is the start of group and identity
    with pytest.raises(TypeError):
        nb.ProductCoord((2,), start=3)
    with pytest.raises(TypeError):
        nb.SubsetCoord(np.array([0, 1]), (2,), start=3)
    with pytest.raises(TypeError):
        nb.Domain.full(("x",), {"x": X}).as_coord(3)


@pytest.mark.parametrize("cell", [(2, 0), (0, 3), (-1, 0), (0, -1)])
def test_a_product_coord_raises_for_a_cell_outside_its_sizes(cell):
    # (0, 3) over sizes (2, 3) ravels to 3, the position of cell (1, 0)
    expected = re.escape(f"cell {cell} is not in the product of sizes (2, 3)")
    with pytest.raises(KeyError, match=expected):
        nb.ProductCoord((2, 3)).to_position(np.array(cell).reshape(2, 1))


def test_a_subset_coord_raises_for_a_cell_outside_its_sizes():
    # (0, 5) over sizes (2, 3) ravels to 5, the code of member (1, 2)
    coord = nb.SubsetCoord(np.array([0, 5]), (2, 3))
    with pytest.raises(KeyError, match=re.escape("cell (0, 5) is not in the subset")):
        coord.to_position(np.array([[0], [5]]))


@pytest.mark.parametrize(
    "coord", [nb.ProductCoord((2,)), nb.SubsetCoord(np.array([0, 1]), (2,))]
)
@pytest.mark.parametrize("index", [np.array([[0], [0]]), np.array([0])])
def test_a_generated_coord_raises_for_an_index_without_a_row_per_axis(coord, index):
    expected = re.escape(
        f"index has shape {index.shape}; pass a 2-D index matrix with 1 row(s), "
        f"one per axis of sizes (2,)"
    )
    with pytest.raises(ValueError, match=expected):
        coord.to_position(index)


def test_from_long_raises_for_a_cell_outside_a_product_coord():
    expected = re.escape("cell (5,) is not in the product of sizes (2,)")
    with pytest.raises(KeyError, match=expected):
        nb.from_long(
            ("k",), {"k": nb.ProductCoord((2,))}, {"k": np.array([[5]])}, [1.0]
        )


@pytest.mark.parametrize("position", [2, -1])
def test_the_constructor_raises_for_a_position_outside_the_extent(position):
    # numpy reads -1 as the last cell, and 2 is past the end
    expected = re.escape(
        f"row 0 of the index has positions from {position} to {position} and extent 2"
    )
    with pytest.raises(ValueError, match=expected):
        nb.SparseArray(np.array([[position]]), np.array([1.0]), {"x": X}, ("x",))


@pytest.mark.parametrize("columns, values", [(1, 2), (2, 1)])
def test_the_constructor_raises_for_one_value_per_column_missing(columns, values):
    expected = re.escape(
        f"index has {columns} column(s) and the values have shape ({values},)"
    )
    with pytest.raises(ValueError, match=expected):
        nb.SparseArray(
            np.arange(columns).reshape(1, columns), np.ones(values), {"x": X}, ("x",)
        )


def test_the_constructor_raises_for_an_index_without_a_row_per_dimension():
    expected = re.escape("index has 2 row(s) and shape (2,) has 1 extent(s)")
    with pytest.raises(ValueError, match=expected):
        nb.SparseArray(np.array([[0], [0]]), np.array([1.0]), {"x": X}, ("x",))


def test_the_constructor_raises_for_an_index_that_is_not_2d():
    with pytest.raises(ValueError, match=re.escape("index has 1 dimension(s)")):
        nb.SparseArray(np.array([0, 1]), np.array([1.0, 2.0]), {"x": X}, ("x",))


def test_the_constructor_accepts_an_array_over_no_dimensions():
    # a frame over no dimensions is one cell, and its index has no rows
    arr = nb.SparseArray(np.empty((0, 1), dtype=np.int32), np.array([4.0]), {}, ())
    assert arr.to_dense() == 4.0


@pytest.mark.parametrize("code", [6, -1])
def test_a_domain_raises_for_a_code_outside_its_shape(code):
    # over shape (2, 3) the codes run from 0 to 5; code 6 would wrap to (0, 0)
    expected = re.escape(
        f"domain codes run from {code} to {code} and shape (2, 3) has 6 cells"
    )
    with pytest.raises(ValueError, match=expected):
        nb.Domain(np.array([code]), ("x", "y"), {"x": X, "y": Y}, (2, 3))


def test_from_coordinates_raises_for_a_position_outside_its_extent():
    # (0, 5) over shape (2, 3) ravels to 5, the code of member (1, 2)
    expected = re.escape("row 1 of the index has positions from 5 to 5 and extent 3")
    with pytest.raises(ValueError, match=expected):
        nb.Domain.from_coordinates(("x", "y"), {"x": X, "y": Y}, [[0], [5]])
