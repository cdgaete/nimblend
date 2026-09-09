import itertools

import numpy as np
import pytest

from nimblend import DenseArray, SparseArray
from nimblend.sparse import combined_dims

LABELS = {n: np.array([f"{n}{i}" for i in range(2)]) for n in "PQR"}


def dense(dims, seed=0):
    """A dense array over `dims`, distinct values."""
    shape = (2,) * len(dims)
    values = np.arange(1.0, 2 ** len(dims) + 1).reshape(shape) + seed
    return DenseArray.from_dense(values, {d: LABELS[d] for d in dims})


def sparse(dims, seed=0):
    return SparseArray.from_dense(
        dense(dims, seed).to_dense(), {d: LABELS[d] for d in dims}
    )


SHAPES = [("P",), ("P", "Q"), ("Q", "P"), ("P", "Q", "R"), ("Q",), ("Q", "R")]


def test_the_frame_a_result_carries_is_read_from_the_dimensions_alone():
    assert combined_dims(("P",), ("P",)) == ("P",)
    assert combined_dims(("P",), ("P", "Q")) == ("P", "Q")
    assert combined_dims(("P", "Q"), ("P",)) == ("P", "Q")
    assert combined_dims(("P", "Q"), ("Q", "R")) == ("P", "Q", "R")


def test_frames_sharing_no_dimension_have_nothing_to_align_on():
    with pytest.raises(ValueError, match="share no dimension"):
        combined_dims(("P",), ("Q",))


@pytest.mark.parametrize("left,right", list(itertools.product(SHAPES, SHAPES)))
def test_every_operator_carries_the_frame_the_rule_names(left, right):
    # one rule, read by all four: a result's frame never depends on which
    # operator produced it
    if not set(left) & set(right):
        pytest.skip("disjoint frames are refused, which another test states")
    wanted = combined_dims(left, right)
    a, b = sparse(left), sparse(right, seed=1)
    for got in (a + b, a - b, a * b, a / b):
        assert got.dims == wanted, (left, right)


@pytest.mark.parametrize("left,right", list(itertools.product(SHAPES, SHAPES)))
def test_both_implementations_answer_the_ladder_alike(left, right):
    if not set(left) & set(right):
        pytest.skip("disjoint frames are refused, which another test states")
    for op in ("__add__", "__sub__", "__mul__", "__truediv__"):
        want = getattr(sparse(left), op)(sparse(right, seed=1))
        got = getattr(dense(left), op)(dense(right, seed=1))
        assert got.dims == want.dims, (left, right, op)
        assert np.allclose(got.to_dense(), want.to_dense()), (left, right, op)


def test_an_addition_over_a_wider_frame_replicates_the_narrower_operand():
    # the narrower operand supplies a term at every coordinate of the wider
    # frame, which is the operation and is what makes it cost the frame
    got = sparse(("P",)) + sparse(("P", "Q"), seed=1)
    assert got.dims == ("P", "Q")
    assert np.allclose(
        got.to_dense(),
        dense(("P",)).to_dense()[:, None] + dense(("P", "Q"), seed=1).to_dense(),
    )


def test_a_quotient_over_a_wider_frame_divides_every_coordinate():
    got = sparse(("P", "Q"), seed=1) / sparse(("P",))
    assert got.dims == ("P", "Q")
    assert np.allclose(
        got.to_dense(),
        dense(("P", "Q"), seed=1).to_dense() / dense(("P",)).to_dense()[:, None],
    )


@pytest.mark.parametrize("op", ["__add__", "__sub__", "__mul__", "__truediv__"])
def test_no_operator_combines_frames_that_share_no_dimension(op):
    for a, b in ((sparse(("P",)), sparse(("Q",))), (dense(("P",)), dense(("Q",)))):
        with pytest.raises(ValueError, match="share no dimension"):
            getattr(a, op)(b)


def test_conforming_keeps_the_absence_both_operands_declare():
    a, b = sparse(("P",)), sparse(("P", "Q"), seed=1)
    assert (a + b).absence == "empty"
    assert (a.as_unknown() + b.as_unknown()).absence == "unknown"


def test_two_arrays_declaring_different_absence_do_not_combine():
    a, b = sparse(("P",)), sparse(("P", "Q"), seed=1).as_unknown()
    with pytest.raises(ValueError, match="absence"):
        a + b


@pytest.mark.parametrize("dims", SHAPES)
def test_a_power_by_a_number_raises_every_entry_alike(dims):
    exponent = 2.0
    want = dense(dims).to_dense() ** exponent
    for arr in (sparse(dims), dense(dims)):
        got = arr**exponent
        assert got.dims == dims
        assert np.allclose(got.to_dense(), want)


def test_a_power_by_an_array_is_not_an_operation_an_array_offers():
    with pytest.raises(TypeError):
        sparse(("P",)) ** sparse(("P",))
    with pytest.raises(TypeError):
        dense(("P",)) ** dense(("P",))


def test_a_power_leaves_an_absent_coordinate_absent():
    held = SparseArray.from_dense(
        np.array([[1.0, 2.0], [3.0, 4.0]]), {"P": LABELS["P"], "Q": LABELS["Q"]}
    ).restrict(
        SparseArray.from_dense(
            np.array([[1.0, 0.0], [0.0, 1.0]]), {"P": LABELS["P"], "Q": LABELS["Q"]}
        ).domain()
    )
    assert (held**2.0).nnz == held.nnz


@pytest.mark.parametrize("dims", SHAPES)
def test_a_number_divided_by_an_array_answers_alike_in_both(dims):
    want = 2.0 / dense(dims).to_dense()
    for arr in (sparse(dims), dense(dims)):
        got = 2.0 / arr
        assert got.dims == dims
        assert np.allclose(got.to_dense(), want)
