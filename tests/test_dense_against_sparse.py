"""The two implementations answer the contract alike, over random arrays."""

import numpy as np
import pytest

from nimblend import DenseArray, SparseArray, StoredCoord

LABELS = {"x": np.array(["a", "b", "c"]), "y": np.array([10, 20, 30, 40])}


def pair(seed, density, absence):
    """The same logical array as a SparseArray and as a DenseArray."""
    rng = np.random.default_rng(seed)
    values = np.round(rng.random((3, 4)) * 10, 3)
    mask = rng.random((3, 4)) < density
    mask[0, 0] = True
    index = np.stack(np.nonzero(mask)).astype(np.int32)
    sparse = SparseArray.from_canonical(
        index,
        values[mask],
        {n: DenseArray.from_dense(values, LABELS).coords[n] for n in LABELS},
        ("x", "y"),
        absence,
    )
    if absence == "unknown":
        tagged = np.where(mask, values, np.nan)
        dense = DenseArray.from_dense(tagged, LABELS, "unknown")
    else:
        dense = DenseArray.from_dense(values, LABELS, "empty", mask)
    return sparse, dense


CASES = [
    (seed, density, absence)
    for seed in (0, 1, 2)
    for density in (1.0, 0.6)
    for absence in ("empty", "unknown")
]


@pytest.mark.parametrize("seed,density,absence", CASES)
def test_the_two_implementations_carry_the_same_entries(seed, density, absence):
    sparse, dense = pair(seed, density, absence)
    assert dense.nnz == sparse.nnz
    assert np.array_equal(dense.coordinates(), sparse.coordinates())
    assert np.allclose(dense.values(), sparse.values())


@pytest.mark.parametrize("seed,density,absence", CASES)
def test_the_two_implementations_answer_the_same_domain(seed, density, absence):
    sparse, dense = pair(seed, density, absence)
    for dims in (("x",), ("y",), ("x", "y")):
        assert np.array_equal(dense.domain(dims).codes, sparse.domain(dims).codes)


@pytest.mark.parametrize("seed,density,absence", CASES)
def test_the_two_implementations_reduce_alike(seed, density, absence):
    sparse, dense = pair(seed, density, absence)
    policy = {"skip": True} if absence == "unknown" else {}
    assert np.isclose(dense.sum(**policy), sparse.sum(**policy))
    assert np.isclose(dense.min(**policy), sparse.min(**policy))
    assert np.isclose(dense.max(**policy), sparse.max(**policy))
    for dim in ("x", "y"):
        mine = dense.sum(dim, **policy)
        theirs = sparse.sum(dim, **policy)
        assert np.allclose(mine.to_dense(fill=0.0), theirs.to_dense(fill=0.0))


@pytest.mark.parametrize("seed,density,absence", CASES)
def test_the_two_implementations_select_and_transpose_alike(seed, density, absence):
    sparse, dense = pair(seed, density, absence)
    assert np.allclose(
        dense.sel({"x": "b"}).to_dense(fill=0.0),
        sparse.sel({"x": "b"}).to_dense(fill=0.0),
    )
    assert np.allclose(
        dense.transpose("y", "x").to_dense(fill=0.0),
        sparse.transpose("y", "x").to_dense(fill=0.0),
    )
    assert dense.rename({"x": "z"}).dims == sparse.rename({"x": "z"}).dims


@pytest.mark.parametrize("seed,density,absence", CASES)
def test_the_two_implementations_restrict_alike(seed, density, absence):
    sparse, dense = pair(seed, density, absence)
    rows = sparse.domain(("x",))
    assert np.allclose(
        dense.restrict(rows).to_dense(fill=0.0),
        sparse.restrict(rows).to_dense(fill=0.0),
    )
    assert dense.restrict(rows).nnz == sparse.restrict(rows).nnz


@pytest.mark.parametrize("seed,density", [(0, 1.0), (1, 0.6)])
def test_the_two_implementations_add_alike(seed, density):
    sparse_a, dense_a = pair(seed, density, "empty")
    sparse_b, dense_b = pair(seed + 10, density, "empty")
    assert np.allclose((dense_a + dense_b).to_dense(), (sparse_a + sparse_b).to_dense())
    assert (dense_a + dense_b).nnz == (sparse_a + sparse_b).nnz
    assert np.allclose((dense_a * dense_b).to_dense(), (sparse_a * sparse_b).to_dense())
    assert (dense_a * dense_b).nnz == (sparse_a * sparse_b).nnz


@pytest.mark.parametrize("seed,density,absence", CASES)
def test_the_two_implementations_shift_and_roll_alike(seed, density, absence):
    sparse, dense = pair(seed, density, absence)
    for shifts in ({"x": 1}, {"x": -1}, {"y": 2}, {"x": 1, "y": -1}, {"x": 0}):
        for mine, theirs in (
            (dense.shift(shifts), sparse.shift(shifts)),
            (dense.roll(shifts), sparse.roll(shifts)),
        ):
            assert mine.nnz == theirs.nnz, (shifts, mine.nnz, theirs.nnz)
            assert np.allclose(mine.to_dense(fill=0.0), theirs.to_dense(fill=0.0))


@pytest.mark.parametrize("seed,density,absence", CASES)
def test_the_two_implementations_conform_alike(seed, density, absence):
    sparse, dense = pair(seed, density, absence)
    wanted = {"x": np.array(["c", "a"]), "y": np.array([40, 10, 20])}
    for dims in (["x", "y"], ["y", "x"]):
        mine, theirs = dense.conform(dims, wanted), sparse.conform(dims, wanted)
        assert mine.dims == theirs.dims
        assert mine.nnz == theirs.nnz
        assert np.allclose(mine.to_dense(fill=0.0), theirs.to_dense(fill=0.0))


@pytest.mark.parametrize("seed,density,absence", CASES)
def test_the_two_implementations_take_a_scalar_alike(seed, density, absence):
    sparse, dense = pair(seed, density, absence)
    for mine, theirs in (
        (dense * 3, sparse * 3),
        (3 * dense, 3 * sparse),
        (dense + 3, sparse + 3),
        (3 + dense, 3 + sparse),
        (dense - 3, sparse - 3),
        (3 - dense, 3 - sparse),
        (dense / 2, sparse / 2),
        (-dense, -sparse),
    ):
        assert mine.nnz == theirs.nnz
        assert np.allclose(mine.to_dense(fill=0.0), theirs.to_dense(fill=0.0))


@pytest.mark.parametrize("seed,density", [(0, 1.0), (1, 0.6)])
def test_the_two_implementations_subtract_alike(seed, density):
    sparse_a, dense_a = pair(seed, density, "empty")
    sparse_b, dense_b = pair(seed + 10, density, "empty")
    assert np.allclose((dense_a - dense_b).to_dense(), (sparse_a - sparse_b).to_dense())
    assert (dense_a - dense_b).nnz == (sparse_a - sparse_b).nnz


@pytest.mark.parametrize("seed,density,absence", CASES)
def test_the_two_implementations_refuse_a_repeated_label_alike(seed, density, absence):
    sparse, dense = pair(seed, density, absence)
    wanted = {"x": np.array(["a", "a"]), "y": LABELS["y"]}
    for arr in (dense, sparse):
        with pytest.raises(ValueError, match="is named twice for dimension"):
            arr.conform(["x", "y"], wanted)


FRAMES = {"x": LABELS["x"], "y": LABELS["y"], "z": np.array(["p", "q"])}


def framed(dims, seed, density, absence):
    """The same logical array over `dims`, as a SparseArray and a DenseArray."""
    rng = np.random.default_rng(seed)
    shape = tuple(len(FRAMES[d]) for d in dims)
    values = np.round(rng.random(shape) * 5 + 1, 3)
    mask = rng.random(shape) < density
    mask.flat[0] = True
    coords = {d: StoredCoord(FRAMES[d]) for d in dims}
    sparse = SparseArray.from_canonical(
        np.stack(np.nonzero(mask)).astype(np.int32),
        values[mask],
        coords,
        dims,
        absence,
    )
    if absence == "unknown":
        dense = DenseArray.from_dense(
            np.where(mask, values, np.nan), {d: FRAMES[d] for d in dims}, "unknown"
        )
    else:
        dense = DenseArray.from_dense(
            values, {d: FRAMES[d] for d in dims}, "empty", mask
        )
    return sparse, dense


# every branch the product routes through: one frame, one nesting inside the
# other in both orders, and frames that share some dimensions in both orders
PRODUCTS = [
    (("x", "y"), ("x", "y")),
    (("y",), ("x", "y")),
    (("x", "y"), ("y",)),
    (("y",), ("x", "y", "z")),
    (("x", "y"), ("y", "z")),
    (("y", "z"), ("x", "y")),
]


@pytest.mark.parametrize("left,right", PRODUCTS)
@pytest.mark.parametrize("absence", ["empty", "unknown"])
def test_the_two_implementations_multiply_over_a_wider_frame_alike(
    left, right, absence
):
    sparse_a, dense_a = framed(left, 1, 0.7, absence)
    sparse_b, dense_b = framed(right, 2, 0.7, absence)
    mine, theirs = dense_a * dense_b, sparse_a * sparse_b
    assert mine.dims == theirs.dims
    assert mine.nnz == theirs.nnz
    assert np.allclose(mine.to_dense(fill=0.0), theirs.to_dense(fill=0.0))


@pytest.mark.parametrize("left,right", PRODUCTS)
def test_a_mixed_product_answers_what_the_sparse_product_answers(left, right):
    # a product intersects presence, so a mixed product carries at most the
    # sparse operand's entries and is answered as a SparseArray
    sparse_a, dense_a = framed(left, 1, 0.7, "empty")
    sparse_b, dense_b = framed(right, 2, 0.7, "empty")
    wanted = sparse_a * sparse_b
    for got in (dense_a * sparse_b, sparse_a * dense_b):
        assert isinstance(got, SparseArray)
        assert got.dims == wanted.dims
        assert np.allclose(got.to_dense(fill=0.0), wanted.to_dense(fill=0.0))


def test_frames_sharing_no_dimension_are_refused_by_both():
    sparse_a, dense_a = framed(("x",), 1, 1.0, "empty")
    sparse_b, dense_b = framed(("z",), 2, 1.0, "empty")
    for a, b in ((dense_a, dense_b), (sparse_a, sparse_b)):
        with pytest.raises(ValueError, match="share no dimension"):
            a * b


def test_a_mixed_sum_over_one_frame_answers_as_the_sparse_sum_does():
    sparse_a, dense_a = framed(("x", "y"), 1, 0.7, "empty")
    sparse_b, dense_b = framed(("x", "y"), 2, 0.7, "empty")
    wanted = sparse_a + sparse_b
    for got in (dense_a + sparse_b, sparse_a + dense_b):
        assert isinstance(got, SparseArray)
        assert np.allclose(got.to_dense(), wanted.to_dense())
