import numpy as np
import pytest

from nimblend.coords import StoredCoord
from nimblend.domain import Domain
from nimblend.sparse import SparseArray
from reference import random_block, to_dense

DIMS = ("a", "b", "c", "d")


def build(rng, shape, nnz):
    """A random array over `shape`, and the dense form of the same entries."""
    idx, data = random_block(rng, shape, nnz)
    dims = DIMS[: len(shape)]
    coords = {name: StoredCoord(np.arange(shape[k])) for k, name in enumerate(dims)}
    arr = SparseArray(idx, data, coords, dims)
    values, present = to_dense(idx, data, shape)
    return arr, values, present


@pytest.mark.parametrize("seed", range(8))
def test_coordinates_match_the_dense_presence_positions(seed):
    rng = np.random.default_rng(seed)
    arr, _, present = build(rng, (3, 4, 5), 20)
    assert arr.coordinates().T.tolist() == np.argwhere(present).tolist()


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("prefix", [1, 2])
def test_domain_over_a_prefix_matches_the_dense_presence(seed, prefix):
    rng = np.random.default_rng(seed)
    shape = (3, 4, 5)
    arr, _, present = build(rng, shape, 20)
    got = arr.domain(DIMS[:prefix]).coordinates().T
    collapsed = present.any(axis=tuple(range(prefix, len(shape))))
    assert got.tolist() == np.argwhere(collapsed).tolist()


@pytest.mark.parametrize("seed", range(8))
def test_restrict_matches_masking_the_dense_form(seed):
    rng = np.random.default_rng(seed)
    shape = (3, 4, 5)
    arr, values, present = build(rng, shape, 20)
    keeper, _, keep_present = build(rng, shape[:2], 6)
    got = arr.restrict(keeper.domain())
    wanted = np.broadcast_to(keep_present[:, :, None], shape) & present
    assert np.allclose(got.to_dense(), np.where(wanted, values, 0.0))


@pytest.mark.parametrize("seed", range(8))
def test_expand_matches_repeating_the_dense_form(seed):
    rng = np.random.default_rng(seed)
    arr, values, _ = build(rng, (3, 4), 7)
    got = arr.expand(("z",), {"z": StoredCoord(np.arange(2))})
    assert np.allclose(got.to_dense(), np.repeat(values[:, :, None], 2, axis=2))


@pytest.mark.parametrize("seed", range(8))
def test_group_matches_gathering_the_present_rows_of_the_dense_form(seed):
    rng = np.random.default_rng(seed)
    shape = (3, 4, 5)
    arr, values, present = build(rng, shape, 20)
    got = arr.group(("a", "b"), "g")
    rows = present.reshape(shape[0] * shape[1], shape[2]).any(axis=1)
    assert np.allclose(got.to_dense(), values.reshape(-1, shape[2])[rows])


@pytest.mark.parametrize("seed", range(8))
def test_grouping_conserves_every_entry(seed):
    rng = np.random.default_rng(seed)
    arr, _, _ = build(rng, (3, 4, 5), 20)
    assert arr.group(("a", "b"), "g").nnz == arr.nnz


@pytest.mark.parametrize("seed", range(8))
def test_expanding_then_summing_the_new_dimension_scales_the_values(seed):
    rng = np.random.default_rng(seed)
    arr, values, _ = build(rng, (3, 4), 7)
    widened = arr.expand(("z",), {"z": StoredCoord(np.arange(3))})
    assert np.allclose(widened.sum("z").to_dense(), values * 3.0)


@pytest.mark.parametrize("seed", range(8))
def test_values_match_the_dense_form_at_the_entrys_own_coordinate(seed):
    rng = np.random.default_rng(seed)
    arr, dense, _ = build(rng, (3, 4, 5), 20)
    index = arr.coordinates()
    wanted = [dense[tuple(index[:, k])] for k in range(arr.nnz)]
    assert arr.values().tolist() == wanted


@pytest.mark.parametrize("seed", range(8))
def test_a_domain_from_labels_carries_what_the_arrays_own_domain_carries(seed):
    rng = np.random.default_rng(seed)
    shape = (3, 4)
    arr, _, _ = build(rng, shape, 7)
    dims = DIMS[: len(shape)]
    own = arr.domain(dims)
    got = Domain.from_labels(
        dims, {name: arr.coords[name] for name in dims}, own.labels()
    )
    assert list(got.codes) == list(own.codes)


@pytest.mark.parametrize("seed", range(8))
def test_expand_then_transpose_names_what_a_built_cross_product_names(seed):
    # the equivalence a caller crossing a domain with further dimensions
    # relies on: code arithmetic answers what an index matrix would
    rng = np.random.default_rng(seed)
    shape, extra = (3, 4), (2, 5)
    coords = {name: StoredCoord(np.arange(n)) for name, n in zip(DIMS, shape + extra)}
    held = ("a", "b")
    added = ("c", "d")
    keep = ("c", "a", "d", "b")
    index = np.unique(
        np.stack([rng.integers(0, n, 12) for n in shape]).astype(np.int32), axis=1
    )
    start = Domain.from_coordinates(held, coords, index)

    got = start.expand(added, coords).transpose(*keep)

    grid = np.array(np.meshgrid(*[np.arange(n) for n in extra], indexing="ij")).reshape(
        len(extra), -1
    )
    columns = {name: np.repeat(index[k], grid.shape[1]) for k, name in enumerate(held)}
    columns.update(
        {name: np.tile(grid[k], index.shape[1]) for k, name in enumerate(added)}
    )
    want = Domain.from_coordinates(
        keep, coords, np.stack([columns[name] for name in keep]).astype(np.int32)
    )
    assert got.dims == want.dims
    assert got.shape == want.shape
    assert list(got.codes) == list(want.codes)
