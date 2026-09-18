import re

import numpy as np
import pytest

from nimblend.coords import ProductCoord, StoredCoord
from nimblend.domain import Domain

COORDS = {
    "x": StoredCoord(np.array(["a", "b", "c"])),
    "y": StoredCoord(np.array([10, 20])),
    "z": StoredCoord(np.array(["p", "q", "r", "s"])),
    "w": ProductCoord((2, 3)),
}


def random_domain(rng, dims, density):
    full = Domain.full(dims, COORDS)
    keep = rng.random(full.size) < density
    return Domain(full.codes[keep], dims, COORDS, full.shape)


def reference(left, right):
    wide = left.expand(right.dims, right.coords)
    other = right.expand(left.dims, left.coords).transpose(*left.dims, *right.dims)
    return wide.intersect(other)


PAIRS = [
    (("x",), ("z",)),
    (("x", "y"), ("z",)),
    (("z",), ("y", "x")),
    (("x",), ("w", "y")),
    ((), ("x",)),
]


@pytest.mark.parametrize("left_dims,right_dims", PAIRS)
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("density", [1.0, 0.5, 0.0])
def test_cross_is_every_pair_of_members(left_dims, right_dims, seed, density):
    rng = np.random.default_rng(seed)
    left = random_domain(rng, left_dims, density)
    right = random_domain(rng, right_dims, 0.6)
    got = left.cross(right)
    wanted = reference(left, right)
    assert got.dims == left_dims + right_dims
    assert got.shape == left.shape + right.shape
    assert got.coords == wanted.coords
    assert list(got.codes) == list(wanted.codes)
    assert got.size == left.size * right.size


def test_cross_orders_the_left_members_first():
    left = Domain.from_labels(("x",), COORDS, {"x": np.array(["a", "c"])})
    right = Domain.from_labels(("y",), COORDS, {"y": np.array([10, 20])})
    got = left.cross(right).labels()
    assert got["x"].tolist() == ["a", "a", "c", "c"]
    assert got["y"].tolist() == [10, 20, 10, 20]


def test_cross_with_an_empty_domain_is_empty():
    left = Domain.full(("x",), COORDS)
    empty = Domain(np.empty(0, dtype=np.int64), ("z",), COORDS, (4,))
    assert left.cross(empty).size == 0
    assert empty.cross(left).size == 0
    assert empty.cross(left).dims == ("z", "x")


def test_cross_raises_for_a_shared_dimension():
    message = (
        "domains over ('x', 'y') and ('y', 'z') share dimension(s) ['y']; pass "
        "domains over disjoint dimensions"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        Domain.full(("x", "y"), COORDS).cross(Domain.full(("y", "z"), COORDS))


def test_cross_raises_overflow_for_a_frame_beyond_the_int64_range():
    huge = {"u": ProductCoord((2**40,)), "v": ProductCoord((2**40,))}
    left = Domain(np.array([0], dtype=np.int64), ("u",), huge, (2**40,))
    right = Domain(np.array([0], dtype=np.int64), ("v",), huge, (2**40,))
    with pytest.raises(OverflowError, match="exceeds the int64 range"):
        left.cross(right)
