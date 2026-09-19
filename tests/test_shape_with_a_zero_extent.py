import numpy as np
import pytest

import nimblend as nb

LARGE = ("a", "b", "c", "d", "e")


def no_entries(dims):
    # five dimensions of 2**30 members and "z" of 0 members: 0 cells
    coords = {d: nb.ProductCoord((2**30,)) for d in LARGE}
    coords["z"] = nb.ProductCoord((0,))
    labels = {d: np.empty((1, 0), dtype=np.int64) for d in dims}
    return nb.from_long(dims, coords, labels, np.array([], dtype=float))


@pytest.mark.parametrize("dims", [("z", *LARGE), (*LARGE, "z")])
def test_the_domain_over_a_zero_extent_has_no_members(dims):
    got = no_entries(dims).domain(dims)
    assert got.size == 0
    assert got.dims == dims
