import numpy as np
import pytest

import nimblend as nb

DIMS = ("a", "b", "c", "d", "e")
# five dimensions of 2**30 members: 2**150 cells
SHAPE = (2**30,) * 5


def coords():
    return {d: nb.ProductCoord((2**30,)) for d in DIMS}


def test_full_raises_for_a_shape_beyond_the_int64_range():
    with pytest.raises(OverflowError, match="exceeds the int64 range"):
        nb.Domain.full(DIMS, coords())


def test_is_full_raises_for_a_shape_beyond_the_int64_range():
    held = nb.Domain(np.empty(0, dtype=np.int64), DIMS, coords(), SHAPE)
    with pytest.raises(OverflowError, match="exceeds the int64 range"):
        assert held.is_full


def test_the_length_of_a_product_coord_raises_beyond_the_int64_range():
    with pytest.raises(OverflowError, match="exceeds the int64 range"):
        len(nb.ProductCoord(SHAPE))
