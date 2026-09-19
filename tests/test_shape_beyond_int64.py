import numpy as np
import pytest

import nimblend as nb

DIMS = ("a", "b", "c", "d", "e")


def beyond_int64(absence="empty"):
    # five dimensions of 2**30 members: 2**150 cells and no entries
    coords = {d: nb.ProductCoord((2**30,)) for d in DIMS}
    labels = {d: np.empty((1, 0), dtype=np.int64) for d in DIMS}
    return nb.from_long(DIMS, coords, labels, np.array([], dtype=float), absence)


@pytest.mark.parametrize("op", ["min", "max", "mean"])
def test_a_reduction_with_fill_raises_for_a_shape_beyond_the_int64_range(op):
    arr = beyond_int64()
    with pytest.raises(OverflowError, match="exceeds the int64 range"):
        getattr(arr, op)(fill=0.0)


def test_to_dense_raises_for_an_unknown_array_beyond_the_int64_range():
    arr = beyond_int64(absence="unknown")
    with pytest.raises(OverflowError, match="exceeds the int64 range"):
        arr.to_dense()
