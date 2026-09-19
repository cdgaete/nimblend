import numpy as np
import pytest

import nimblend as nb


@pytest.mark.parametrize("op", ["min", "max", "mean"])
def test_a_reduction_over_the_whole_array_reads_present_once(op, monkeypatch):
    coords = {"x": nb.StoredCoord(np.array(["a", "b"]))}
    arr = nb.DenseArray(np.array([1.0, np.nan]), coords, ("x",))
    reads = []

    def counted(self):
        # the presence of an array without a mask
        reads.append(op)
        return ~np.isnan(self.data)

    monkeypatch.setattr(nb.DenseArray, "present", property(counted))
    assert getattr(arr, op)() == 1.0
    assert reads == [op]
