import time

import numpy as np
import pytest

from nimblend import kernel


@pytest.mark.parametrize(
    "shape", [(), (5,), (0,), (3, 4), (2, 3, 4), (3, 0, 2), (1, 1, 7)]
)
def test_cells_equal_the_unravelled_keys_of_every_cell(shape):
    got = kernel.cells(shape)
    want = kernel.unravel(np.arange(kernel.span(shape), dtype=np.int64), shape)
    assert got.dtype == np.int32
    assert got.shape == want.shape
    assert np.array_equal(got, want)


def best_of(runs, call):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        call()
        times.append(time.perf_counter() - start)
    return min(times)


def test_cells_cost_a_fraction_of_unravelling_every_key():
    # writing the grid costs about a thirteenth of dividing every key on this
    # shape; a third leaves room for a loaded machine
    shape = (1911, 2920)
    keys = np.arange(kernel.span(shape), dtype=np.int64)
    kernel.cells(shape)
    grid = best_of(5, lambda: kernel.cells(shape))
    divided = best_of(5, lambda: kernel.unravel(keys, shape))
    assert grid < divided / 3, (
        f"cells {grid * 1e3:.1f} ms, unravel {divided * 1e3:.1f} ms"
    )
