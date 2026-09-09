import numpy as np
import pytest

from nimblend import kernel
from reference import random_block


def test_ravel_matches_numpy_c_order():
    idx = np.array([[0, 1, 2], [3, 0, 1]], dtype=np.int32)
    keys = kernel.ravel(idx, (3, 4))
    assert keys.dtype == np.int64
    assert list(keys) == list(np.ravel_multi_index((idx[0], idx[1]), (3, 4)))


def test_unravel_inverts_ravel():
    rng = np.random.default_rng(1)
    shape = (7, 5, 3)
    idx, _ = random_block(rng, shape, 60)
    back = kernel.unravel(kernel.ravel(idx, shape), shape)
    assert back.dtype == np.int32
    assert np.array_equal(back, idx)


def test_ravel_of_one_dimension_is_the_index_itself():
    idx = np.array([[4, 0, 2]], dtype=np.int32)
    assert list(kernel.ravel(idx, (5,))) == [4, 0, 2]


def test_ravel_refuses_a_shape_product_past_int64():
    idx = np.zeros((3, 1), dtype=np.int32)
    with pytest.raises(OverflowError, match="exceeds"):
        kernel.ravel(idx, (2**40, 2**40, 2**40))


def test_ravelling_over_no_dimensions_puts_every_entry_on_one_key():
    # the product of no sizes is one, so there is one coordinate to land on
    idx = np.empty((0, 3), dtype=np.int32)
    assert kernel.ravel(idx, ()).tolist() == [0, 0, 0]


def test_ravelling_no_entries_over_no_dimensions_is_empty():
    assert kernel.ravel(np.empty((0, 0), dtype=np.int32), ()).size == 0


def test_unravel_inverts_it_to_an_index_of_no_rows():
    keys = kernel.ravel(np.empty((0, 3), dtype=np.int32), ())
    assert kernel.unravel(keys, ()).shape == (0, 3)
