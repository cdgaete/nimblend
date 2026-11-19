import numpy as np
import pytest

import nimblend as nb


def domain():
    coords = {
        "a": nb.StoredCoord(np.array(["x", "y", "z"])),
        "b": nb.ProductCoord((2,)),
    }
    index = np.array([[0, 0, 1, 2], [0, 1, 1, 0]])
    return nb.Domain.from_coordinates(("a", "b"), coords, index)


def test_take_returns_the_members_at_the_positions():
    taken = domain().take([1, 3])
    assert taken.coordinates().tolist() == [[0, 2], [1, 0]]
    assert (taken.dims, taken.shape) == (("a", "b"), (3, 2))
    assert taken.coords == domain().coords


def test_take_of_every_position_and_of_none():
    held = domain()
    assert held.take(np.arange(4)).coordinates().tolist() == held.coordinates().tolist()
    assert held.take(np.array([], dtype=np.int64)).size == 0


@pytest.mark.parametrize("positions", [[2, 1], [1, 1]])
def test_take_raises_for_positions_that_do_not_ascend(positions):
    with pytest.raises(
        ValueError, match="pass positions that ascend without repeating"
    ):
        domain().take(positions)


def test_take_raises_for_a_position_outside_the_domain():
    with pytest.raises(ValueError, match="outside a domain of 4 members"):
        domain().take([0, 4])
