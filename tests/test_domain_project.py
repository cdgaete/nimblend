import numpy as np
import pytest

import nimblend as nb

COORDS = {
    "x": nb.StoredCoord(np.array(["a", "b", "c"])),
    "y": nb.StoredCoord(np.array([10, 20])),
}


def domain(cells):
    index = np.array(cells, dtype=np.int32).T.reshape(2, -1)
    return nb.Domain.from_coordinates(("x", "y"), COORDS, index)


def test_a_projection_contains_each_coordinate_over_the_dimensions_once():
    got = domain([(0, 0), (0, 1), (2, 1)]).project(("x",))
    assert got.dims == ("x",)
    assert got.shape == (3,)
    assert got.coords == {"x": COORDS["x"]}
    assert got.labels()["x"].tolist() == ["a", "c"]


def test_a_projection_follows_the_order_of_the_dimensions_given():
    got = domain([(0, 1), (2, 0)]).project(("y", "x"))
    assert got.dims == ("y", "x")
    assert got.coordinates().tolist() == [[0, 1], [2, 0]]


def test_a_projection_onto_every_dimension_returns_the_members():
    held = domain([(0, 1), (2, 0)])
    assert (
        held.project(("x", "y")).coordinates().tolist() == held.coordinates().tolist()
    )


def test_a_domain_with_no_members_projects_to_no_members():
    empty = nb.Domain(np.empty(0, dtype=np.int64), ("x", "y"), COORDS, (3, 2))
    assert empty.project(("y",)).size == 0


def test_a_dimension_the_domain_does_not_have_raises():
    with pytest.raises(ValueError, match="not in the domain over"):
        domain([(0, 0)]).project(("z",))


def test_a_repeated_dimension_raises():
    with pytest.raises(ValueError, match="appear more than once"):
        domain([(0, 0)]).project(("x", "x"))


def test_no_dimension_raises():
    with pytest.raises(ValueError, match="no dimension is given"):
        domain([(0, 0)]).project(())
