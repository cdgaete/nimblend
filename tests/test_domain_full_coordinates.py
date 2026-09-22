import numpy as np

import nimblend as nb
from nimblend import kernel


def coords():
    return {"a": nb.ProductCoord((3,)), "b": nb.ProductCoord((4,))}


def refuse_unravel(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the codes were unravelled")

    monkeypatch.setattr(kernel, "unravel", refuse)


def test_a_full_domain_writes_its_coordinates_without_unravelling(monkeypatch):
    domain = nb.Domain.full(("a", "b"), coords())
    refuse_unravel(monkeypatch)
    got = domain.coordinates()
    assert got.tolist() == [[0] * 4 + [1] * 4 + [2] * 4, [0, 1, 2, 3] * 3]


def test_a_partial_domain_unravels_its_codes():
    domain = nb.Domain(
        np.array([1, 6, 11], dtype=np.int64), ("a", "b"), coords(), (3, 4)
    )
    assert domain.coordinates().tolist() == [[0, 1, 2], [1, 2, 3]]


def test_a_full_domain_returns_the_coordinates_at_positions():
    domain = nb.Domain.full(("a", "b"), coords())
    assert domain.coordinates([5, 0]).tolist() == [[1, 0], [1, 0]]


def test_an_empty_domain_beyond_the_int64_range_has_no_coordinates():
    # five dimensions of 2**30 members: 2**150 cells
    dims = ("a", "b", "c", "d", "e")
    wide = {d: nb.ProductCoord((2**30,)) for d in dims}
    domain = nb.Domain(np.empty(0, dtype=np.int64), dims, wide, (2**30,) * 5)
    assert domain.coordinates().shape == (5, 0)
