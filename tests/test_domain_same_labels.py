import numpy as np
import pytest

from nimblend import domain as domain_module
from nimblend.coords import StoredCoord, same_labels
from nimblend.domain import Domain


def test_domain_intersect_calls_same_labels_with_the_domain_message(monkeypatch):
    calls = []

    def spy(dims, left, right, **kwargs):
        calls.append((dims, left, right, kwargs))
        raise ValueError("stub")

    monkeypatch.setattr(domain_module, "same_labels", spy)
    left = Domain.full(("x",), {"x": StoredCoord(np.array(["a", "b"]))})
    right = Domain.full(("x",), {"x": StoredCoord(np.array(["a", "b"]))})
    with pytest.raises(ValueError, match="stub"):
        left.intersect(right)
    assert calls == [
        (
            ("x",),
            left.coords,
            right.coords,
            {"operands": "domains", "action": "combine domains over the same labels"},
        )
    ]


def test_same_labels_writes_the_domain_message_from_its_parameters():
    with pytest.raises(
        ValueError,
        match="different labels in the two domains; combine domains over the "
        "same labels",
    ):
        same_labels(
            ("x",),
            {"x": StoredCoord(np.array(["a"]))},
            {"x": StoredCoord(np.array(["b"]))},
            operands="domains",
            action="combine domains over the same labels",
        )
