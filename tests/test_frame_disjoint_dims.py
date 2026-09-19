import pytest

from nimblend import frame


def test_disjoint_dims_raises_the_share_no_dimension_message():
    with pytest.raises(
        ValueError,
        match=r"frames \('x',\) and \('y',\) share no dimension",
    ):
        frame.disjoint_dims(("x",), ("y",))


def test_combined_dims_of_disjoint_frames_raises_through_disjoint_dims(monkeypatch):
    calls = []

    def spy(left, right):
        calls.append((left, right))
        raise ValueError("stub")

    monkeypatch.setattr(frame, "disjoint_dims", spy)
    with pytest.raises(ValueError, match="stub"):
        frame.combined_dims(("x",), ("y",))
    assert calls == [(("x",), ("y",))]
