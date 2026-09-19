import pytest

from nimblend import kernel


def test_span_is_the_number_of_cells_of_a_shape():
    assert kernel.span((2, 3, 4)) == 24
    assert kernel.span(()) == 1
    assert kernel.span((5, 0)) == 0


def test_span_raises_beyond_the_int64_range():
    with pytest.raises(OverflowError, match="exceeds the int64 range"):
        kernel.span((2**32, 2**32))


def test_span_of_a_shape_with_an_extent_of_zero_is_zero_in_every_position():
    assert kernel.span((0, 2**32, 2**32)) == 0
    assert kernel.span((2**32, 0, 2**32)) == 0
    assert kernel.span((2**32, 2**32, 0)) == 0
