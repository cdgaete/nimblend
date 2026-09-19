import pytest

import nimblend as nb


@pytest.mark.parametrize("cls", [nb.SparseArray, nb.DenseArray])
def test_rmul_has_the_return_annotation_of_mul(cls):
    mul = cls.__mul__.__annotations__["return"]
    assert cls.__rmul__.__annotations__["return"] == mul
