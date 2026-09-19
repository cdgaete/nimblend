import numpy as np

from nimblend import kernel
from nimblend.sparse import SparseArray


def block(values, labels):
    """An array holding only the nonzero cells of `values`."""
    arr = SparseArray.from_dense(np.asarray(values, dtype=np.float64), labels)
    keep = arr.data != 0.0
    return SparseArray.from_canonical(
        arr.index[:, keep], arr.data[keep], arr.coords, arr.dims
    )


def test_rename_does_not_call_kernel_canonicalize(monkeypatch):
    arr = block(
        [[1.0, 0.0], [3.0, 4.0]], {"x": np.array(["a", "b"]), "y": np.array([0, 1])}
    )

    def boom(*_args, **_kwargs):
        raise AssertionError("canonicalize is called")

    monkeypatch.setattr(kernel, "canonicalize", boom)
    got = arr.rename({"x": "p"})
    assert got.dims == ("p", "y")
    assert got.coords["p"] is arr.coords["x"]
    assert got.index.tolist() == arr.index.tolist()
    assert got.data.tolist() == arr.data.tolist()
