import numpy as np

from nimblend import protocol


def test_a_class_missing_a_member_is_not_an_array():
    class Partial:
        dims = ()
        shape = ()

    assert not isinstance(Partial(), protocol.Array)


def test_the_protocol_names_every_member_the_contract_requires():
    # equality, not containment: a member an implementation offers and the
    # contract omits is the gap that lets two implementations diverge
    required = {
        "dims",
        "shape",
        "coords",
        "absence",
        "nnz",
        "sel",
        "sum",
        "mean",
        "min",
        "max",
        "shift",
        "roll",
        "rename",
        "transpose",
        "conform",
        "to_dense",
        "domain",
        "coordinates",
        "values",
        "restrict",
        "expand",
        "group",
        "as_empty",
        "as_unknown",
        "__add__",
        "__radd__",
        "__sub__",
        "__rsub__",
        "__mul__",
        "__rmul__",
        "__truediv__",
        "__rtruediv__",
        "__neg__",
        "__pow__",
    }
    assert required == set(protocol.Array.__protocol_attrs__)


def test_absence_values_are_the_two_the_contract_defines():
    assert set(protocol.ABSENCE) == {"empty", "unknown"}


def test_conformance_helper_reports_a_missing_member():
    from conformance import missing_members

    class Half:
        dims = ()

    assert "sum" in missing_members(Half)
    assert np is not None


def test_both_implementations_carry_every_member_the_contract_names():
    import nimblend as nb

    labels = {"x": np.array(["a", "b"]), "y": np.array([1, 2, 3])}
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    coords = {name: nb.StoredCoord(label) for name, label in labels.items()}
    built = (
        nb.SparseArray.from_dense(values, labels),
        nb.DenseArray(values, coords, ("x", "y")),
    )
    for array in built:
        lacking = sorted(
            name
            for name in protocol.Array.__protocol_attrs__
            if not hasattr(array, name)
        )
        assert lacking == [], (type(array).__name__, lacking)


def test_reading_out_as_csr_is_the_sparse_implementations_own_and_not_the_contract():
    import nimblend as nb

    assert "to_csr" not in protocol.Array.__protocol_attrs__
    assert hasattr(nb.SparseArray, "to_csr")
