import numpy as np
import pytest

from nimblend import display
from nimblend.buffer import EntryBuffer
from nimblend.coords import ProductCoord, StoredCoord, SubsetCoord
from nimblend.domain import Domain
from nimblend.sparse import SparseArray


def block(values, labels, absence="empty"):
    """An array holding only the nonzero cells of `values`."""
    arr = SparseArray.from_dense(np.asarray(values, dtype=np.float64), labels)
    keep = arr.data != 0.0
    return SparseArray.from_canonical(
        arr.index[:, keep], arr.data[keep], arr.coords, arr.dims, absence
    )


def xy():
    return block(
        [[120.0, 80.0, 0.0], [90.0, 0.0, 30.0]],
        {
            "region": np.array(["DE", "FR"]),
            "tech": np.array(["wind", "solar", "gas"]),
        },
    )


def wide(n):
    """A one-dimensional array carrying `n` entries."""
    labels = np.array([f"L{k}" for k in range(n)])
    return SparseArray.from_canonical(
        np.arange(n, dtype=np.int32)[None, :],
        np.arange(n, dtype=np.float64),
        {"x": StoredCoord(labels)},
        ("x",),
    )


def test_repr_states_the_frame_and_what_absence_means():
    got = repr(xy())
    assert "SparseArray" in got
    assert "('region', 'tech')" in got
    assert "shape=(2, 3)" in got
    assert "nnz=4" in got
    assert "absence='empty'" in got


def test_repr_is_one_line():
    assert "\n" not in repr(xy())


def test_html_carries_the_labels_not_the_positions():
    got = xy()._repr_html_()
    assert "DE" in got and "FR" in got
    assert "wind" in got and "gas" in got
    assert "120.0" in got


def header_of(rendered):
    """The header row's text, without the styling attributes."""
    return rendered.split("<thead>")[1].split("</thead>")[0]


def test_html_names_a_column_per_dimension_and_one_for_the_value():
    got = header_of(xy()._repr_html_())
    assert "region" in got
    assert "tech" in got
    assert "value" in got


def test_html_shows_only_a_head_of_a_large_array():
    got = wide(50)._repr_html_()
    # one header row plus the head
    assert got.count("<tr>") == 1 + display.HEAD
    assert "showing 10 of 50 entries" in got
    assert "L0" in got
    assert "L49" not in got


def test_html_of_a_small_array_says_it_shows_everything():
    assert "showing 4 of 4 entries" in xy()._repr_html_()


def test_only_the_head_of_the_labels_is_resolved():
    # resolving every label of a large array to display ten would defeat
    # the point of the head
    arr = wide(500)
    asked = []
    resolve = arr.coords["x"].to_index

    def spy(positions):
        asked.append(len(positions))
        return resolve(positions)

    arr.coords["x"].to_index = spy
    arr._repr_html_()
    assert asked
    assert max(asked) <= display.HEAD


def test_a_label_that_looks_like_markup_is_escaped():
    arr = block(
        [[1.0]], {"x": np.array(["<script>alert(1)</script>"]), "y": np.array(["&"])}
    )
    got = arr._repr_html_()
    assert "<script>" not in got
    assert "&lt;script&gt;" in got
    assert "&amp;" in got


def test_an_array_carrying_no_entry_says_so():
    got = block([[0.0]], {"x": np.array(["a"]), "y": np.array([1])})._repr_html_()
    assert "no entries" in got
    assert "<tbody>" not in got


def test_a_generated_coordinate_renders_the_tuple_it_stands_for():
    # a grouped array numbers its new dimension with a SubsetCoord, whose
    # member stands for a coordinate rather than a single label
    values = np.zeros((2, 2, 2))
    values[0, 0, 1] = 5.0
    values[1, 1, 0] = 7.0
    arr = block(
        values,
        {"i": np.array(["p", "q"]), "j": np.array([10, 20]), "c": np.arange(2)},
    )
    got = arr.group(("i", "j"), "g")._repr_html_()
    assert "(0, 0)" in got
    assert "(1, 1)" in got


def test_html_of_an_unknown_array_states_its_absence():
    assert "absence='unknown'" in xy().as_unknown()._repr_html_()


def test_a_domain_states_its_frame_and_size():
    got = repr(xy().domain(("region",)))
    assert "Domain" in got
    assert "('region',)" in got
    assert "shape=(2,)" in got
    assert "size=2" in got


def test_a_domain_renders_its_members_with_labels():
    got = xy().domain(("region", "tech"))._repr_html_()
    assert "region" in header_of(got)
    assert "DE" in got and "wind" in got
    # a domain names coordinates; it carries no value
    assert "value" not in header_of(got)


def test_an_empty_domain_says_it_carries_no_member():
    empty = Domain(
        np.array([], dtype=np.int64), ("x",), {"x": StoredCoord([1, 2])}, (2,)
    )
    assert "no entries" in empty._repr_html_()


def test_each_coordinate_kind_describes_itself():
    assert "3 labels" in repr(StoredCoord(np.array(["a", "b", "c"])))
    assert "ProductCoord" in repr(ProductCoord((2, 3)))
    assert "(2, 3)" in repr(ProductCoord((2, 3)))
    assert "SubsetCoord" in repr(SubsetCoord(np.array([0, 2]), (2, 3)))


def test_a_coordinate_repr_reports_where_it_is_numbered_from():
    assert "start=10" in repr(SubsetCoord(np.array([0, 2]), (2, 3), start=10))
    assert "start=10" in repr(ProductCoord((2, 3), start=10))


def test_a_buffer_reports_how_much_of_it_is_reserved():
    buffer = EntryBuffer(2, 100)
    assert "capacity=100" in repr(buffer)
    buffer.reserve(7)
    assert "at=7" in repr(buffer)


@pytest.mark.parametrize("absence", ["empty", "unknown"])
def test_every_repr_survives_a_round_trip_through_str(absence):
    # str falls back to repr; defining both would duplicate
    arr = block([[1.0, 2.0]], {"x": np.array(["a"]), "y": np.array([1, 2])}, absence)
    assert str(arr) == repr(arr)
