"""Text and HTML representations of arrays and domains.

A representation shows at most the first `HEAD` entries and resolves labels
for those entries only.
"""

import html
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

HEAD = 10

_STYLE = (
    "<style>"
    ".nimblend{border-collapse:collapse;margin-top:0.4em}"
    ".nimblend td,.nimblend th{padding:0.15em 0.6em;text-align:left;"
    "border:1px solid currentColor}"
    "</style>"
)


def one_line(
    name: str, dims: Sequence[str], shape: Sequence[int], **extra: object
) -> str:
    """Return the one-line form `name(dims, shape=..., ...)`."""
    fields = [repr(tuple(dims)), f"shape={tuple(shape)}"]
    fields.extend(f"{key}={value}" for key, value in extra.items())
    return f"{name}({', '.join(fields)})"


def as_text(labels: npt.ArrayLike) -> list[str]:
    """Return one string per entry from the result of a coordinate's `to_index`.

    A one-dimensional input has one label per entry. A two-dimensional input
    is an index matrix, and each column is written as a tuple of integers.
    """
    labels = np.asarray(labels)
    if labels.ndim == 1:
        return [str(value) for value in labels]
    return [
        "(" + ", ".join(str(int(value)) for value in labels[:, at]) + ")"
        for at in range(labels.shape[1])
    ]


def table(
    name: str,
    dims: Sequence[str],
    shape: Sequence[int],
    headers: Sequence[str],
    entries: Sequence[Iterable[Any]],
    total: int,
    **extra: object,
) -> str:
    """Return an HTML table of `entries` below the one-line summary.

    `entries` contains at most the first `HEAD` entries. `total` is the
    number of entries in the array, reported below the table.
    """
    summary = html.escape(one_line(name, dims, shape, **extra), quote=False)
    if total == 0:
        return f"<div><code>{summary}</code><p><em>no entries</em></p></div>"
    header = "".join(f"<th>{html.escape(str(c), quote=False)}</th>" for c in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(cell), quote=False)}</td>" for cell in entry)
        + "</tr>"
        for entry in entries
    )
    plural = "entry" if total == 1 else "entries"
    return (
        f"<div>{_STYLE}<code>{summary}</code>"
        f'<table class="nimblend"><thead><tr>{header}</tr></thead>'
        f"<tbody>{body}</tbody></table>"
        f"<small>showing {len(entries)} of {total} {plural}</small></div>"
    )
