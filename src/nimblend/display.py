"""How an array and its parts render in a terminal and in a notebook.

A rendered array shows a head of the entries it carries, never the whole of
a large one, and resolves labels for that head alone.
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
    """`name(dims, shape=..., ...)`, the form a terminal shows."""
    fields = [repr(tuple(dims)), f"shape={tuple(shape)}"]
    fields.extend(f"{key}={value}" for key, value in extra.items())
    return f"{name}({', '.join(fields)})"


def as_text(answered: npt.ArrayLike) -> list[str]:
    """One string per entry, whatever shape a coordinate answered with.

    A stored coordinate answers one label per entry; a generated one answers
    the index matrix its members stand for, so a member reads as the
    coordinate tuple rather than a single label.
    """
    answered = np.asarray(answered)
    if answered.ndim == 1:
        return [str(value) for value in answered]
    return [
        "(" + ", ".join(str(int(value)) for value in answered[:, at]) + ")"
        for at in range(answered.shape[1])
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
    """A head of `entries` as an HTML table under the one-line summary.

    `entries` carries at most the head; `total` is how many exist, so a
    caller resolves labels for the head alone and states the rest here.
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
