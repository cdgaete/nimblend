"""Where a sparse representation overtakes a dense one.

Addition of two arrays over one coordinate grid, sweeping how much of the
grid carries a value.
"""

import timeit

import numpy as np

import nimblend as nb


def measure(size, density):
    """Time and bytes for a dense and a sparse addition at `density`."""
    rng = np.random.default_rng(0)
    nnz = max(int(size * size * density), 1)
    coords = {
        "x": nb.StoredCoord(np.arange(size)),
        "y": nb.StoredCoord(np.arange(size)),
    }

    dense_a = np.zeros((size, size))
    dense_b = np.zeros((size, size))
    for target in (dense_a, dense_b):
        rows = rng.integers(0, size, nnz)
        cols = rng.integers(0, size, nnz)
        target[rows, cols] = rng.random(nnz)

    dense_ms = min(timeit.repeat(lambda: dense_a + dense_b, number=3, repeat=3)) / 3

    def to_sparse(dense):
        rows, cols = np.nonzero(dense)
        return nb.from_long(
            ("x", "y"),
            coords,
            {"x": rows, "y": cols},
            dense[rows, cols],
        )

    sparse_a, sparse_b = to_sparse(dense_a), to_sparse(dense_b)
    sparse_ms = min(timeit.repeat(lambda: sparse_a + sparse_b, number=3, repeat=3)) / 3

    sparse_bytes = sparse_a.index.nbytes + sparse_a.data.nbytes
    return {
        "dense_ms": dense_ms * 1e3,
        "sparse_ms": sparse_ms * 1e3,
        "dense_mb": dense_a.nbytes / 1e6,
        "sparse_mb": sparse_bytes / 1e6,
    }


if __name__ == "__main__":
    print(
        f"{'density':>8} {'dense ms':>10} {'sparse ms':>10} "
        f"{'dense MB':>9} {'sparse MB':>10}"
    )
    for density in (0.5, 0.1, 0.01, 0.001):
        got = measure(3000, density)
        print(
            f"{density:8.1%} {got['dense_ms']:10.2f} {got['sparse_ms']:10.2f} "
            f"{got['dense_mb']:9.1f} {got['sparse_mb']:10.1f}"
        )
