"""Peak memory of assembling constraint blocks into one buffer.

Each block is reduced directly into its destination slice, so the assembled
matrix exists once. The working set that remains is the sort-merge, which is
per-block and transient, so the ratio falls as blocks accumulate.
"""

import tracemalloc

import numpy as np

from nimblend import kernel


def _source(n_rows, n_cols):
    per = n_rows * n_cols
    idx = np.empty((3, per), dtype=np.int32)
    idx[0] = np.repeat(np.arange(n_rows, dtype=np.int32), n_cols)
    idx[1] = np.tile(np.arange(n_cols, dtype=np.int32), n_rows)
    idx[2] = np.arange(per, dtype=np.int32)
    return idx, np.ones(per)


def measure(n_rows, n_cols, n_blocks):
    """Peak and final bytes of assembling `n_blocks` blocks through the kernel."""
    per_block = n_rows * n_cols
    total = n_blocks * per_block
    src_idx, src_data = _source(n_rows, n_cols)

    tracemalloc.start()
    index = np.empty((2, total), dtype=np.int32)
    data = np.empty(total, dtype=np.float64)
    at = 0
    for _ in range(n_blocks):
        kernel.reduce_axis(
            src_idx,
            src_data,
            1,
            (n_rows, n_cols, per_block),
            out=(index[:, at : at + per_block], data[at : at + per_block]),
        )
        at += per_block
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    final = index.nbytes + data.nbytes
    block = per_block * (2 * 4 + 8)
    return {
        "final_mb": final / 1e6,
        "peak_mb": peak / 1e6,
        "ratio": peak / final,
        "excess_mb": (peak - final) / 1e6,
        "block_mb": block / 1e6,
    }


if __name__ == "__main__":
    for blocks in (1, 3, 8, 16):
        got = measure(2000, 500, blocks)
        print(
            f"{blocks:2d} blocks  final {got['final_mb']:7.2f} MB  "
            f"peak {got['peak_mb']:7.2f} MB  {got['ratio']:.2f}x  "
            f"excess {got['excess_mb']:6.2f} MB"
        )
