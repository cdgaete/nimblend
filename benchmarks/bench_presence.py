"""What a dense array's presence costs, carried as a mask or as a NaN tag.

An `"unknown"` array's absence propagates through every operator, which a NaN
does in the hardware and a mask does by a second pass. An `"empty"` array's
absence is the additive identity, which a mask substitutes in one `where` and
a NaN needs stripping and restoring. The two declarations therefore want
opposite encodings, which is why `DenseArray` stores what it declares.
"""

import timeit

import numpy as np


def measure(size=2000, density=0.9, repeat=5):
    """Milliseconds for each operation under both encodings."""
    rng = np.random.default_rng(0)
    a, b = rng.random((size, size)), rng.random((size, size))
    ma = rng.random((size, size)) < density
    mb = rng.random((size, size)) < density
    an, bn = a.copy(), b.copy()
    an[~ma], bn[~mb] = np.nan, np.nan

    def best(fn):
        return min(timeit.repeat(fn, number=1, repeat=repeat)) * 1e3

    def nan_identity():
        out = np.nan_to_num(an, nan=0.0) + np.nan_to_num(bn, nan=0.0)
        out[np.isnan(an) & np.isnan(bn)] = np.nan
        return out

    return {
        "propagate_mask_ms": best(lambda: (np.where(ma & mb, a + b, 0.0), ma & mb)),
        "propagate_nan_ms": best(lambda: an + bn),
        "identity_mask_ms": best(
            lambda: (np.where(ma, a, 0.0) + np.where(mb, b, 0.0), ma | mb)
        ),
        "identity_nan_ms": best(nan_identity),
        "reduce_mask_ms": best(lambda: np.where(ma, a, 0.0).sum(axis=0)),
        "reduce_nan_ms": best(lambda: np.nansum(an, axis=0)),
        "mask_mb": size * size / 1e6,
        "values_mb": size * size * 8 / 1e6,
    }


if __name__ == "__main__":
    for density in (0.9, 0.5):
        got = measure(density=density)
        print(f"\ndensity {density}, 2000x2000 float64")
        print(
            f"  a+b propagating   mask {got['propagate_mask_ms']:6.2f} ms   "
            f"nan {got['propagate_nan_ms']:6.2f} ms"
        )
        print(
            f"  a+b as identity   mask {got['identity_mask_ms']:6.2f} ms   "
            f"nan {got['identity_nan_ms']:6.2f} ms"
        )
        print(
            f"  sum over present  mask {got['reduce_mask_ms']:6.2f} ms   "
            f"nan {got['reduce_nan_ms']:6.2f} ms"
        )
        print(
            f"  storage           mask {got['mask_mb']:6.1f} MB beside "
            f"{got['values_mb']:.1f} MB of values   nan none"
        )
