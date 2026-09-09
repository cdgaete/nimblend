import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))


def test_assembly_holds_no_second_copy_of_the_result():
    from bench_assembly import measure

    # the excess over the final matrix is the per-block sort-merge working
    # set: one block's worth at most, however many blocks are assembled.
    # A second copy of any block would push the excess past block_mb.
    three = measure(n_rows=500, n_cols=100, n_blocks=3)
    twelve = measure(n_rows=500, n_cols=100, n_blocks=12)
    for got in (three, twelve):
        assert got["excess_mb"] <= got["block_mb"], got
    assert twelve["excess_mb"] < three["excess_mb"] * 1.5, (three, twelve)


def test_sparse_addition_is_faster_than_dense_below_the_crossover():
    from bench_crossover import measure

    # on a 3000x3000 grid the two cost the same at about 0.9% density; at
    # 0.5% the sparse addition is the faster one, which is the claim that
    # justifies carrying a sparse implementation at all
    got = measure(size=3000, density=0.005)
    assert got["sparse_ms"] < got["dense_ms"], got
    assert got["sparse_mb"] < got["dense_mb"] / 10, got


def test_each_declaration_prefers_the_encoding_dense_array_stores():
    from bench_presence import measure

    # a propagating absence is what a NaN does for free and a mask pays a
    # pass for; an absence used as the additive identity is the reverse.
    # DenseArray stores each declaration the way this table says to.
    got = measure(size=800, density=0.9, repeat=3)
    assert got["propagate_nan_ms"] < got["propagate_mask_ms"], got
    assert got["identity_mask_ms"] < got["identity_nan_ms"], got
