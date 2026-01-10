"""
Comprehensive benchmark: nimblend vs xarray with fair join comparisons.

- Aligned arrays: xarray inner join (default) vs nimblend
- Misaligned arrays: xarray outer join vs nimblend outer join
- Large arrays: test parallel Rust implementation (>1M elements)

Results are saved to benchmarks/results/ for tracking over time.
"""

import gc
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable, List

import numpy as np


@dataclass
class BenchResult:
    """Result of a benchmark run."""
    name: str
    size: str
    elements: int
    nimblend_ms: float
    xarray_ms: float
    speedup: float
    category: str

    def __str__(self):
        return (
            f"{self.name:45} | "
            f"nb: {self.nimblend_ms:8.2f}ms | "
            f"xa: {self.xarray_ms:8.2f}ms | "
            f"speedup: {self.speedup:5.1f}x"
        )


def measure(func: Callable, warmup: int = 2, runs: int = 5) -> float:
    """Measure median time of a function."""
    for _ in range(warmup):
        func()
        gc.collect()

    times = []
    for _ in range(runs):
        gc.collect()
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)

    return np.median(times)


def run_benchmark(
    name: str,
    size: str,
    elements: int,
    nimblend_fn: Callable,
    xarray_fn: Callable,
    category: str,
) -> BenchResult:
    """Run a single benchmark."""
    nb_time = measure(nimblend_fn)
    xa_time = measure(xarray_fn)
    speedup = xa_time / nb_time if nb_time > 0 else float('inf')
    return BenchResult(
        name=name,
        size=size,
        elements=elements,
        nimblend_ms=nb_time * 1000,
        xarray_ms=xa_time * 1000,
        speedup=speedup,
        category=category,
    )


# ============================================================================
# BENCHMARK CASES
# ============================================================================

def benchmark_aligned(sizes: List[int]) -> List[BenchResult]:
    """
    Benchmark aligned arrays.
    
    xarray uses inner join by default, which is equivalent to outer
    when arrays are aligned. This is the fair comparison.
    """
    import nimblend
    import xarray as xr

    results = []
    for n in sizes:
        data1 = np.random.randn(n, n)
        data2 = np.random.randn(n, n)
        x_coords = [f"x{i}" for i in range(n)]
        y_coords = [f"y{i}" for i in range(n)]

        nb1 = nimblend.Array(data1, {"x": x_coords, "y": y_coords})
        nb2 = nimblend.Array(data2, {"x": x_coords, "y": y_coords})

        xa1 = xr.DataArray(data1, coords={"x": x_coords, "y": y_coords}, dims=["x", "y"])
        xa2 = xr.DataArray(data2, coords={"x": x_coords, "y": y_coords}, dims=["x", "y"])

        results.append(run_benchmark(
            name=f"aligned add ({n}x{n})",
            size=f"{n}x{n}",
            elements=n * n,
            nimblend_fn=lambda: nb1 + nb2,
            xarray_fn=lambda: xa1 + xa2,  # inner join (default)
            category="aligned",
        ))

    return results


def benchmark_misaligned_outer(sizes: List[int]) -> List[BenchResult]:
    """
    Benchmark misaligned arrays with outer join.
    
    Both nimblend and xarray use outer join semantics.
    50% coordinate overlap on x dimension.
    """
    import nimblend
    import xarray as xr

    results = []
    for n in sizes:
        data1 = np.random.randn(n, n)
        data2 = np.random.randn(n, n)
        x1 = [f"x{i}" for i in range(n)]
        x2 = [f"x{i}" for i in range(n // 2, n + n // 2)]  # 50% overlap
        y_coords = [f"y{i}" for i in range(n)]

        nb1 = nimblend.Array(data1, {"x": x1, "y": y_coords})
        nb2 = nimblend.Array(data2, {"x": x2, "y": y_coords})

        xa1 = xr.DataArray(data1, coords={"x": x1, "y": y_coords}, dims=["x", "y"])
        xa2 = xr.DataArray(data2, coords={"x": x2, "y": y_coords}, dims=["x", "y"])

        result_elements = int(n * 1.5) * n  # outer join expands x by 50%

        def xa_outer():
            with xr.set_options(arithmetic_join='outer'):
                return xa1 + xa2

        results.append(run_benchmark(
            name=f"misaligned outer ({n}x{n} -> {n}x{int(n*1.5)})",
            size=f"{n}x{n}",
            elements=result_elements,
            nimblend_fn=lambda: nb1 + nb2,
            xarray_fn=xa_outer,
            category="misaligned_outer",
        ))

    return results


def benchmark_large_parallel(sizes: List[int]) -> List[BenchResult]:
    """
    Benchmark large arrays to test parallel Rust implementation.
    
    Parallel threshold is 1M elements, so we test arrays above this.
    Tests both aligned and misaligned cases.
    """
    import nimblend
    import xarray as xr

    results = []
    for n in sizes:
        # Aligned large
        data1 = np.random.randn(n, n)
        data2 = np.random.randn(n, n)
        x_coords = [f"x{i}" for i in range(n)]
        y_coords = [f"y{i}" for i in range(n)]

        nb1 = nimblend.Array(data1, {"x": x_coords, "y": y_coords})
        nb2 = nimblend.Array(data2, {"x": x_coords, "y": y_coords})

        xa1 = xr.DataArray(data1, coords={"x": x_coords, "y": y_coords}, dims=["x", "y"])
        xa2 = xr.DataArray(data2, coords={"x": x_coords, "y": y_coords}, dims=["x", "y"])

        results.append(run_benchmark(
            name=f"LARGE aligned ({n}x{n}, {n*n/1e6:.1f}M)",
            size=f"{n}x{n}",
            elements=n * n,
            nimblend_fn=lambda: nb1 + nb2,
            xarray_fn=lambda: xa1 + xa2,
            category="large_aligned",
        ))

        # Misaligned large
        x1 = [f"x{i}" for i in range(n)]
        x2 = [f"x{i}" for i in range(n // 2, n + n // 2)]

        nb1m = nimblend.Array(data1, {"x": x1, "y": y_coords})
        nb2m = nimblend.Array(data2, {"x": x2, "y": y_coords})

        xa1m = xr.DataArray(data1, coords={"x": x1, "y": y_coords}, dims=["x", "y"])
        xa2m = xr.DataArray(data2, coords={"x": x2, "y": y_coords}, dims=["x", "y"])

        result_elements = int(n * 1.5) * n

        def xa_outer_large():
            with xr.set_options(arithmetic_join='outer'):
                return xa1m + xa2m

        results.append(run_benchmark(
            name=f"LARGE misaligned ({n}x{n}, {result_elements/1e6:.1f}M)",
            size=f"{n}x{n}",
            elements=result_elements,
            nimblend_fn=lambda: nb1m + nb2m,
            xarray_fn=xa_outer_large,
            category="large_misaligned",
        ))

    return results


def benchmark_reductions(sizes: List[int]) -> List[BenchResult]:
    """Benchmark reduction operations."""
    import nimblend
    import xarray as xr

    results = []
    for n in sizes:
        data = np.random.randn(n, n)
        x_coords = [f"x{i}" for i in range(n)]
        y_coords = [f"y{i}" for i in range(n)]

        nb_arr = nimblend.Array(data, {"x": x_coords, "y": y_coords})
        xa_arr = xr.DataArray(data, coords={"x": x_coords, "y": y_coords}, dims=["x", "y"])

        results.append(run_benchmark(
            name=f"sum all ({n}x{n})",
            size=f"{n}x{n}",
            elements=n * n,
            nimblend_fn=lambda: nb_arr.sum(),
            xarray_fn=lambda: xa_arr.sum(),
            category="reduction",
        ))

        results.append(run_benchmark(
            name=f"mean(dim='x') ({n}x{n})",
            size=f"{n}x{n}",
            elements=n * n,
            nimblend_fn=lambda: nb_arr.mean("x"),
            xarray_fn=lambda: xa_arr.mean("x"),
            category="reduction",
        ))

    return results


def benchmark_selection(sizes: List[int]) -> List[BenchResult]:
    """Benchmark selection operations."""
    import nimblend
    import xarray as xr

    results = []
    for n in sizes:
        data = np.random.randn(n, n)
        x_coords = [f"x{i}" for i in range(n)]
        y_coords = [f"y{i}" for i in range(n)]

        nb_arr = nimblend.Array(data, {"x": x_coords, "y": y_coords})
        xa_arr = xr.DataArray(data, coords={"x": x_coords, "y": y_coords}, dims=["x", "y"])

        target = f"x{n // 2}"
        results.append(run_benchmark(
            name=f"sel single ({n}x{n})",
            size=f"{n}x{n}",
            elements=n * n,
            nimblend_fn=lambda: nb_arr.sel({"x": target}),
            xarray_fn=lambda: xa_arr.sel(x=target),
            category="selection",
        ))

        results.append(run_benchmark(
            name=f"isel ({n}x{n})",
            size=f"{n}x{n}",
            elements=n * n,
            nimblend_fn=lambda: nb_arr.isel({"x": n // 2}),
            xarray_fn=lambda: xa_arr.isel(x=n // 2),
            category="selection",
        ))

    return results


# ============================================================================
# MAIN
# ============================================================================

def save_results(results: List[BenchResult], filename: str):
    """Save benchmark results to JSON."""
    os.makedirs("benchmarks/results", exist_ok=True)
    filepath = f"benchmarks/results/{filename}"
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "results": [asdict(r) for r in results],
        "summary": {
            "total_benchmarks": len(results),
            "nimblend_wins": sum(1 for r in results if r.speedup > 1),
            "xarray_wins": sum(1 for r in results if r.speedup < 1),
            "median_speedup": float(np.median([r.speedup for r in results])),
            "mean_speedup": float(np.mean([r.speedup for r in results])),
            "min_speedup": float(np.min([r.speedup for r in results])),
            "max_speedup": float(np.max([r.speedup for r in results])),
        }
    }
    
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to {filepath}")


def main():
    """Run all benchmarks."""
    print("=" * 80)
    print("NIMBLEND vs XARRAY COMPREHENSIVE BENCHMARK")
    print("=" * 80)
    print("\nJoin semantics:")
    print("  - Aligned arrays: xarray inner join (default) vs nimblend")
    print("  - Misaligned arrays: xarray outer join vs nimblend outer join")
    print("  - Large arrays: test parallel Rust (threshold >1M elements)")
    print()

    all_results = []

    # Standard sizes
    standard_sizes = [100, 500, 1000]
    
    # Large sizes for parallel testing (>1M elements)
    # 1000x1000 = 1M, 1500x1500 = 2.25M, 2000x2000 = 4M, 3000x3000 = 9M
    large_sizes = [1500, 2000, 3000]

    print("--- Aligned Arrays (xarray inner join) ---")
    for r in benchmark_aligned(standard_sizes):
        print(r)
        all_results.append(r)

    print("\n--- Misaligned Arrays (both outer join, 50% overlap) ---")
    for r in benchmark_misaligned_outer(standard_sizes):
        print(r)
        all_results.append(r)

    print("\n--- Large Arrays (parallel Rust, >1M elements) ---")
    for r in benchmark_large_parallel(large_sizes):
        print(r)
        all_results.append(r)

    print("\n--- Reductions ---")
    for r in benchmark_reductions(standard_sizes):
        print(r)
        all_results.append(r)

    print("\n--- Selection ---")
    for r in benchmark_selection(standard_sizes):
        print(r)
        all_results.append(r)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    speedups = [r.speedup for r in all_results]
    print(f"Total benchmarks: {len(all_results)}")
    print(f"Median speedup:   {np.median(speedups):.1f}x")
    print(f"Mean speedup:     {np.mean(speedups):.1f}x")
    print(f"Min speedup:      {np.min(speedups):.1f}x")
    print(f"Max speedup:      {np.max(speedups):.1f}x")

    nb_wins = sum(1 for r in all_results if r.speedup > 1)
    xa_wins = sum(1 for r in all_results if r.speedup < 1)
    print(f"\nNimblend faster: {nb_wins}/{len(all_results)} benchmarks")
    print(f"Xarray faster:   {xa_wins}/{len(all_results)} benchmarks")

    # Category breakdown
    print("\n--- By Category ---")
    categories = {}
    for r in all_results:
        if r.category not in categories:
            categories[r.category] = []
        categories[r.category].append(r.speedup)
    
    for cat, speeds in sorted(categories.items()):
        print(f"{cat:20}: median {np.median(speeds):.1f}x, range {np.min(speeds):.1f}x - {np.max(speeds):.1f}x")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(all_results, f"bench_{timestamp}.json")
    save_results(all_results, "bench_latest.json")


if __name__ == "__main__":
    main()
