"""Benchmark nimblend vs xarray performance."""

import gc
import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, List

import numpy as np


@dataclass
class BenchResult:
    """Result of a benchmark run."""
    name: str
    nimblend_time: float
    xarray_time: float
    nimblend_mem: float  # KB
    xarray_mem: float    # KB
    speedup: float       # xarray_time / nimblend_time
    
    def __str__(self):
        return (
            f"{self.name:40} | "
            f"nimblend: {self.nimblend_time*1000:8.2f}ms, {self.nimblend_mem:8.1f}KB | "
            f"xarray: {self.xarray_time*1000:8.2f}ms, {self.xarray_mem:8.1f}KB | "
            f"speedup: {self.speedup:5.1f}x"
        )


def measure(func: Callable, warmup: int = 2, runs: int = 5) -> tuple[float, float]:
    """Measure time and memory of a function."""
    # Warmup
    for _ in range(warmup):
        func()
        gc.collect()
    
    # Measure time
    times = []
    for _ in range(runs):
        gc.collect()
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    
    # Measure memory
    gc.collect()
    tracemalloc.start()
    func()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return np.median(times), peak / 1024  # time in sec, memory in KB


def run_benchmark(name: str, nimblend_fn: Callable, xarray_fn: Callable) -> BenchResult:
    """Run a benchmark comparing nimblend vs xarray."""
    nb_time, nb_mem = measure(nimblend_fn)
    xa_time, xa_mem = measure(xarray_fn)
    speedup = xa_time / nb_time if nb_time > 0 else float('inf')
    return BenchResult(name, nb_time, xa_time, nb_mem, xa_mem, speedup)


# ============================================================================
# BENCHMARK CASES
# ============================================================================

def benchmark_creation(sizes: List[int]) -> List[BenchResult]:
    """Benchmark array creation."""
    import nimblend
    import xarray as xr
    
    results = []
    for n in sizes:
        data = np.random.randn(n, n)
        x_coords = [f"x{i}" for i in range(n)]
        y_coords = [f"y{i}" for i in range(n)]
        
        def nb_create():
            return nimblend.Array(data, {"x": x_coords, "y": y_coords})
        
        def xa_create():
            return xr.DataArray(data, coords={"x": x_coords, "y": y_coords}, dims=["x", "y"])
        
        results.append(run_benchmark(f"creation ({n}x{n})", nb_create, xa_create))
    
    return results


def benchmark_arithmetic_aligned(sizes: List[int]) -> List[BenchResult]:
    """Benchmark arithmetic with fully aligned arrays (no outer join needed)."""
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
        
        def nb_add():
            return nb1 + nb2
        
        def xa_add():
            return xa1 + xa2
        
        results.append(run_benchmark(f"add aligned ({n}x{n})", nb_add, xa_add))
    
    return results


def benchmark_arithmetic_misaligned(sizes: List[int]) -> List[BenchResult]:
    """Benchmark arithmetic with partial overlap (outer join)."""
    import nimblend
    import xarray as xr
    
    results = []
    for n in sizes:
        # 50% overlap
        data1 = np.random.randn(n, n)
        data2 = np.random.randn(n, n)
        x1 = [f"x{i}" for i in range(n)]
        x2 = [f"x{i}" for i in range(n//2, n + n//2)]  # 50% overlap
        y_coords = [f"y{i}" for i in range(n)]
        
        nb1 = nimblend.Array(data1, {"x": x1, "y": y_coords})
        nb2 = nimblend.Array(data2, {"x": x2, "y": y_coords})
        
        xa1 = xr.DataArray(data1, coords={"x": x1, "y": y_coords}, dims=["x", "y"])
        xa2 = xr.DataArray(data2, coords={"x": x2, "y": y_coords}, dims=["x", "y"])
        
        def nb_add():
            return nb1 + nb2
        
        # Use outer join for fair comparison (nimblend always uses outer join)
        def xa_add():
            with xr.set_options(arithmetic_join='outer'):
                return xa1 + xa2
        
        results.append(run_benchmark(f"add misaligned ({n}x{n}, 50% overlap)", nb_add, xa_add))
    
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
        
        # Sum all
        results.append(run_benchmark(
            f"sum all ({n}x{n})",
            lambda: nb_arr.sum(),
            lambda: xa_arr.sum()
        ))
        
        # Sum along one dim
        results.append(run_benchmark(
            f"sum(dim='x') ({n}x{n})",
            lambda: nb_arr.sum("x"),
            lambda: xa_arr.sum("x")
        ))
        
        # Mean
        results.append(run_benchmark(
            f"mean all ({n}x{n})",
            lambda: nb_arr.mean(),
            lambda: xa_arr.mean()
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
        
        # Single selection
        target = f"x{n//2}"
        results.append(run_benchmark(
            f"sel single ({n}x{n})",
            lambda: nb_arr.sel({"x": target}),
            lambda: xa_arr.sel(x=target)
        ))
        
        # Multi selection
        targets = [f"x{i}" for i in range(n//4, 3*n//4)]
        results.append(run_benchmark(
            f"sel multi ({n}x{n}, {len(targets)} items)",
            lambda: nb_arr.sel({"x": targets}),
            lambda: xa_arr.sel(x=targets)
        ))
        
        # Integer selection
        results.append(run_benchmark(
            f"isel ({n}x{n})",
            lambda: nb_arr.isel({"x": n//2}),
            lambda: xa_arr.isel(x=n//2)
        ))
    
    return results



# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all benchmarks."""
    print("=" * 100)
    print("NIMBLEND vs XARRAY BENCHMARK")
    print("=" * 100)
    
    sizes = [100, 500, 1000]
    
    all_results = []
    
    print("\n--- Array Creation ---")
    for r in benchmark_creation(sizes):
        print(r)
        all_results.append(r)
    
    print("\n--- Arithmetic (aligned) ---")
    for r in benchmark_arithmetic_aligned(sizes):
        print(r)
        all_results.append(r)
    
    print("\n--- Arithmetic (misaligned, outer join) ---")
    for r in benchmark_arithmetic_misaligned(sizes):
        print(r)
        all_results.append(r)
    
    print("\n--- Reductions ---")
    for r in benchmark_reductions(sizes):
        print(r)
        all_results.append(r)
    
    print("\n--- Selection ---")
    for r in benchmark_selection(sizes):
        print(r)
        all_results.append(r)
    
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    speedups = [r.speedup for r in all_results]
    print(f"Median speedup: {np.median(speedups):.1f}x")
    print(f"Mean speedup:   {np.mean(speedups):.1f}x")
    print(f"Min speedup:    {np.min(speedups):.1f}x")
    print(f"Max speedup:    {np.max(speedups):.1f}x")
    
    # Count wins
    nb_wins = sum(1 for r in all_results if r.speedup > 1)
    xa_wins = sum(1 for r in all_results if r.speedup < 1)
    print(f"\nNimblend faster in {nb_wins}/{len(all_results)} benchmarks")
    print(f"Xarray faster in {xa_wins}/{len(all_results)} benchmarks")


if __name__ == "__main__":
    main()
