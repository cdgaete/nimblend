"""Profile hot paths in nimblend to identify bottlenecks."""

import cProfile
import pstats
import io
import numpy as np

import sys
sys.path.insert(0, '/home/carlos/projects/nimblend/src')
import nimblend


def profile_aligned_add():
    """Profile addition with aligned arrays."""
    n = 500
    data1 = np.random.randn(n, n)
    data2 = np.random.randn(n, n)
    x_coords = [f"x{i}" for i in range(n)]
    y_coords = [f"y{i}" for i in range(n)]
    
    nb1 = nimblend.Array(data1, {"x": x_coords, "y": y_coords})
    nb2 = nimblend.Array(data2, {"x": x_coords, "y": y_coords})
    
    pr = cProfile.Profile()
    pr.enable()
    
    for _ in range(10):
        _ = nb1 + nb2
    
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print("=== ALIGNED ADD (500x500, 10 iterations) ===")
    print(s.getvalue())


def profile_misaligned_add():
    """Profile addition with misaligned arrays (outer join)."""
    n = 500
    data1 = np.random.randn(n, n)
    data2 = np.random.randn(n, n)
    x1 = [f"x{i}" for i in range(n)]
    x2 = [f"x{i}" for i in range(n//2, n + n//2)]  # 50% overlap
    y_coords = [f"y{i}" for i in range(n)]
    
    nb1 = nimblend.Array(data1, {"x": x1, "y": y_coords})
    nb2 = nimblend.Array(data2, {"x": x2, "y": y_coords})
    
    pr = cProfile.Profile()
    pr.enable()
    
    for _ in range(10):
        _ = nb1 + nb2
    
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print("=== MISALIGNED ADD (500x500, 50% overlap, 10 iterations) ===")
    print(s.getvalue())


def profile_multi_sel():
    """Profile multi-value selection."""
    n = 1000
    data = np.random.randn(n, n)
    x_coords = [f"x{i}" for i in range(n)]
    y_coords = [f"y{i}" for i in range(n)]
    
    nb_arr = nimblend.Array(data, {"x": x_coords, "y": y_coords})
    targets = [f"x{i}" for i in range(n//4, 3*n//4)]
    
    pr = cProfile.Profile()
    pr.enable()
    
    for _ in range(10):
        _ = nb_arr.sel({"x": targets})
    
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print("=== MULTI-SEL (1000x1000, 500 items, 10 iterations) ===")
    print(s.getvalue())


if __name__ == "__main__":
    profile_aligned_add()
    print("\n" + "="*80 + "\n")
    profile_misaligned_add()
    print("\n" + "="*80 + "\n")
    profile_multi_sel()
