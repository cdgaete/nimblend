import numpy as np

from nimblend import Array

# Try to import dask to check if available
try:
    import dask.array as da

    HAS_DASK = True
except ImportError:
    HAS_DASK = False
    print("Dask not installed. Install with: pip install dask[array]")

if HAS_DASK:
    # Create a large dataset that benefits from lazy computation
    shape = (1000, 1000)  # 1 million elements

    # Create coordinates
    coords = {"x": np.arange(shape[0]), "y": np.arange(shape[1])}

    print("Creating a large array with lazy computation...")

    # Create a lazy array directly without materializing
    # Let's create something computationally expensive
    x, y = np.meshgrid(coords["x"], coords["y"], indexing="ij")

    # Convert to dask arrays first for efficiency
    x_dask = da.from_array(x, chunks=(100, 100))
    y_dask = da.from_array(y, chunks=(100, 100))

    # Create a complex computation that benefits from lazy evaluation
    data = da.sin(x_dask / 50) * da.cos(y_dask / 50) + da.exp(
        -(x_dask**2 + y_dask**2) / 100000
    )

    # Create the NimbleNd Array with lazy data
    lazy_array = Array(data, coords, dims=["x", "y"], name="Large Computation")

    print(f"Array shape: {lazy_array.data.shape}")
    print(f"Is lazy: {lazy_array.is_lazy}")
    print(f"Chunk structure: {lazy_array.data.chunks}")

    # Compute a subset without materializing the entire array
    subset = lazy_array[{"x": slice(0, 100), "y": slice(0, 100)}]
    print(f"\nSubset shape: {subset.data.shape}")
    print(f"Subset is still lazy: {subset.is_lazy}")

    # Materialize just the subset
    print("\nMaterializing subset...")
    materialized_subset = subset.compute()
    print(f"Materialized shape: {materialized_subset.data.shape}")
    print(f"Is lazy: {materialized_subset.is_lazy}")

    # Demonstrate persist (keeps the lazy structure but caches in memory)
    print("\nPersisting subset in memory...")
    persisted = subset.persist()
    print(f"Persisted is still lazy: {persisted.is_lazy}")
    print("But data is now in memory for faster access")
else:
    print(
        "This example requires Dask. Please install it with 'pip install dask[array]'"
    )
