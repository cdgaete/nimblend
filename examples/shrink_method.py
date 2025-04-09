import numpy as np

from nimblend import Array

# Try to import dask to check if available
try:
    import dask.array as da

    HAS_DASK = True
except ImportError:
    HAS_DASK = False
    print("Dask not installed. Install with: pip install dask[array]")


def section(title):
    """Print a section divider with title."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


section("Shrinking Eager (NumPy) Arrays")

# Create a sample 3D array
data = np.arange(60).reshape(3, 4, 5)
coords = {
    "time": ["2023-01-01", "2023-01-02", "2023-01-03"],
    "lat": [30, 35, 40, 45],
    "lon": [70, 75, 80, 85, 90],
}

# Create a NimbleNd Array
array = Array(data, coords, dims=["time", "lat", "lon"], name="Temperature Data")

print("Original array:")
print(f"Shape: {array.data.shape}")
print(f"Dimensions: {array.dims}")
print("Coordinates:")
for dim, values in array.coords.items():
    print(f"  {dim}: {values}")
print(f"Data sample (first element): {array.data[0, 0, 0]}")

# Shrink along a single dimension
shrunk_single = array.shrink({"time": ["2023-01-01", "2023-01-03"]})

print("\nArray after shrinking along 'time' dimension:")
print(f"Shape: {shrunk_single.data.shape}")
print("Coordinates:")
for dim, values in shrunk_single.coords.items():
    print(f"  {dim}: {values}")
print(f"Data sample (first element): {shrunk_single.data[0, 0, 0]}")

# Shrink along multiple dimensions
shrunk_multiple = array.shrink(
    {"time": ["2023-01-01", "2023-01-03"], "lat": [30, 40], "lon": [70, 80, 90]}
)

print("\nArray after shrinking along multiple dimensions:")
print(f"Shape: {shrunk_multiple.data.shape}")
print("Coordinates:")
for dim, values in shrunk_multiple.coords.items():
    print(f"  {dim}: {values}")
print(f"Data sample (first element): {shrunk_multiple.data[0, 0, 0]}")

# Example with non-sequential indices
shrunk_nonseq = array.shrink(
    {
        "time": ["2023-01-03", "2023-01-01"],  # Reversed order
        "lat": [45, 35],  # Non-sequential
        "lon": [90, 70],  # First and last
    }
)

print("\nArray after shrinking with non-sequential indices:")
print(f"Shape: {shrunk_nonseq.data.shape}")
print("Coordinates:")
for dim, values in shrunk_nonseq.coords.items():
    print(f"  {dim}: {values}")
print(f"Data sample (first element): {shrunk_nonseq.data[0, 0, 0]}")

if HAS_DASK:
    section("Shrinking Lazy (Dask) Arrays")

    # Create a larger data array for demonstrating Dask
    lazy_data = da.random.random((5, 10, 15), chunks=(2, 3, 4))
    lazy_coords = {
        "time": [f"2023-01-{i + 1:02d}" for i in range(5)],
        "lat": np.linspace(25, 45, 10),
        "lon": np.linspace(70, 100, 15),
    }

    # Create a lazy NimbleNd Array
    lazy_array = Array(
        lazy_data, lazy_coords, dims=["time", "lat", "lon"], name="Large Weather Data"
    )

    print("Original lazy array:")
    print(f"Shape: {lazy_array.data.shape}")
    print(f"Chunks: {lazy_array.data.chunks}")
    print(f"Dimensions: {lazy_array.dims}")
    print(f"Is lazy: {lazy_array.is_lazy}")

    # Shrink the lazy array
    shrunk_lazy = lazy_array.shrink(
        {
            "time": [
                f"2023-01-{i + 1:02d}" for i in range(0, 5, 2)
            ],  # Every other date
            "lat": np.linspace(25, 45, 10)[::3],  # Every third latitude
            "lon": np.linspace(70, 100, 15)[::5],  # Every fifth longitude
        }
    )

    print("\nLazy array after shrinking:")
    print(f"Shape: {shrunk_lazy.data.shape}")
    print(f"Chunks: {shrunk_lazy.data.chunks}")
    print(f"Is lazy: {shrunk_lazy.is_lazy}")
    print(f"Dimensions: {shrunk_lazy.dims}")

    print("\nCoordinates after shrinking:")
    for dim, values in shrunk_lazy.coords.items():
        print(f"  {dim}: {values}")

    # Compute a lazy array to show actual values
    print("\nComputing a small section of the lazy array...")
    small_section = shrunk_lazy[
        {"time": ["2023-01-01"], "lat": [shrunk_lazy.coords["lat"][0]]}
    ]
    computed = small_section.compute()

    print(f"Computed shape: {computed.data.shape}")
    print("Sample data:")
    print(computed.data[0, 0, :])
else:
    print("\nSkipping lazy array examples as Dask is not installed")

section("Advanced Example: Combining Shrink with Operations")

# Create two arrays with different coordinate ranges
array1 = Array(
    np.random.rand(4, 6),
    coords={"x": ["a", "b", "c", "d"], "y": [10, 20, 30, 40, 50, 60]},
    name="Array 1",
)

array2 = Array(
    np.random.rand(5, 4),
    coords={"x": ["b", "c", "d", "e", "f"], "y": [20, 30, 40, 50]},
    name="Array 2",
)

print("Array 1 shape and coords:")
print(f"Shape: {array1.data.shape}")
print(f"Coordinates: {array1.coords}")

print("\nArray 2 shape and coords:")
print(f"Shape: {array2.data.shape}")
print(f"Coordinates: {array2.coords}")

# Shrink both arrays to matching coordinates
shrunk1 = array1.shrink({"x": ["b", "c", "d"], "y": [20, 30, 40, 50]})
shrunk2 = array2.shrink({"x": ["b", "c", "d"], "y": [20, 30, 40, 50]})

print("\nAfter shrinking to matching coordinates:")
print(f"Array 1 shape: {shrunk1.data.shape}")
print(f"Array 2 shape: {shrunk2.data.shape}")

# Perform an operation between the aligned arrays
result = shrunk1 + shrunk2

print("\nResult of operation between aligned arrays:")
print(f"Shape: {result.data.shape}")
print(f"Coordinates: {result.coords}")

print("\nExample complete!")
