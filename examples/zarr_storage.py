import os
import shutil

import numpy as np

from nimblend import Array, from_zarr, to_zarr

# Create a sample array with string coordinates
data = np.array(
    [
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
    ]
)

coords = {
    "x": np.array(["A", "B"]),
    "y": np.array(["one", "two"]),
    "z": np.array(["i", "ii", "iii"]),
}

# Create a NimbleNd Array
arr = Array(data, coords=coords, dims=["x", "y", "z"], name="Test Array")

# Save path
zarr_path = "test_array.zarr"

# Save array to Zarr format
print(f"Saving array to {zarr_path}")
to_zarr(arr, zarr_path)

# Load from Zarr
print(f"Loading array from {zarr_path}")
loaded = from_zarr(zarr_path)

# Verify the loaded array matches the original
print("\nOriginal array:")
print(arr.data)
print(arr.coords)

print("\nLoaded array:")
print(loaded.data)
print(loaded.coords)

# Clean up - remove the zarr directory after the example
if os.path.exists(zarr_path):
    shutil.rmtree(zarr_path)
    print(f"\nRemoved {zarr_path}")
