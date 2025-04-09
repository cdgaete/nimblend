import numpy as np

from nimblend import Array

# Create two arrays with different coordinates
data1 = np.array([[1, 2], [3, 4]])
coords1 = {"x": ["a", "b"], "y": [0, 1]}
arr1 = Array(data1, coords1)
print("Array 1:")
print(arr1.data)
print(arr1.coords)

data2 = np.array([[10, 20, 30], [40, 50, 60]])
coords2 = {"x": ["b", "c"], "y": [0, 1, 2]}
arr2 = Array(data2, coords2)
print("\nArray 2:")
print(arr2.data)
print(arr2.coords)

# Operation automatically aligns coordinates
result = arr1 + arr2
print("\nResult of addition (with automatic alignment):")
print(result.data)
print(result.coords)

# Create a lazy array with Dask
lazy_arr = Array(data1, coords1, chunks="auto")
print(f"\nIs lazy array?: {lazy_arr.is_lazy}")  # True if dask is installed
