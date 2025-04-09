import os
import shutil

import icechunk
import numpy as np

from nimblend import Array, from_icechunk, to_icechunk


# Display section dividers for better readability
def section(title):
    print(f"\n{'=' * 50}")
    print(f" {title}")
    print(f"{'=' * 50}")


section("Creating Sample Arrays")

# Create a sample array with numeric data
data = np.array([[1, 2, 3], [4, 5, 6]])
coords = {"x": ["a", "b"], "y": [10, 20, 30]}
arr = Array(data, coords, name="Numeric Array")
print(f"Created array with shape {arr.data.shape}")
print(f"Dimensions: {arr.dims}")
print(f"Coordinates: {arr.coords}")

# Create a second array with string data to demonstrate string handling
string_data = np.array([["red", "green", "blue"], ["cyan", "magenta", "yellow"]])
string_arr = Array(string_data, coords, name="Color Array")
print(f"\nCreated string array with shape {string_arr.data.shape}")

section("Setting Up IceChunk Storage")

# Create temporary directory for storage
storage_path = "temp_icechunk_storage"
os.makedirs(storage_path, exist_ok=True)

# Initialize IceChunk storage
storage = icechunk.local_filesystem_storage(storage_path)
repo = icechunk.Repository.open_or_create(storage)
session = repo.writable_session("main")
print(f"IceChunk repository initialized at: {storage_path}")

section("Storing Arrays in IceChunk")

# Store the numeric array
print("Storing numeric array...")
to_icechunk(arr, session, group="numeric_data")
print("Numeric array stored successfully")

# Store the string array
print("\nStoring string array...")
to_icechunk(string_arr, session, group="string_data")
print("String array stored successfully")

# Commit the changes to the repository
print("\nCommitting changes to the repository...")
session.commit("Added sample arrays")
print("Changes committed")

section("Retrieving Arrays from IceChunk")

# Create a readonly session for retrieval
readonly_session = repo.readonly_session("main")

# Load the numeric array
print("Loading numeric array...")
loaded_arr = from_icechunk(readonly_session, group="numeric_data")
print(f"Loaded array name: {loaded_arr.name}")
print(f"Loaded array shape: {loaded_arr.data.shape}")
print(f"Loaded array type: {type(loaded_arr.data).__name__}")
print(f"Sample data: {loaded_arr.data[0, 0:3]}")

# Load the string array
print("\nLoading string array...")
loaded_string_arr = from_icechunk(readonly_session, group="string_data")
print(f"Loaded string array name: {loaded_string_arr.name}")
print(f"Loaded string array shape: {loaded_string_arr.data.shape}")
print(f"Loaded string array dtype: {loaded_string_arr.data.dtype}")
print(f"Sample data: {loaded_string_arr.data[0, 0:3]}")

section("Loading as Lazy Dask Arrays")

try:
    # Load as lazy Dask arrays
    print("Loading as lazy Dask arrays...")
    lazy_arr = from_icechunk(readonly_session, group="numeric_data", chunks="auto")
    print(f"Lazy array type: {type(lazy_arr.data).__name__}")
    print(f"Lazy array chunks: {lazy_arr.data.chunks}")

    # Perform a computation without loading the entire array
    mean_value = lazy_arr.data.mean().compute()
    print(f"Mean value (computed lazily): {mean_value}")
except ImportError:
    print("Dask not available - skipping lazy loading demonstration")

section("Data Modification and Versioning")

# Create a new session to demonstrate versioning
new_session = repo.writable_session("main")

# Modify the array and save it
modified_arr = loaded_arr * 10  # Multiply all values by 10
print("Storing modified array...")
to_icechunk(modified_arr, new_session, group="numeric_data", mode="a")
new_session.commit("Modified numeric array (×10)")
print("Modified array stored and committed")

# Load the modified array from the latest version
latest_session = repo.readonly_session("main")
latest_arr = from_icechunk(latest_session, group="numeric_data")
print(f"\nLoaded modified array. Sample data: {latest_arr.data[0, 0:3]}")

section("Cleanup")

# Clean up (optional)
if os.path.exists(storage_path):
    shutil.rmtree(storage_path)
    print(f"Removed temporary storage directory: {storage_path}")

print("\nExample completed successfully!")
