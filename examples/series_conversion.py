import numpy as np
import pandas as pd

from nimblend import from_series, to_series

# Create a sample pandas Series with a MultiIndex
# This represents data in a "long" format
idx = pd.MultiIndex.from_product(
    [
        ["A", "B", "C"],  # dimension 1 values
        [1, 2, 3, 4],  # dimension 2 values
    ],
    names=["dim_x", "dim_y"],
)

# Create random data for the Series
np.random.seed(42)
data = np.random.rand(12)  # 3 x-values × 4 y-values = 12 points
series = pd.Series(data, index=idx, name="measurements")

print("Original pandas Series:")
print(series)

# Convert Series to NimbleNd Array
nimble_array = from_series(series)

print("\nConverted to NimbleNd Array:")
print(f"Shape: {nimble_array.data.shape}")
print(f"Dimensions: {nimble_array.dims}")
print(f"Coordinates: {nimble_array.coords}")
print(f"Data:\n{nimble_array.data}")

# Convert the Array back to a Series
series_again = to_series(nimble_array, format="pandas")

print("\nConverted back to pandas Series:")
print(series_again)

# Optionally demonstrate polars conversion if polars is installed
try:
    import polars as pl

    # Create a polars DataFrame with explicit dimension columns
    # For polars, we need to work with a DataFrame in long format
    # because polars doesn't support MultiIndex
    print("\n--- Polars Example ---")

    # Create a proper long-format DataFrame for polars
    df_long = pd.DataFrame(series).reset_index()
    print("\nLong format DataFrame (before polars conversion):")
    print(df_long.head())

    # Convert to polars DataFrame
    pl_df = pl.from_pandas(df_long)
    print("\nPolars DataFrame:")
    print(pl_df)

    # Convert the entire DataFrame to a NimbleNd Array
    # Note: in this approach, we use the DataFrame directly rather than a Series
    print("\nConverting from polars DataFrame to NimbleNd Array...")
    from_polars_df = from_series(pl_df["measurements"], dims=["dim_x", "dim_y"])

    print("\nConverted from polars to NimbleNd Array:")
    print(f"Shape: {from_polars_df.data.shape}")
    print(f"Dimensions: {from_polars_df.dims}")
    print(f"Data:\n{from_polars_df.data}")

    # Convert NimbleNd Array back to polars
    print("\nConverting back to polars...")
    polars_again = to_series(nimble_array, format="polars")
    print("\nConverted back to polars Series (via a DataFrame):")
    print(polars_again)

    # Show the parent DataFrame to illustrate the structure
    print("\nParent DataFrame of the polars Series:")
    # Get the parent DataFrame from the Series
    parent_df = polars_again._df
    print(parent_df)

except ImportError:
    print("\nPolars not installed, skipping polars example")
