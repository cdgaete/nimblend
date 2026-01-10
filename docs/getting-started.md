# Getting Started

## Installation

```bash
pip install nimblend
```

Or install from source:

```bash
git clone https://github.com/your-username/nimblend.git
cd nimblend
pip install -e .
```

## Creating Arrays

A nimblend `Array` wraps a NumPy array with labeled dimensions:

```python
import numpy as np
from nimblend import Array

# 1D array
temps = Array(
    np.array([20.5, 22.1, 19.8]),
    {"city": ["NYC", "LA", "Chicago"]}
)

# 2D array
data = Array(
    np.array([[1, 2, 3], [4, 5, 6]]),
    {"x": ["a", "b"], "y": [10, 20, 30]}
)
```

## Selection

### By Label (`sel`)

```python
# Single value - reduces dimension
temps.sel({"city": "NYC"})  # Returns: 20.5

# Multiple values - preserves dimension  
temps.sel({"city": ["NYC", "LA"]})  # Returns Array with 2 values

# Shorthand with __getitem__
data[{"x": "a"}]  # Same as data.sel({"x": "a"})
```

### By Index (`isel`)

```python
temps.isel({"city": 0})      # First city
temps.isel({"city": -1})     # Last city
temps.isel({"city": [0, 2]}) # First and third
```

### Slicing

```python
data[0]      # First row
data[0:2]    # First two rows
data["x"]    # Get coordinate values for dimension "x"
```

## Arithmetic with Alignment

The key feature of nimblend is automatic coordinate alignment:

```python
a = Array(np.array([1, 2, 3]), {"x": ["p", "q", "r"]})
b = Array(np.array([10, 20]), {"x": ["q", "s"]})

result = a + b
# Coordinates: ["p", "q", "r", "s"]
# Values:      [1,   22,  3,   20]
#              p:1+0  q:2+10  r:3+0  s:0+20
```

Missing values are filled with **zero**, not NaN.

## Reductions

```python
data = Array(np.arange(6).reshape(2, 3), {"x": ["a", "b"], "y": [0, 1, 2]})

data.sum()           # Sum all: 15
data.sum("x")        # Sum along x: Array([3, 5, 7])
data.mean("y")       # Mean along y: Array([1.0, 4.0])
data.max()           # Maximum: 5
```

Available: `sum`, `mean`, `min`, `max`, `std`, `prod`

## Shape Manipulation

```python
# Transpose
data.T                    # Reverse dimensions
data.transpose("y", "x")  # Explicit order

# Add/remove dimensions
arr.expand_dims("time", coord="2024")
arr.squeeze("time")  # Remove size-1 dimension
```

## Conditional Operations

```python
# Replace values where condition is False
arr.where(arr > 0, 0)  # Negative values become 0

# Clamp to range
arr.clip(0, 100)       # Values bounded to [0, 100]
```

## Comparison

```python
arr1 == arr2      # Element-wise, returns boolean Array
arr > 5           # Compare with scalar
arr1.equals(arr2) # True if identical (dims, coords, data)
```
