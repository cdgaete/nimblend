# Nimblend: Labeled N-Dimensional Arrays with Outer-Join Alignment

Nimblend is a lightweight Python library for labeled N-dimensional arrays. It solves a specific problem in energy system modeling: when combining data from different sources, missing combinations should contribute zero to calculations, not propagate as NaN.

## The Problem

In energy modeling, you often combine arrays with different coordinate coverage:

```python
# Generation capacity by region and technology
capacity = Array(...)  # regions: ['DE', 'FR'], techs: ['solar', 'wind']

# Capacity factors (efficiency) by technology  
cf = Array(...)        # techs: ['solar', 'wind', 'gas']

# Calculate potential generation
generation = capacity * cf
```

With standard tools like xarray, mismatched coordinates produce NaN values that cascade through calculations. In energy models, a missing combination typically means "zero capacity" or "not applicable" - a valid numeric value, not missing data.

## The Solution

Nimblend uses **outer-join alignment** with **zero-fill**:

1. **Outer join**: Results contain the union of all coordinates from both arrays
2. **Zero fill**: Missing values become 0, not NaN

This means:

- `solar_DE + wind_FR = solar_DE + wind_FR` (both preserved)
- `solar_DE * gas_DE = 0` (gas_DE doesn't exist in capacity, so 0 × anything = 0)

## Installation

```bash
pip install nimblend
```

Requires only NumPy.

## Quick Start

```python
import numpy as np
from nimblend import Array

# Create arrays with different regional coverage
de_fr_data = np.array([[100, 200], [150, 250]])
arr1 = Array(de_fr_data, {
    'region': ['DE', 'FR'],
    'year': [2020, 2030]
})

fr_es_data = np.array([[10, 20], [15, 25]])
arr2 = Array(fr_es_data, {
    'region': ['FR', 'ES'],
    'year': [2020, 2030]
})

# Addition: outer join preserves all regions
result = arr1 + arr2
print(result.coords['region'])  # ['DE', 'ES', 'FR']

# Result breakdown:
# DE: [100, 200] + [0, 0]   = [100, 200]  (arr2 has no DE)
# ES: [0, 0]     + [15, 25] = [15, 25]    (arr1 has no ES)
# FR: [150, 250] + [10, 20] = [160, 270]  (both have FR)
```

## Dimension Order Independence

Arrays can have dimensions in any order. Nimblend aligns by dimension name, not position:

```python
# Same data, different dimension order
arr1 = Array(data, {'region': r, 'tech': t, 'year': y})  # shape: (3, 4, 5)
arr2 = Array(data, {'tech': t, 'region': r, 'year': y})  # shape: (4, 3, 5)

result = arr1 + arr2  # Aligns by name, not position
```

This prevents subtle bugs when combining data from different sources that happen to structure dimensions differently.

## Operations

### Binary Operations with Alignment

All binary operations trigger automatic alignment:

```python
result = arr1 + arr2   # Addition
result = arr1 - arr2   # Subtraction  
result = arr1 * arr2   # Multiplication (0 where either is missing)
result = arr1 / arr2   # Division (caution: 0/0 = nan, x/0 = inf)
```

### Scalar Operations

Scalars apply element-wise without alignment:

```python
result = arr * 2.5     # Multiply all values
result = arr + 100     # Add to all values
result = arr ** 2      # Square all values
```

### Reductions

Sum over dimensions to aggregate:

```python
total = arr.sum()                      # Sum everything → scalar
by_region = arr.sum('year')            # Sum over year → Array
by_year = arr.sum(['region', 'tech'])  # Sum over multiple → Array
```

## Comparison with xarray

| Aspect | xarray | Nimblend |
|--------|--------|----------|
| Alignment | Inner join (intersection) | Outer join (union) |
| Missing values | NaN | 0 |
| Dependencies | Heavy (pandas, etc.) | NumPy only |
| Use case | General scientific data | Energy system modeling |

xarray is excellent for scientific data where NaN correctly represents "no measurement". Nimblend is designed for modeling scenarios where missing means "zero contribution".

## API Reference

### Array(data, coords, dims=None, name=None)

Create a labeled array.

**Parameters:**

- `data`: NumPy array with the values
- `coords`: Dict mapping dimension names to coordinate arrays
- `dims`: Optional list specifying dimension order (defaults to coords key order)
- `name`: Optional string name for the array

**Properties:**

- `shape`: Tuple of dimension sizes
- `dims`: List of dimension names  
- `coords`: Dict of coordinate arrays
- `data`: The underlying NumPy array

**Methods:**

- `sum(dim=None)`: Sum over dimension(s)

## License

MIT License
