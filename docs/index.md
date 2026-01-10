# Nimblend

**Lightweight labeled N-dimensional arrays with outer-join alignment.**

Nimblend is a pure NumPy library for working with labeled arrays. It provides automatic coordinate alignment on binary operations using outer-join semantics, with zero-fill for missing values.

## Features

- **Labeled dimensions** - Name your axes and index by coordinate labels
- **Automatic alignment** - Binary operations align arrays by coordinates
- **Outer-join semantics** - Results contain union of all coordinates
- **Zero-fill** - Missing values become 0, not NaN
- **Pure NumPy** - No heavy dependencies

## Quick Example

```python
import numpy as np
from nimblend import Array

# Create labeled arrays
sales_q1 = Array(
    np.array([100, 200, 150]),
    {"product": ["A", "B", "C"]}
)

sales_q2 = Array(
    np.array([120, 180, 90]),
    {"product": ["B", "C", "D"]}  # Different products!
)

# Automatic alignment on addition
total = sales_q1 + sales_q2
print(total.coords["product"])  # ['A', 'B', 'C', 'D']
print(total.data)               # [100, 320, 240, 90]
#                                   A: 100+0, B: 200+120, C: 150+90, D: 0+90
```

## Installation

```bash
pip install nimblend
```

## Why Nimblend?

- **Simpler than xarray** - No Dask, no NetCDF, just arrays
- **Faster startup** - Pure NumPy means minimal import time  
- **Predictable** - Zero-fill instead of NaN propagation
- **Lightweight** - Single file, ~800 lines of code
