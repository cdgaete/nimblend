import numpy as np

from nimblend import Array

# Create a sample array
data = np.array([[1, 2, 3], [4, 5, 6]])
coords = {"x": ["a", "b"], "y": [10, 20, 30]}
arr = Array(data, coords)

# 1. Dictionary-based indexing by coordinate values
# Get the value at x="a", y=20
value1 = arr[{"x": "a", "y": 20}]  # Returns value at position [0, 1] which is 2

# Get all values for x="b"
values_for_b = arr[{"x": "b"}]  # Returns Array with data [4, 5, 6]

# Get values for a list of coordinates
multiple_y = arr[{"y": [10, 30]}]  # Returns Array with first and third y-values

print(multiple_y.data)
