"""
Core classes for NimbleNd with Dask support.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np

# Check for Dask availability
try:
    import dask.array as da

    HAS_DASK = True
except ImportError:
    HAS_DASK = False

# Type variable for array-like objects
ArrayLike = TypeVar("ArrayLike", np.ndarray, "da.Array")


class CoordinateMap:
    """
    Efficient mapping between dimension coordinates and their indices.

    This class handles the relationship between labeled coordinates and
    their positions in the array, with lazy generation of index mappings.
    """

    def __init__(self, dims: List[str], coords: Dict[str, np.ndarray]):
        """
        Initialize a CoordinateMap.

        Parameters
        ----------
        dims : List[str]
            List of dimension names
        coords : Dict[str, np.ndarray]
            Dictionary mapping dimension names to their coordinate values
        """
        self.dims = dims
        self.coords = {dim: np.asarray(coords[dim]) for dim in dims}
        self._indices = {}  # Lazy index mapping

        # Validate inputs
        for dim in dims:
            if dim not in coords:
                raise ValueError(f"Dimension '{dim}' not found in coordinates")
            if len(np.unique(coords[dim])) != len(coords[dim]):
                raise ValueError(f"Coordinates for dimension '{dim}' must be unique")

    def get_index_mapping(self, dim: str) -> Dict[Any, int]:
        """
        Get a mapping from coordinate values to their indices for a dimension.

        Parameters
        ----------
        dim : str
            Dimension name

        Returns
        -------
        Dict[Any, int]
            Mapping from coordinate values to their positions
        """
        if dim not in self._indices:
            self._indices[dim] = {val: i for i, val in enumerate(self.coords[dim])}
        return self._indices[dim]

    def get_indices(self, dim: str, values: np.ndarray) -> np.ndarray:
        """
        Get the indices for a set of coordinate values in a dimension.

        Parameters
        ----------
        dim : str
            Dimension name
        values : np.ndarray
            Coordinate values to look up

        Returns
        -------
        np.ndarray
            Indices corresponding to the values
        """
        mapping = self.get_index_mapping(dim)
        indices = np.array([mapping.get(v, -1) for v in values])
        if np.any(indices == -1):
            raise ValueError(f"Some values not found in dimension '{dim}'")
        return indices

    def get_position(self, coords_dict: Dict[str, Any]) -> Tuple[int, ...]:
        """
        Get the position of a set of coordinates in the array.

        Parameters
        ----------
        coords_dict : Dict[str, Any]
            Dictionary mapping dimension names to coordinate values

        Returns
        -------
        Tuple[int, ...]
            Position in the array
        """
        position = []
        for dim in self.dims:
            if dim not in coords_dict:
                raise ValueError(f"Missing coordinate for dimension '{dim}'")
            mapping = self.get_index_mapping(dim)
            if coords_dict[dim] not in mapping:
                raise ValueError(
                    f"Coordinate {coords_dict[dim]} not found in dimension '{dim}'"
                )
            position.append(mapping[coords_dict[dim]])
        return tuple(position)

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Get the shape implied by the coordinates.

        Returns
        -------
        Tuple[int, ...]
            Shape of the array that would be indexed by these coordinates
        """
        return tuple(len(self.coords[dim]) for dim in self.dims)


class Array:
    """
    Main class for working with labeled N-dimensional arrays.

    This class combines an array (NumPy or Dask) with coordinate information,
    allowing for operations that properly align coordinates.
    """

    def __init__(
        self,
        data: Union[np.ndarray, "da.Array", List, Tuple],
        coords: Dict[str, np.ndarray],
        dims: Optional[List[str]] = None,
        name: Optional[str] = None,
        chunks: Optional[Union[str, Tuple[int, ...], Dict[str, int]]] = None,
    ):
        """
        Initialize an Array.

        Parameters
        ----------
        data : Union[np.ndarray, da.Array, List, Tuple]
            The array data, can be NumPy array, Dask array, or convertible
        coords : Dict[str, np.ndarray]
            Dictionary mapping dimension names to their coordinate values
        dims : Optional[List[str]], optional
            List of dimension names (default: sorted keys of coords)
        name : Optional[str], optional
            Name for the array
        chunks : Optional[Union[str, Tuple[int, ...], Dict[str, int]]], optional
            Chunk specification for Dask arrays. If data is already a Dask array,
            this is ignored unless it's "auto". If "auto", uses heuristics to
            determine good chunk sizes. For NumPy arrays, this determines if they
            should be converted to Dask arrays.
        """
        # Handle dimensions if not provided
        if dims is None:
            dims = sorted(coords.keys())

        # Convert coordinates to numpy arrays
        coords_arrays = {dim: np.asarray(coords[dim]) for dim in dims}

        # Validate dimensions match data shape
        self._validate_dims_shape(data, dims, coords_arrays)

        # Convert data to the right format with appropriate chunking
        self.data = self._prepare_data(data, chunks, dims, coords_arrays)

        # Store coordinates and dimensions
        self.coords = coords_arrays
        self.dims = dims
        self.name = name
        self.coordinate_map = CoordinateMap(dims, self.coords)

    def _validate_dims_shape(
        self,
        data: Union[np.ndarray, "da.Array", List, Tuple],
        dims: List[str],
        coords: Dict[str, np.ndarray],
    ) -> None:
        """Validate dimensions match data shape."""
        # Get the shape of the data
        if hasattr(data, "shape"):
            shape = data.shape
        else:
            # Convert to numpy array to get shape
            shape = np.asarray(data).shape

        # Validate dimensions match
        if len(dims) != len(shape):
            raise ValueError(
                f"Number of dimensions ({len(dims)}) must match data dimensions ({len(shape)})"
            )

        # Validate each dimension size matches coordinate length
        for i, dim in enumerate(dims):
            if dim not in coords:
                raise ValueError(f"Dimension '{dim}' not found in coordinates")
            if len(coords[dim]) != shape[i]:
                raise ValueError(
                    f"Length of coordinates for dimension '{dim}' ({len(coords[dim])}) "
                    f"does not match data shape ({shape[i]})"
                )

    def _prepare_data(
        self,
        data: Union[np.ndarray, "da.Array", List, Tuple],
        chunks: Optional[Union[str, Tuple[int, ...], Dict[str, int]]],
        dims: List[str],
        coords: Dict[str, np.ndarray],
    ) -> Union[np.ndarray, "da.Array"]:
        """Prepare the data array with appropriate chunking."""
        # Check if it's already a Dask array
        if HAS_DASK and isinstance(data, da.Array):
            if chunks == "auto":
                # Get shape for chunk calculation
                shape = data.shape
                # Rechunk using heuristics
                chunk_sizes = self._compute_chunks(shape, dims)
                return data.rechunk(chunk_sizes)
            return data

        # Convert to numpy array if not already
        array_data = np.asarray(data)

        # If chunks is specified and Dask is available, convert to Dask array
        if HAS_DASK and chunks is not None:
            if chunks == "auto":
                # Determine good chunk sizes heuristically
                chunk_sizes = self._compute_chunks(array_data.shape, dims)
                return da.from_array(array_data, chunks=chunk_sizes)
            return da.from_array(array_data, chunks=chunks)

        # Otherwise, keep as numpy array
        return array_data

    def _compute_chunks(
        self, shape: Tuple[int, ...], dims: List[str]
    ) -> Tuple[int, ...]:
        """Compute good chunk sizes based on data shape and dimensions."""
        # Start with a reasonable minimum chunk size
        chunk_sizes = []
        for i, dim_size in enumerate(shape):
            # Simple heuristic: ~1000 elements per dimension or the full dimension
            chunk_size = min(max(1, dim_size // 4), dim_size)
            chunk_sizes.append(chunk_size)

        return tuple(chunk_sizes)

    @property
    def is_lazy(self) -> bool:
        """Check if the data is a lazy (Dask) array."""
        return HAS_DASK and isinstance(self.data, da.Array)

    def compute(self) -> "Array":
        """
        Compute a lazy array, returning a new Array with an eager NumPy array.

        Returns
        -------
        Array
            A new Array with computed data
        """
        if not self.is_lazy:
            return self

        computed_data = self.data.compute()
        return Array(computed_data, self.coords, self.dims, self.name)

    def persist(self) -> "Array":
        """
        Persist a lazy array in memory, maintaining the lazy representation.

        Returns
        -------
        Array
            A new Array with persisted data
        """
        if not self.is_lazy:
            return self

        persisted_data = self.data.persist()
        return Array(persisted_data, self.coords, self.dims, self.name)

    def __repr__(self) -> str:
        """String representation of the array."""
        # Build dimensions string without curly braces
        dims_parts = []
        for dim in self.dims:
            dims_parts.append("'{}': {}".format(dim, self.coords[dim].shape))
        dims_str = ", ".join(dims_parts)

        name_str = ""
        if self.name:
            name_str = ", name='{}'".format(self.name)

        lazy_str = " (lazy)" if self.is_lazy else ""

        # Using parentheses instead of curly braces for the dimensions section
        return "Array({}{}, [{}]{})".format(
            self.data.shape, lazy_str, dims_str, name_str
        )

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        # Build dimensions string without curly braces
        dims_parts = []
        for dim in self.dims:
            dims_parts.append("<b>{}</b>: {}".format(dim, self.coords[dim].shape))
        dims_str = ", ".join(dims_parts)

        name_str = ""
        if self.name:
            name_str = ", name=<b>{}</b>".format(self.name)

        lazy_str = " <i>(lazy)</i>" if self.is_lazy else ""

        html = [
            "<div>",
            # Using square brackets instead of curly braces for the dimensions section
            "<p><b>Array</b>({}){}, [{}]{}</p>".format(
                self.data.shape, lazy_str, dims_str, name_str
            ),
            "<table>",
            "<tr><th>Dimensions</th><th>Coordinates</th></tr>",
        ]

        for dim in self.dims:
            coord_row = "<tr><td>{}</td><td>{}".format(dim, self.coords[dim][:10])
            html.append(coord_row)
            if len(self.coords[dim]) > 10:
                html.append("...")
            html.append("</td></tr>")

        html.append("</table>")

        # Show a small preview of the data
        html.append("<p><b>Data preview:</b></p>")
        html.append("<pre>")
        if self.is_lazy:
            preview = str(self.data)
        else:
            preview = str(self.data.flat[:10])
            if self.data.size > 10:
                preview = preview[:-1] + ", ...]"
        html.append(preview)
        html.append("</pre>")

        html.append("</div>")
        return "".join(html)

    def _align_with(
        self, other: "Array"
    ) -> Tuple[
        Union[np.ndarray, "da.Array"],
        Union[np.ndarray, "da.Array"],
        Dict[str, np.ndarray],
        List[str],
    ]:
        """
        Align this array with another array for operations, preserving laziness if present.
        """
        # Find union of dimensions
        all_dims = list(set(self.dims) | set(other.dims))

        # Find union of coordinates for each dimension
        union_coords = {}
        for dim in all_dims:
            if dim in self.coords and dim in other.coords:
                union_coords[dim] = np.unique(
                    np.concatenate([self.coords[dim], other.coords[dim]])
                )
            elif dim in self.coords:
                union_coords[dim] = self.coords[dim].copy()
            else:
                union_coords[dim] = other.coords[dim].copy()

        # Create result arrays with the union shape - maintaining laziness
        result_shape = tuple(len(union_coords[dim]) for dim in all_dims)

        # Determine if result should be lazy
        use_dask = self.is_lazy or other.is_lazy

        # Get indices of self and other in the aligned arrays
        self_indices = []
        other_indices = []

        for dim in all_dims:
            # Create mappings from coordinate values to positions in union_coords
            dim_map = {val: i for i, val in enumerate(union_coords[dim])}

            if dim in self.dims:
                # Map self coordinates to positions in union
                self_dim_indices = [dim_map[val] for val in self.coords[dim]]
                self_indices.append(self_dim_indices)
            else:
                # Use all indices for dimensions not in self
                self_indices.append(np.zeros(1, dtype=int))

            if dim in other.dims:
                # Map other coordinates to positions in union
                other_dim_indices = [dim_map[val] for val in other.coords[dim]]
                other_indices.append(other_dim_indices)
            else:
                # Use all indices for dimensions not in other
                other_indices.append(np.zeros(1, dtype=int))

        # Create meshgrids for all combinations of indices
        self_mesh = np.meshgrid(*self_indices, indexing="ij")
        other_mesh = np.meshgrid(*other_indices, indexing="ij")

        # Create output arrays - either NumPy or Dask
        if use_dask and HAS_DASK:
            # For Dask arrays, create empty arrays with appropriate chunks
            chunks = "auto"  # We could be more sophisticated here
            self_expanded = da.zeros(result_shape, dtype=self.data.dtype, chunks=chunks)
            other_expanded = da.zeros(
                result_shape, dtype=other.data.dtype, chunks=chunks
            )

            # Using the Dask map_blocks approach would be more efficient but complex
            # For simplicity, we'll use a basic approach

            # Flatten meshes for indexing
            self_flat_indices = tuple(m.flatten() for m in self_mesh)
            other_flat_indices = tuple(m.flatten() for m in other_mesh)

            # Create sparse arrays with values at the right positions
            self_data_flat = self.data.flatten()
            other_data_flat = other.data.flatten()

            # This simplified approach materializes arrays, not ideal for large arrays
            if isinstance(self_data_flat, da.Array):
                self_data_flat = self_data_flat.compute()
            if isinstance(other_data_flat, da.Array):
                other_data_flat = other_data_flat.compute()

            # Set values in expanded arrays
            # In a production implementation, we would use a more efficient approach
            # that preserves laziness throughout the operation
            for i in range(len(self_data_flat)):
                idx = tuple(fidx[i] for fidx in self_flat_indices)
                self_expanded[idx] = self_data_flat[i]

            for i in range(len(other_data_flat)):
                idx = tuple(fidx[i] for fidx in other_flat_indices)
                other_expanded[idx] = other_data_flat[i]
        else:
            # For NumPy arrays, use a simpler approach
            self_expanded = np.zeros(result_shape, dtype=self.data.dtype)
            other_expanded = np.zeros(result_shape, dtype=other.data.dtype)

            # Flatten meshes for indexing
            self_flat_indices = tuple(m.flatten() for m in self_mesh)
            other_flat_indices = tuple(m.flatten() for m in other_mesh)

            # Create sparse arrays with values at the right positions
            self_data_flat = self.data.flatten()
            other_data_flat = other.data.flatten()

            # Set values in expanded arrays
            for i in range(len(self_data_flat)):
                idx = tuple(fidx[i] for fidx in self_flat_indices)
                self_expanded[idx] = self_data_flat[i]

            for i in range(len(other_data_flat)):
                idx = tuple(fidx[i] for fidx in other_flat_indices)
                other_expanded[idx] = other_data_flat[i]

        return self_expanded, other_expanded, union_coords, all_dims

    def _operation(
        self, other: Union["Array", np.ndarray, "da.Array", int, float], op: Callable
    ) -> "Array":
        """Apply an operation between this array and another array or scalar."""
        if isinstance(other, (int, float, np.number)):
            # Simple operation with a scalar
            result_data = op(self.data, other)
            return Array(result_data, self.coords, self.dims, self.name)
        elif isinstance(other, np.ndarray) or (
            HAS_DASK and isinstance(other, da.Array)
        ):
            # Operation with a raw array - must have compatible shape
            if other.shape != self.data.shape:
                raise ValueError(
                    f"Incompatible shapes: {self.data.shape} vs {other.shape}"
                )
            result_data = op(self.data, other)
            return Array(result_data, self.coords, self.dims, self.name)
        elif isinstance(other, Array):
            # Operation with another Array - needs alignment
            self_aligned, other_aligned, union_coords, all_dims = self._align_with(
                other
            )
            result_data = op(self_aligned, other_aligned)
            return Array(result_data, union_coords, all_dims, self.name)
        else:
            raise TypeError(f"Unsupported type for operation: {type(other)}")

    def __add__(
        self, other: Union["Array", np.ndarray, "da.Array", int, float]
    ) -> "Array":
        """Add this array with another array or scalar."""
        return self._operation(other, lambda x, y: x + y)

    def __sub__(
        self, other: Union["Array", np.ndarray, "da.Array", int, float]
    ) -> "Array":
        """Subtract another array or scalar from this array."""
        return self._operation(other, lambda x, y: x - y)

    def __mul__(
        self, other: Union["Array", np.ndarray, "da.Array", int, float]
    ) -> "Array":
        """Multiply this array with another array or scalar."""
        return self._operation(other, lambda x, y: x * y)

    def __truediv__(
        self, other: Union["Array", np.ndarray, "da.Array", int, float]
    ) -> "Array":
        """Divide this array by another array or scalar."""
        return self._operation(other, lambda x, y: x / y)

    def __radd__(self, other: Union[np.ndarray, "da.Array", int, float]) -> "Array":
        """Add a scalar or array to this array."""
        return self._operation(other, lambda x, y: y + x)

    def __rsub__(self, other: Union[np.ndarray, "da.Array", int, float]) -> "Array":
        """Subtract this array from a scalar or array."""
        return self._operation(other, lambda x, y: y - x)

    def __rmul__(self, other: Union[np.ndarray, "da.Array", int, float]) -> "Array":
        """Multiply a scalar or array with this array."""
        return self._operation(other, lambda x, y: y * x)

    def __rtruediv__(self, other: Union[np.ndarray, "da.Array", int, float]) -> "Array":
        """Divide a scalar or array by this array."""
        return self._operation(other, lambda x, y: y / x)

    def __neg__(self) -> "Array":
        """Negate this array."""
        return Array(-self.data, self.coords, self.dims, self.name)

    def __pow__(self, other: Union[int, float]) -> "Array":
        """Raise this array to the power of another array or scalar."""
        if isinstance(other, (int, float)):
            return self._operation(other, lambda x, y: x**y)
        else:
            raise Exception("Not implemented")

    def sum(
        self, dim: Optional[Union[str, List[str]]] = None, keepdims: bool = False
    ) -> "Array":
        """
        Sum array elements along specified dimensions.

        Parameters
        ----------
        dim : Optional[Union[str, List[str]]], optional
            Dimension(s) to sum over. If None, sum over all dimensions.
        keepdims : bool, default=False
            If True, the summed dimensions are kept with size 1.

        Returns
        -------
        Array
            A new Array with summed data
        """
        if dim is None:
            # Sum over all dimensions
            result = np.sum(self.data) if not self.is_lazy else self.data.sum()
            if keepdims:
                result_data = (
                    np.array([[result]]) if not self.is_lazy else da.array([[result]])
                )
                result_coords = {dim: [0] for dim in self.dims}
                return Array(result_data, result_coords, self.dims, self.name)
            else:
                # Return a scalar - wrapped as 0-dimensional array
                return Array(
                    np.array(result) if not self.is_lazy else da.array(result),
                    {},
                    [],
                    self.name,
                )

        if isinstance(dim, str):
            dim = [dim]

        # Convert dimension names to axis indices
        axes = [self.dims.index(d) for d in dim]

        # Apply sum
        result_data = (
            np.sum(self.data, axis=tuple(axes), keepdims=keepdims)
            if not self.is_lazy
            else self.data.sum(axis=tuple(axes), keepdims=keepdims)
        )

        # Create new coordinates and dimensions
        if keepdims:
            new_coords = {
                d: np.array([0]) if d in dim else self.coords[d].copy()
                for d in self.dims
            }
            new_dims = self.dims.copy()
        else:
            new_coords = {d: self.coords[d].copy() for d in self.dims if d not in dim}
            new_dims = [d for d in self.dims if d not in dim]

        return Array(result_data, new_coords, new_dims, self.name)

    def to_dask(
        self, chunks: Optional[Union[str, Tuple[int, ...], Dict[str, int]]] = "auto"
    ) -> "Array":
        """
        Convert to a lazy Dask array.

        Parameters
        ----------
        chunks : Optional[Union[str, Tuple[int, ...], Dict[str, int]]], optional
            Chunk specification (default: "auto")

        Returns
        -------
        Array
            A new Array with lazy Dask array data
        """
        if not HAS_DASK:
            raise ImportError("Dask is required for this operation")

        if self.is_lazy:
            if chunks == "auto" or chunks is None:
                return self
            return Array(self.data.rechunk(chunks), self.coords, self.dims, self.name)

        # Convert NumPy to Dask
        if chunks == "auto":
            chunks = self._compute_chunks(self.data.shape, self.dims)

        dask_data = da.from_array(self.data, chunks=chunks)
        return Array(dask_data, self.coords, self.dims, self.name)

    def to_numpy(self) -> "Array":
        """
        Convert to an eager NumPy array.

        Returns
        -------
        Array
            A new Array with eager NumPy array data
        """
        if not self.is_lazy:
            return self

        return self.compute()

    def shrink(self, dims_coords: Dict[str, Union[List, np.ndarray]]) -> "Array":
        """
        Reduce the size of the array by selecting coordinates along specified dimensions.

        Parameters
        ----------
        dims_coords : Dict[str, Union[List, np.ndarray]]
            Dictionary mapping dimension names to the coordinates to keep.

        Returns
        -------
        Array
            A new array with reduced size.

        Examples
        --------
        >>> array = Array(np.arange(12).reshape(3, 4),
        ...               {"x": ["a", "b", "c"], "y": [10, 20, 30, 40]})
        >>> smaller = array.shrink({"x": ["a", "c"], "y": [10, 40]})
        >>> smaller.data.shape
        (2, 2)
        """
        if not dims_coords:
            return self

        # Make a copy of current coords for the result
        new_coords = {dim: coord.copy() for dim, coord in self.coords.items()}

        # Process each dimension and build new coordinates
        dim_indices = {}
        for dim, coords_to_keep in dims_coords.items():
            if dim not in self.dims:
                raise ValueError(f"Dimension '{dim}' not found in array dimensions")

            # Convert coordinates to keep to an array for consistent handling
            coords_to_keep = np.asarray(coords_to_keep)

            # Get the indices of the coordinates to keep
            dim_map = self.coordinate_map.get_index_mapping(dim)
            indices = np.array([dim_map.get(coord, -1) for coord in coords_to_keep])

            # Check if any coordinates weren't found
            if np.any(indices == -1):
                missing = [c for c, idx in zip(coords_to_keep, indices) if idx == -1]
                raise ValueError(
                    f"Coordinates {missing} not found in dimension '{dim}'"
                )

            # Sort indices to maintain original order
            indices.sort()

            # Update coordinates for this dimension
            new_coords[dim] = self.coords[dim][indices]

            # Store indices for this dimension
            dim_indices[dim] = indices

        # For each dimension that we want to subset, create a selection
        # We need to use a sequential approach rather than trying to do all dimensions at once
        result_data = self.data

        # First, make a copy of the data if using numpy (for dask this is handled differently)
        if not self.is_lazy:
            result_data = result_data.copy()

        # Apply selections for each dimension one at a time
        for dim_idx, dim in enumerate(self.dims):
            if dim in dim_indices:
                # We need to create a tuple of slices with exactly one array of indices
                selection = [slice(None)] * len(self.dims)
                selection[dim_idx] = dim_indices[dim]

                # Apply this selection
                if self.is_lazy and HAS_DASK:
                    result_data = result_data[tuple(selection)]
                else:
                    result_data = result_data[tuple(selection)]

        # Create and return the new array
        return Array(result_data, new_coords, self.dims, self.name)

    def __getitem__(self, key: Dict[str, Any]) -> "Array":
        """
        Get item or slice from the array using dimension names.

        This method provides a convenient wrapper around the shrink method,
        allowing for more intuitive dictionary-based indexing.

        Parameters
        ----------
        key : Dict[str, Any]
            Dictionary where keys are dimension names and values are coordinates to select.
            Values can be single items (str, int, float) or sequences of coordinates.

        Returns
        -------
        Array
            A new array with the selected data.

        Examples
        --------
        >>> arr = Array(np.ones((2, 3)), {"x": ["a", "b"], "y": [1, 2, 3]})
        >>> # Select by dimension name
        >>> arr[{"x": "a"}]  # Select just the "a" coordinate from x dimension
        >>> # Select multiple coordinates
        >>> arr[{"x": ["a", "b"], "y": [1, 3]}]  # Select specific coordinates
        """
        if not isinstance(key, dict):
            raise TypeError("Indexing only supports dictionary input")

        # Prepare a dictionary to pass to shrink
        # Convert single values (string, int, float) to lists
        shrink_dict = {}
        for dim, val in key.items():
            if dim not in self.dims:
                raise ValueError(f"Dimension '{dim}' not found in array dimensions")

            if isinstance(val, (str, int, float)):
                # Single value - convert to a list with one element
                shrink_dict[dim] = [val]
            elif isinstance(val, (list, tuple, np.ndarray)):
                # Already a sequence or other type
                shrink_dict[dim] = val
            else:
                raise TypeError(
                    f"Unsupported type for dimension '{dim}': {type(val).__name__}"
                )

        return self.shrink(shrink_dict)
