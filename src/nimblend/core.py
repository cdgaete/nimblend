"""
Labeled N-dimensional arrays with outer-join alignment and zero-fill for missing values.
"""

from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


def _normalize_label(label: Any, coord: np.ndarray, is_datetime: bool = None) -> Any:
    """
    Normalize a label to match the dtype of the coordinate array.

    Handles datetime conversion: strings and python dates are converted
    to numpy datetime64 when the coordinate has a datetime dtype.

    Parameters
    ----------
    label : Any
        The label to normalize.
    coord : np.ndarray
        The coordinate array to match dtype against.
    is_datetime : bool, optional
        If provided, skip the issubdtype check (optimization for batch calls).
    """
    if is_datetime is None:
        is_datetime = np.issubdtype(coord.dtype, np.datetime64)

    if not is_datetime:
        return label

    # Coordinate is datetime64 - normalize the label
    if isinstance(label, np.datetime64):
        return label.astype(coord.dtype)
    elif isinstance(label, str):
        return np.datetime64(label).astype(coord.dtype)
    elif isinstance(label, (date, datetime)):
        return np.datetime64(label).astype(coord.dtype)
    return label


def _normalize_labels_batch(labels: list, coord: np.ndarray) -> list:
    """
    Normalize a batch of labels to match the coordinate dtype.

    This is more efficient than calling _normalize_label repeatedly
    because it checks the dtype once.
    """
    is_datetime = np.issubdtype(coord.dtype, np.datetime64)
    if not is_datetime:
        return labels

    # Coordinate is datetime64 - normalize all labels
    result = []
    target_dtype = coord.dtype
    for label in labels:
        if isinstance(label, np.datetime64):
            result.append(label.astype(target_dtype))
        elif isinstance(label, str):
            result.append(np.datetime64(label).astype(target_dtype))
        elif isinstance(label, (date, datetime)):
            result.append(np.datetime64(label).astype(target_dtype))
        else:
            result.append(label)
    return result


class Array:
    """
    Labeled N-dimensional array with automatic coordinate alignment.

    Binary operations between arrays use outer join semantics: the result
    contains the union of all coordinates, with missing values filled as zero.

    Parameters
    ----------
    data : array-like
        The underlying array data.
    coords : dict
        Mapping of dimension names to coordinate values.
    dims : list of str, optional
        Dimension names in axis order. Defaults to coords key order.
    name : str, optional
        Name for the array.

    Attributes
    ----------
    data : np.ndarray
        The underlying NumPy array.
    coords : dict
        Mapping of dimension names to coordinate arrays.
    dims : list
        List of dimension names.
    shape : tuple
        Shape of the data array.
    name : str or None
        Optional name for the array.

    Examples
    --------
    >>> import numpy as np
    >>> from nimblend import Array

    Create a 2D array:

    >>> data = np.array([[1, 2], [3, 4]])
    >>> arr = Array(data, {'x': ['a', 'b'], 'y': [0, 1]})

    Arithmetic with automatic alignment:

    >>> arr1 = Array(np.array([1, 2]), {'x': ['a', 'b']})
    >>> arr2 = Array(np.array([10, 20]), {'x': ['b', 'c']})
    >>> result = arr1 + arr2  # outer join: x=['a', 'b', 'c']

    Selection by label:

    >>> arr.sel({'x': 'a'})  # returns 1D array
    >>> arr.sel({'x': 'a', 'y': 0})  # returns scalar
    """

    def __init__(
        self,
        data: np.ndarray,
        coords: Dict[str, np.ndarray],
        dims: Optional[List[str]] = None,
        name: Optional[str] = None,
    ):
        """
        Create a labeled array.

        Parameters
        ----------
        data : np.ndarray
            The underlying array data.
        coords : dict
            Mapping of dimension names to coordinate values.
        dims : list of str, optional
            Dimension names in axis order. Defaults to coords key order.
        name : str, optional
            Name for the array.
        """
        if dims is None:
            dims = list(coords.keys())

        self.dims = dims
        self.coords = {dim: np.asarray(coords[dim]) for dim in dims}
        self.data = np.asarray(data)
        self.name = name
        self._validate()

    def _validate(self):
        """Check that dimensions and coordinates match data shape."""
        if len(self.dims) != self.data.ndim:
            raise ValueError(
                f"Dimension mismatch: got {len(self.dims)} dimension names "
                f"{self.dims} but data has {self.data.ndim} dimensions "
                f"with shape {self.data.shape}"
            )
        for i, dim in enumerate(self.dims):
            if dim not in self.coords:
                raise ValueError(
                    f"Dimension '{dim}' not found in coords. "
                    f"Available: {list(self.coords.keys())}"
                )
            if len(self.coords[dim]) != self.data.shape[i]:
                raise ValueError(
                    f"Coordinate length mismatch for dimension '{dim}': "
                    f"coords has {len(self.coords[dim])} values but "
                    f"data has size {self.data.shape[i]} along axis {i}"
                )

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the underlying data array."""
        return self.data.shape

    @property
    def values(self) -> np.ndarray:
        """Alias for data. Returns the underlying NumPy array."""
        return self.data

    def __repr__(self) -> str:
        dims_str = ", ".join(f"{d}:{len(self.coords[d])}" for d in self.dims)
        return f"Array({self.shape}, [{dims_str}])"

    def _coords_match(self, other: "Array") -> bool:
        """Check if coordinates are identical (fast path optimization)."""
        if self.dims != other.dims:
            return False
        for dim in self.dims:
            sc = self.coords[dim]
            oc = other.coords[dim]
            if len(sc) != len(oc):
                return False
            if not np.array_equal(sc, oc):
                return False
        return True

    def _align_with(
        self, other: "Array"
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], List[str]]:
        """
        Align two arrays to a common shape using outer join on coordinates.

        Returns
        -------
        self_expanded : np.ndarray
            Self's data expanded to the union shape, zero-filled where missing.
        other_expanded : np.ndarray
            Other's data expanded to the union shape, zero-filled where missing.
        union_coords : dict
            The union coordinates for each dimension.
        all_dims : list
            Ordered list of all dimensions.
        """
        all_dims = list(self.dims)
        for dim in other.dims:
            if dim not in all_dims:
                all_dims.append(dim)

        union_coords = {}
        for dim in all_dims:
            if dim in self.coords and dim in other.coords:
                # Preserve order: self's coords first, then new coords from other
                self_coords = self.coords[dim]
                other_coords = other.coords[dim]
                # Use int64 view for datetime64 (faster hashing)
                if np.issubdtype(self_coords.dtype, np.datetime64):
                    self_set = set(self_coords.view("int64"))
                    new_mask = ~np.isin(
                        other_coords.view("int64"), self_coords.view("int64")
                    )
                    new_from_other = other_coords[new_mask]
                else:
                    self_set = set(self_coords)
                    new_from_other = [c for c in other_coords if c not in self_set]
                if len(new_from_other) > 0:
                    union_coords[dim] = np.concatenate([self_coords, new_from_other])
                else:
                    union_coords[dim] = self_coords.copy()
            elif dim in self.coords:
                union_coords[dim] = self.coords[dim].copy()
            else:
                union_coords[dim] = other.coords[dim].copy()

        result_shape = tuple(len(union_coords[dim]) for dim in all_dims)

        self_expanded = np.zeros(result_shape, dtype=self.data.dtype)
        other_expanded = np.zeros(result_shape, dtype=other.data.dtype)

        self._fill_expanded(self_expanded, all_dims, union_coords)
        other._fill_expanded(other_expanded, all_dims, union_coords)

        return self_expanded, other_expanded, union_coords, all_dims

    def _fill_expanded(
        self,
        expanded: np.ndarray,
        all_dims: List[str],
        union_coords: Dict[str, np.ndarray],
    ):
        """
        Place this array's values into an expanded array at the correct positions.

        Handles dimension reordering and coordinate mapping to align self's data
        with the target dimension order and union coordinates.
        """
        data_to_assign = self.data

        self_dims_in_all_order = [d for d in all_dims if d in self.dims]

        if self_dims_in_all_order != list(self.dims):
            perm = [self.dims.index(d) for d in self_dims_in_all_order]
            data_to_assign = np.transpose(data_to_assign, perm)

        for i, dim in enumerate(all_dims):
            if dim not in self.dims:
                data_to_assign = np.expand_dims(data_to_assign, axis=i)

        # Check if we can use fast slice-based indexing (contiguous indices)
        slices_or_indices = []
        can_use_slices = True
        for dim in all_dims:
            if dim in self.dims:
                union_coord = union_coords[dim]
                self_coord = self.coords[dim]

                # Use searchsorted for large sorted numeric/datetime arrays
                is_numeric = np.issubdtype(union_coord.dtype, np.number)
                is_datetime = np.issubdtype(union_coord.dtype, np.datetime64)

                if (is_numeric or is_datetime) and len(union_coord) > 100:
                    # Check if sorted (O(n) but enables O(log n) lookups)
                    is_sorted = np.all(union_coord[:-1] <= union_coord[1:])
                    if is_sorted:
                        indices = np.searchsorted(union_coord, self_coord)
                    elif is_datetime:
                        # datetime64: use int64 view for faster hashing
                        u_view = union_coord.view("int64")
                        s_view = self_coord.view("int64")
                        coord_to_idx = {v: i for i, v in enumerate(u_view)}
                        indices = np.array([coord_to_idx[v] for v in s_view])
                    else:
                        coord_to_idx = {v: i for i, v in enumerate(union_coord)}
                        indices = np.array([coord_to_idx[v] for v in self_coord])
                elif is_datetime:
                    # Small datetime: use int64 view
                    u_view = union_coord.view("int64")
                    s_view = self_coord.view("int64")
                    coord_to_idx = {v: i for i, v in enumerate(u_view)}
                    indices = np.array([coord_to_idx[v] for v in s_view])
                else:
                    # String or small numeric: use dict
                    coord_to_idx = {v: i for i, v in enumerate(union_coord)}
                    indices = np.array([coord_to_idx[v] for v in self_coord])

                # Check if indices are contiguous (can use slice)
                if len(indices) > 0:
                    start_idx = indices[0]
                    expected = np.arange(start_idx, start_idx + len(indices))
                    if np.array_equal(indices, expected):
                        # Contiguous range - use slice
                        end_idx = start_idx + len(indices)
                        slices_or_indices.append(slice(start_idx, end_idx))
                    else:
                        can_use_slices = False
                        slices_or_indices.append(indices)
                else:
                    slices_or_indices.append(slice(0, 0))
            else:
                slices_or_indices.append(slice(None))  # Full slice for new dims

        if can_use_slices:
            # Fast path: use slice indexing (no copy, no meshgrid)
            expanded[tuple(slices_or_indices)] = data_to_assign
        else:
            # Slow path: advanced indexing
            # Try Rust acceleration for float64 arrays
            from nimblend._accel import HAS_RUST

            use_rust = HAS_RUST and expanded.dtype == np.float64

            if use_rust:
                # Convert slices to index arrays for Rust
                idx_arrays = []
                for i, s in enumerate(slices_or_indices):
                    if isinstance(s, slice):
                        dim = all_dims[i]
                        start = s.start or 0
                        stop = s.stop if s.stop is not None else len(union_coords[dim])
                        idx_arrays.append(np.arange(start, stop, dtype=np.int64))
                    else:
                        idx_arrays.append(s.astype(np.int64))

                if expanded.ndim == 2:
                    from nimblend._accel import fill_expanded_2d

                    fill_expanded_2d(
                        expanded,
                        data_to_assign.astype(np.float64),
                        idx_arrays[0],
                        idx_arrays[1],
                    )
                else:
                    from nimblend._accel import fill_expanded_nd

                    fill_expanded_nd(
                        expanded,
                        data_to_assign.astype(np.float64),
                        idx_arrays,
                    )
            else:
                # Pure NumPy fallback with meshgrid
                idx_lists = []
                for i, s in enumerate(slices_or_indices):
                    if isinstance(s, slice):
                        dim = all_dims[i]
                        if s.start is None and s.stop is None:
                            idx_lists.append(np.arange(len(union_coords[dim])))
                        else:
                            stop = s.stop or len(union_coords[dim])
                            idx_lists.append(np.arange(s.start or 0, stop))
                    else:
                        idx_lists.append(s)
                idx_arrays = np.meshgrid(*idx_lists, indexing="ij")
                expanded[tuple(idx_arrays)] = data_to_assign

    def _binary_op(self, other: Union["Array", int, float], op: Callable) -> "Array":
        """
        Apply a binary operation between this array and another array or scalar.
        """
        if isinstance(other, (int, float, np.number)):
            return Array(op(self.data, other), self.coords, self.dims, self.name)

        if isinstance(other, Array):
            # Fast path: identical coordinates - skip alignment
            if self._coords_match(other):
                result_data = op(self.data, other.data)
                return Array(result_data, self.coords, self.dims, self.name)

            # Slow path: need alignment
            self_exp, other_exp, union_coords, all_dims = self._align_with(other)
            result_data = op(self_exp, other_exp)
            return Array(result_data, union_coords, all_dims, self.name)

        raise TypeError(f"Unsupported type: {type(other)}")

    def _fast_aligned_binop_2d(
        self, other: "Array", op_name: str
    ) -> Optional["Array"]:
        """
        Fast path for 2D arrays where only one dimension is misaligned.

        Uses Rust accelerated aligned_binop_2d which combines fill + operation
        in a single pass, avoiding intermediate array allocation.

        Returns None if conditions not met (falls back to generic path).
        """
        # Check conditions
        if self.data.ndim != 2 or other.data.ndim != 2:
            return None
        if self.dims != other.dims:
            return None
        if self.data.dtype != np.float64 or other.data.dtype != np.float64:
            return None

        # Check: exactly one dimension misaligned, other dimension identical
        dim0, dim1 = self.dims
        coords_match_0 = np.array_equal(self.coords[dim0], other.coords[dim0])
        coords_match_1 = np.array_equal(self.coords[dim1], other.coords[dim1])

        if coords_match_0 and coords_match_1:
            return None  # Fully aligned - use even faster path
        if not coords_match_0 and not coords_match_1:
            return None  # Both misaligned - need full alignment

        # One dimension misaligned
        from nimblend._accel import HAS_RUST

        if not HAS_RUST:
            return None

        from nimblend._accel import aligned_binop_2d

        if not coords_match_0:
            # Dim 0 (rows) misaligned, dim 1 (cols) aligned
            self_c = self.coords[dim0]
            other_c = other.coords[dim0]

            # Compute union coords
            if np.issubdtype(self_c.dtype, np.datetime64):
                self_set = set(self_c.view("int64"))
                new_mask = ~np.isin(other_c.view("int64"), self_c.view("int64"))
            else:
                self_set = set(self_c)
                new_mask = np.array([c not in self_set for c in other_c])

            new_from_other = other_c[new_mask]
            if len(new_from_other) > 0:
                union_c0 = np.concatenate([self_c, new_from_other])
            else:
                union_c0 = self_c.copy()

            # Build index mappings
            if np.issubdtype(union_c0.dtype, np.datetime64):
                u_view = union_c0.view("int64")
                c2i = {v: i for i, v in enumerate(u_view)}
                idx1 = np.array(
                    [c2i[v] for v in self_c.view("int64")], dtype=np.int64
                )
                idx2 = np.array(
                    [c2i[v] for v in other_c.view("int64")], dtype=np.int64
                )
            else:
                c2i = {v: i for i, v in enumerate(union_c0)}
                idx1 = np.array([c2i[v] for v in self_c], dtype=np.int64)
                idx2 = np.array([c2i[v] for v in other_c], dtype=np.int64)

            result = np.zeros((len(union_c0), self.data.shape[1]), dtype=np.float64)
            aligned_binop_2d(result, self.data, idx1, other.data, idx2, op_name)

            union_coords = {dim0: union_c0, dim1: self.coords[dim1].copy()}
            return Array(result, union_coords, self.dims, self.name)

        else:
            # Dim 1 (cols) misaligned - transpose, compute, transpose back
            # For now, fall back to generic path
            return None

    def __add__(self, other):
        if isinstance(other, Array):
            result = self._fast_aligned_binop_2d(other, "add")
            if result is not None:
                return result
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self._binary_op(other, lambda a, b: b + a)

    def __sub__(self, other):
        if isinstance(other, Array):
            result = self._fast_aligned_binop_2d(other, "sub")
            if result is not None:
                return result
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._binary_op(other, lambda a, b: b - a)

    def __mul__(self, other):
        if isinstance(other, Array):
            result = self._fast_aligned_binop_2d(other, "mul")
            if result is not None:
                return result
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self._binary_op(other, lambda a, b: b * a)

    def __truediv__(self, other):
        if isinstance(other, Array):
            result = self._fast_aligned_binop_2d(other, "div")
            if result is not None:
                return result
        return self._binary_op(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return self._binary_op(other, lambda a, b: b / a)

    def __neg__(self):
        return Array(-self.data, self.coords, self.dims, self.name)

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            return Array(self.data**other, self.coords, self.dims, self.name)
        raise TypeError("Power only supported with scalars")

    # Comparison operators - return boolean Arrays
    def __eq__(self, other) -> "Array":
        return self._binary_op(other, lambda a, b: a == b)

    def __ne__(self, other) -> "Array":
        return self._binary_op(other, lambda a, b: a != b)

    def __lt__(self, other) -> "Array":
        return self._binary_op(other, lambda a, b: a < b)

    def __le__(self, other) -> "Array":
        return self._binary_op(other, lambda a, b: a <= b)

    def __gt__(self, other) -> "Array":
        return self._binary_op(other, lambda a, b: a > b)

    def __ge__(self, other) -> "Array":
        return self._binary_op(other, lambda a, b: a >= b)

    def _reduce(
        self,
        func: Callable,
        dim: Optional[Union[str, List[str]]] = None,
    ) -> Union["Array", Any]:
        """Apply a reduction function over specified dimensions."""
        if dim is None:
            return func(self.data)

        if isinstance(dim, str):
            dim = [dim]

        axes = tuple(self.dims.index(d) for d in dim)
        result_data = func(self.data, axis=axes)

        new_dims = [d for d in self.dims if d not in dim]
        new_coords = {d: self.coords[d] for d in new_dims}

        # Return scalar if no dimensions remain
        if len(new_dims) == 0:
            return result_data.item()

        return Array(result_data, new_coords, new_dims, self.name)

    def sum(self, dim: Optional[Union[str, List[str]]] = None):
        """Sum over dimension(s). If None, sum all."""
        return self._reduce(np.sum, dim)

    def mean(self, dim: Optional[Union[str, List[str]]] = None):
        """Mean over dimension(s). If None, mean of all."""
        return self._reduce(np.mean, dim)

    def min(self, dim: Optional[Union[str, List[str]]] = None):
        """Minimum over dimension(s). If None, min of all."""
        return self._reduce(np.min, dim)

    def max(self, dim: Optional[Union[str, List[str]]] = None):
        """Maximum over dimension(s). If None, max of all."""
        return self._reduce(np.max, dim)

    def std(self, dim: Optional[Union[str, List[str]]] = None):
        """Standard deviation over dimension(s). If None, std of all."""
        return self._reduce(np.std, dim)

    def prod(self, dim: Optional[Union[str, List[str]]] = None):
        """Product over dimension(s). If None, product of all."""
        return self._reduce(np.prod, dim)

    def sel(self, indexers: Dict[str, Union[Any, List]]) -> Union["Array", Any]:
        """
        Select data by coordinate labels.

        Parameters
        ----------
        indexers : dict
            Mapping of dimension names to coordinate values or lists of values.

        Returns
        -------
        Array or scalar
            Selected subset. Returns scalar if all dimensions are indexed
            with single values.

        Examples
        --------
        >>> arr.sel({'x': 'a'})           # Select single value along x
        >>> arr.sel({'x': ['a', 'b']})    # Select multiple values along x
        >>> arr.sel({'x': 'a', 'y': 0})   # Select along multiple dimensions
        """
        result_data = self.data
        result_coords = dict(self.coords)
        result_dims = list(self.dims)

        for dim, labels in indexers.items():
            if dim not in result_dims:
                raise KeyError(
                    f"Dimension '{dim}' not found. "
                    f"Available dimensions: {result_dims}"
                )

            axis = result_dims.index(dim)
            coord = result_coords[dim]

            if isinstance(labels, (list, np.ndarray)):
                # Multiple labels: build lookup dict (amortized over many labels)
                coord_to_idx = {v: i for i, v in enumerate(coord)}
                # Normalize labels to match coord dtype (handles datetime)
                labels_list = labels if isinstance(labels, list) else labels.tolist()
                normalized = _normalize_labels_batch(labels_list, coord)
                missing = [lbl for lbl, norm in zip(labels_list, normalized)
                           if norm not in coord_to_idx]
                if missing:
                    available = coord.tolist()[:10]
                    raise ValueError(
                        f"Labels {missing} not found in dimension '{dim}'. "
                        f"Available: {available}"
                        f"{'...' if len(coord) > 10 else ''}"
                    )
                indices = [coord_to_idx[norm] for norm in normalized]
                result_data = np.take(result_data, indices, axis=axis)
                result_coords[dim] = np.array([coord[i] for i in indices])
            else:
                # Single label: use searchsorted for sorted numeric/datetime arrays
                normalized = _normalize_label(labels, coord)
                is_numeric = np.issubdtype(coord.dtype, np.number)
                is_datetime = np.issubdtype(coord.dtype, np.datetime64)

                idx = None
                if (is_numeric or is_datetime) and len(coord) > 100:
                    # Check if sorted (O(n) but amortized benefit for large arrays)
                    if np.all(coord[:-1] <= coord[1:]):
                        # Use binary search O(log n)
                        pos = np.searchsorted(coord, normalized)
                        if pos < len(coord) and coord[pos] == normalized:
                            idx = pos

                if idx is None:
                    # Linear search fallback - avoid full dict for single lookup
                    for i, v in enumerate(coord):
                        if v == normalized:
                            idx = i
                            break

                if idx is None:
                    available = coord.tolist()[:10]
                    raise ValueError(
                        f"Label '{labels}' not found in dimension '{dim}'. "
                        f"Available: {available}"
                        f"{'...' if len(coord) > 10 else ''}"
                    )

                result_data = np.take(result_data, idx, axis=axis)
                del result_coords[dim]
                result_dims.remove(dim)

        if len(result_dims) == 0:
            return result_data.item() if result_data.ndim == 0 else result_data

        return Array(result_data, result_coords, result_dims, self.name)

    def isel(self, indexers: Dict[str, Union[int, List[int]]]) -> Union["Array", Any]:
        """
        Select data by integer index positions.

        Parameters
        ----------
        indexers : dict
            Mapping of dimension names to integer indices or lists of indices.

        Returns
        -------
        Array or scalar
            Selected subset. Returns scalar if all dimensions are indexed
            with single values.

        Examples
        --------
        >>> arr.isel({'x': 0})            # Select first element along x
        >>> arr.isel({'x': [0, 2]})       # Select first and third along x
        >>> arr.isel({'x': 0, 'y': -1})   # Negative indexing supported
        """
        result_data = self.data
        result_coords = dict(self.coords)
        result_dims = list(self.dims)

        for dim, indices in indexers.items():
            if dim not in result_dims:
                raise KeyError(
                    f"Dimension '{dim}' not found. "
                    f"Available dimensions: {result_dims}"
                )

            axis = result_dims.index(dim)
            coord = result_coords[dim]
            dim_size = len(coord)

            if isinstance(indices, (list, np.ndarray)):
                # Multiple indices: select and keep dimension
                for idx in indices:
                    if idx >= dim_size or idx < -dim_size:
                        raise IndexError(
                            f"Index {idx} out of bounds for dimension '{dim}' "
                            f"with size {dim_size}"
                        )
                result_data = np.take(result_data, indices, axis=axis)
                result_coords[dim] = coord[indices]
            else:
                # Single index: reduce this dimension
                if indices >= dim_size or indices < -dim_size:
                    raise IndexError(
                        f"Index {indices} out of bounds for dimension '{dim}' "
                        f"with size {dim_size}"
                    )
                result_data = np.take(result_data, indices, axis=axis)
                del result_coords[dim]
                result_dims.remove(dim)

        if len(result_dims) == 0:
            return result_data.item() if result_data.ndim == 0 else result_data

        return Array(result_data, result_coords, result_dims, self.name)

    def transpose(self, *dims: str) -> "Array":
        """
        Reorder dimensions.

        Parameters
        ----------
        *dims : str
            New dimension order. Must include all dimensions.

        Returns
        -------
        Array
            Array with reordered dimensions.

        Examples
        --------
        >>> arr.transpose('y', 'x')  # Swap x and y
        >>> arr.T  # Reverse all dimensions
        """
        if not dims:
            # Reverse order like numpy .T
            dims = tuple(reversed(self.dims))

        if set(dims) != set(self.dims):
            missing = set(self.dims) - set(dims)
            extra = set(dims) - set(self.dims)
            msg = "Dimension mismatch in transpose: "
            if missing:
                msg += f"missing {missing}"
            if extra:
                msg += f"{', ' if missing else ''}unknown {extra}"
            msg += f". Available: {self.dims}"
            raise ValueError(msg)

        axes = [self.dims.index(d) for d in dims]
        new_data = np.transpose(self.data, axes)
        new_coords = {d: self.coords[d] for d in dims}

        return Array(new_data, new_coords, list(dims), self.name)

    @property
    def T(self) -> "Array":
        """Transpose: reverse dimension order."""
        return self.transpose()

    def squeeze(self, dim: Optional[str] = None) -> "Array":
        """
        Remove dimensions of size 1.

        Parameters
        ----------
        dim : str, optional
            Specific dimension to squeeze. If None, squeeze all size-1 dims.

        Returns
        -------
        Array
            Array with size-1 dimensions removed.
        """
        if dim is not None:
            if dim not in self.dims:
                raise KeyError(
                    f"Dimension '{dim}' not found. "
                    f"Available dimensions: {self.dims}"
                )
            if len(self.coords[dim]) != 1:
                raise ValueError(
                    f"Cannot squeeze dimension '{dim}': size is "
                    f"{len(self.coords[dim])}, must be 1"
                )
            axis = self.dims.index(dim)
            new_data = np.squeeze(self.data, axis=axis)
            new_dims = [d for d in self.dims if d != dim]
            new_coords = {d: self.coords[d] for d in new_dims}
            return Array(new_data, new_coords, new_dims, self.name)

        # Squeeze all size-1 dimensions
        new_dims = []
        new_coords = {}
        axes_to_squeeze = []

        for i, dim in enumerate(self.dims):
            if len(self.coords[dim]) == 1:
                axes_to_squeeze.append(i)
            else:
                new_dims.append(dim)
                new_coords[dim] = self.coords[dim]

        new_data = self.data
        for axis in reversed(axes_to_squeeze):
            new_data = np.squeeze(new_data, axis=axis)

        return Array(new_data, new_coords, new_dims, self.name)

    def expand_dims(self, dim: str, coord: Any = None) -> "Array":
        """
        Add a new dimension of size 1.

        Parameters
        ----------
        dim : str
            Name for the new dimension.
        coord : Any, optional
            Coordinate value for the new dimension. Defaults to 0.

        Returns
        -------
        Array
            Array with new dimension added at the front.
        """
        if dim in self.dims:
            raise ValueError(
                f"Dimension '{dim}' already exists. " f"Current dimensions: {self.dims}"
            )

        if coord is None:
            coord = 0

        new_data = np.expand_dims(self.data, axis=0)
        new_dims = [dim] + list(self.dims)
        new_coords = {dim: np.array([coord])}
        new_coords.update(self.coords)

        return Array(new_data, new_coords, new_dims, self.name)

    def rename(self, name_map: Dict[str, str]) -> "Array":
        """
        Rename dimensions.

        Parameters
        ----------
        name_map : dict
            Mapping from old names to new names.

        Returns
        -------
        Array
            Array with renamed dimensions.
        """
        new_dims = [name_map.get(d, d) for d in self.dims]
        new_coords = {name_map.get(d, d): self.coords[d] for d in self.dims}
        return Array(self.data.copy(), new_coords, new_dims, self.name)

    def copy(self) -> "Array":
        """
        Create a deep copy.

        Returns
        -------
        Array
            Copy with new data and coords arrays.
        """
        new_coords = {d: self.coords[d].copy() for d in self.dims}
        return Array(self.data.copy(), new_coords, list(self.dims), self.name)

    def astype(self, dtype) -> "Array":
        """
        Cast data to specified dtype.

        Parameters
        ----------
        dtype : numpy dtype
            Target data type.

        Returns
        -------
        Array
            Array with converted data type.
        """
        return Array(self.data.astype(dtype), self.coords, self.dims, self.name)

    def __getitem__(self, key) -> Union["Array", np.ndarray, Any]:
        """
        Index into the array.

        Supports three forms of indexing:

        1. Dict-based selection (alias for sel):
           arr[{'x': 'a'}] is equivalent to arr.sel({'x': 'a'})

        2. Integer/slice indexing on first dimension:
           arr[0] selects first element along first dim
           arr[0:2] slices first dimension

        3. Dimension name (returns coordinate array):
           arr['x'] returns the coordinate values for dimension 'x'

        Parameters
        ----------
        key : dict, int, slice, or str
            Indexer for the array.

        Returns
        -------
        Array, np.ndarray, or scalar
            Selected data. Dict and int/slice return Array or scalar.
            String returns coordinate array.
        """
        # Dict-based selection: delegate to sel
        if isinstance(key, dict):
            return self.sel(key)

        # String: return coordinate array for that dimension
        if isinstance(key, str):
            if key not in self.dims:
                raise KeyError(
                    f"Dimension '{key}' not found. "
                    f"Available dimensions: {self.dims}"
                )
            return self.coords[key]

        # Integer or slice: index first dimension
        if isinstance(key, (int, slice)):
            first_dim = self.dims[0]

            if isinstance(key, int):
                # Use isel for integer indexing
                return self.isel({first_dim: key})
            else:
                # Slice: keep dimension
                indices = range(*key.indices(len(self.coords[first_dim])))
                return self.isel({first_dim: list(indices)})

        raise TypeError(f"Invalid index type: {type(key).__name__}")

    def where(
        self, cond: Union["Array", np.ndarray], other: Union["Array", int, float] = 0
    ) -> "Array":
        """
        Replace values where condition is False.

        Parameters
        ----------
        cond : Array or np.ndarray
            Boolean condition. Where True, keep original values.
            Where False, replace with `other`.
        other : Array, int, or float, optional
            Replacement value(s). Default is 0.

        Returns
        -------
        Array
            Array with replaced values.

        Examples
        --------
        >>> arr.where(arr > 0, 0)      # Replace negatives with 0
        >>> arr.where(arr < 100, 100)  # Cap values at 100
        """
        # Extract data from Array conditions
        if isinstance(cond, Array):
            cond_data = cond.data
        else:
            cond_data = np.asarray(cond)

        # Extract data from Array other
        if isinstance(other, Array):
            other_data = other.data
        else:
            other_data = other

        result_data = np.where(cond_data, self.data, other_data)
        return Array(result_data, self.coords, self.dims, self.name)

    def clip(
        self,
        min: Optional[Union[int, float]] = None,
        max: Optional[Union[int, float]] = None,
    ) -> "Array":
        """
        Bound values to a range.

        Parameters
        ----------
        min : int or float, optional
            Minimum value. Values below this are set to min.
        max : int or float, optional
            Maximum value. Values above this are set to max.

        Returns
        -------
        Array
            Array with clipped values.

        Examples
        --------
        >>> arr.clip(0, 100)    # Clamp to [0, 100]
        >>> arr.clip(min=0)     # Floor at 0
        >>> arr.clip(max=100)   # Cap at 100
        """
        result_data = np.clip(self.data, min, max)
        return Array(result_data, self.coords, self.dims, self.name)

    def equals(self, other: "Array") -> bool:
        """
        Check if two arrays are identical.

        Arrays are equal if they have the same dimensions, coordinates,
        and data values. Names are ignored.

        Parameters
        ----------
        other : Array
            Array to compare against.

        Returns
        -------
        bool
            True if arrays are identical.

        Examples
        --------
        >>> arr1.equals(arr2)
        >>> arr1.equals(arr1.copy())  # True
        """
        if not isinstance(other, Array):
            return False

        if self.dims != other.dims:
            return False

        for dim in self.dims:
            if not np.array_equal(self.coords[dim], other.coords[dim]):
                return False

        return np.array_equal(self.data, other.data)

    def broadcast_like(self, other: "Array") -> "Array":
        """
        Expand array to match another array's shape.

        Uses outer-join alignment to broadcast self to match other's
        dimensions. New dimensions are added with values broadcast
        along them.

        Parameters
        ----------
        other : Array
            Target array whose shape to match.

        Returns
        -------
        Array
            Array broadcast to match other's dimensions and coordinates.

        Examples
        --------
        >>> small.broadcast_like(large)
        """
        # Use alignment machinery - add self to zeros with other's shape
        zeros = Array(
            np.zeros(other.shape, dtype=self.data.dtype),
            other.coords,
            other.dims,
            self.name,
        )
        return self + zeros

    def fillna(self, value: Union[int, float]) -> "Array":
        """
        Replace NaN values with a specified value.

        Parameters
        ----------
        value : int or float
            Value to replace NaN with.

        Returns
        -------
        Array
            Array with NaN values replaced.

        Examples
        --------
        >>> arr.fillna(0)  # Replace NaN with 0
        """
        result_data = np.where(np.isnan(self.data), value, self.data)
        return Array(result_data, self.coords, self.dims, self.name)

    def dropna(self, dim: str) -> "Array":
        """
        Remove coordinates with NaN values along a dimension.

        Parameters
        ----------
        dim : str
            Dimension along which to drop coordinates containing NaN.

        Returns
        -------
        Array
            Array with NaN-containing slices removed.

        Examples
        --------
        >>> arr.dropna('x')  # Drop x coords where any value is NaN
        """
        if dim not in self.dims:
            raise KeyError(
                f"Dimension '{dim}' not found. Available dimensions: {self.dims}"
            )

        axis = self.dims.index(dim)
        # Find which indices along dim have any NaN
        # Move target axis to first, reshape to 2D, check for NaN in each slice
        other_axes = tuple(i for i in range(self.data.ndim) if i != axis)
        nan_mask = np.any(np.isnan(self.data), axis=other_axes)
        keep_mask = ~nan_mask

        # Select only non-NaN indices
        keep_indices = np.where(keep_mask)[0]
        result_data = np.take(self.data, keep_indices, axis=axis)
        new_coords = {d: c for d, c in self.coords.items()}
        new_coords[dim] = self.coords[dim][keep_indices]

        return Array(result_data, new_coords, self.dims, self.name)
