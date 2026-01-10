"""
Labeled N-dimensional arrays with outer-join alignment and zero-fill for missing values.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class Array:
    """
    Labeled N-dimensional array with automatic coordinate alignment.

    Binary operations between arrays use outer join semantics: the result
    contains the union of all coordinates, with missing values filled as zero.
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

    def __repr__(self) -> str:
        dims_str = ", ".join(f"{d}:{len(self.coords[d])}" for d in self.dims)
        return f"Array({self.shape}, [{dims_str}])"

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
                self_set = set(self_coords.tolist())
                new_from_other = [c for c in other_coords if c not in self_set]
                union_coords[dim] = np.concatenate([self_coords, new_from_other])
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

        slices = []
        for dim in all_dims:
            if dim in self.dims:
                union_coord = union_coords[dim]
                self_coord = self.coords[dim]
                coord_to_idx = {v: i for i, v in enumerate(union_coord)}
                indices = np.array([coord_to_idx[v] for v in self_coord])
                slices.append(indices)
            else:
                slices.append(np.arange(len(union_coords[dim])))

        idx_arrays = np.meshgrid(*slices, indexing="ij")
        expanded[tuple(idx_arrays)] = data_to_assign

    def _binary_op(self, other: Union["Array", int, float], op: Callable) -> "Array":
        """
        Apply a binary operation between this array and another array or scalar.
        """
        if isinstance(other, (int, float, np.number)):
            return Array(op(self.data, other), self.coords, self.dims, self.name)

        if isinstance(other, Array):
            self_exp, other_exp, union_coords, all_dims = self._align_with(other)
            result_data = op(self_exp, other_exp)
            return Array(result_data, union_coords, all_dims, self.name)

        raise TypeError(f"Unsupported type: {type(other)}")

    def __add__(self, other):
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self._binary_op(other, lambda a, b: b + a)

    def __sub__(self, other):
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._binary_op(other, lambda a, b: b - a)

    def __mul__(self, other):
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self._binary_op(other, lambda a, b: b * a)

    def __truediv__(self, other):
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

    def sel(
        self, indexers: Dict[str, Union[Any, List]]
    ) -> Union["Array", Any]:
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
            coord_list = coord.tolist()

            if isinstance(labels, (list, np.ndarray)):
                # Multiple labels: find indices and select
                missing = [lbl for lbl in labels if lbl not in coord_list]
                if missing:
                    raise ValueError(
                        f"Labels {missing} not found in dimension '{dim}'. "
                        f"Available: {coord_list[:10]}"
                        f"{'...' if len(coord_list) > 10 else ''}"
                    )
                indices = [coord_list.index(lbl) for lbl in labels]
                result_data = np.take(result_data, indices, axis=axis)
                result_coords[dim] = np.array(labels)
            else:
                # Single label: reduce this dimension
                if labels not in coord_list:
                    raise ValueError(
                        f"Label '{labels}' not found in dimension '{dim}'. "
                        f"Available: {coord_list[:10]}"
                        f"{'...' if len(coord_list) > 10 else ''}"
                    )
                idx = coord_list.index(labels)
                result_data = np.take(result_data, idx, axis=axis)
                del result_coords[dim]
                result_dims.remove(dim)

        if len(result_dims) == 0:
            return result_data.item() if result_data.ndim == 0 else result_data

        return Array(result_data, result_coords, result_dims, self.name)


    def isel(
        self, indexers: Dict[str, Union[int, List[int]]]
    ) -> Union["Array", Any]:
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
                f"Dimension '{dim}' already exists. "
                f"Current dimensions: {self.dims}"
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
