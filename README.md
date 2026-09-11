# nimblend

Labeled sparse N-dimensional arrays for Python.

The documentation site is at <https://cdgaete.github.io/nimopt/nimblend/>.

`nimblend` stores an N-dimensional array as its entries, not as a grid, and identifies every position by a label, not by an offset. Two arrays combine by aligning their labels, not by matching their shapes. The dimensions of an operand may be in another order, may be a subset of the other operand's dimensions, or may overlap them partially. The frame of the result follows from the dimension names alone. The package has one dependency, `numpy`, and its vocabulary is dimensions, labels, entries and alignment.

Every array declares what an absent coordinate means. Under `"empty"`, an absent coordinate contributes nothing. Under `"unknown"`, it was not modeled. Every operator and reduction follows the declaration. A stored `0.0` is distinct from an absent coordinate under both.

## Installation

```bash
pip install nimblend
```

For work on the package, from a checkout:

```bash
pip install -e ".[dev]"  # editable, with the test and lint tooling
```

The package requires Python 3.12, 3.13 or 3.14, and `numpy >= 2.3`.

## Quick start

```python
import numpy as np
import nimblend as nb

years = nb.StoredCoord(np.array([2030, 2040, 2050]))
regions = nb.StoredCoord(np.array(["DE", "FR"]))
coords = {"year": years, "region": regions}

demand = nb.from_long(
    ("year", "region"),
    coords,
    {"year": np.array([2030, 2040, 2040]), "region": np.array(["DE", "DE", "FR"])},
    np.array([5.0, 6.0, 7.0]),
)
demand
# SparseArray(('year', 'region'), shape=(3, 2), nnz=3, absence='empty')
```

Three of the six coordinates of the frame have an entry. `to_dense()` writes the additive identity at the other three: under `absence="empty"`, an absent coordinate contributes nothing.

```python
demand.to_dense()
# array([[5., 0.],
#        [6., 7.],
#        [0., 0.]])
```

The product of an array over one dimension and an array over two aligns on the shared dimension. The narrower operand provides a factor at every coordinate of the wider frame, and the result is over the wider frame:

```python
price = nb.from_long(
    ("region",), coords, {"region": np.array(["DE", "FR"])}, np.array([2.0, 3.0])
)
cost = demand * price
cost.dims  # ('year', 'region')
cost.to_dense()
# array([[10.,  0.],
#        [12., 21.],
#        [ 0.,  0.]])
cost.sum("region")  # SparseArray(('year',), shape=(3,), nnz=2, absence='empty')
cost.sum()  # 43.0
```

The sum over `region` has two entries, not three. `cost` has no entry at 2050, and the sum has no entry there either.

## Six concepts

The package defines six concepts. Each concept addresses a question that follows from the previous one. Each subsection below shows the error that occurs without the concept.

| | Concept | Question |
|---|---|---|
| 1 | a coordinate | Which position has the label `"FR"`, and which label is at position 3? |
| 2 | `StoredCoord`, `ProductCoord`, `SubsetCoord` | Does a dimension require its labels stored in an array? |
| 3 | `SparseArray`, `DenseArray`, `Array` | Store every cell of the grid, or only the entries that exist? |
| 4 | `absence` | Does an absent coordinate mean zero, or unknown? |
| 5 | `Domain` | Which coordinates exist over several dimensions at once? |
| 6 | `EntryBuffer` | How is one result assembled from many separate blocks? |

### 1. A label is not a position

numpy aligns by position. For two arrays with the same data in different orders, numpy combines cell with cell and returns a wrong result with no error:

```python
import numpy as np
import nimblend as nb

a = nb.from_dense(
    np.array([[1.0, 2.0], [3.0, 4.0]]),
    {"year": np.array([2030, 2040]), "region": np.array(["DE", "FR"])},
)
c = nb.from_dense(
    np.array([[2.0, 1.0], [4.0, 3.0]]),
    {"year": np.array([2030, 2040]), "region": np.array(["FR", "DE"])},
)

a.to_dense() + c.to_dense()  # numpy: adds DE to FR
# array([[3., 3.],
#        [7., 7.]])
```

`c` has the same value as `a` at each pair of labels. Only the order of its regions differs. The sum by label is therefore `a` doubled, `[[2, 4], [6, 8]]`. The sum of the grids differs: numpy pairs cell with cell, and column 0 is `"DE"` in one grid and `"FR"` in the other.

An operation that aligns by label does not produce that error. A coordinate provides the alignment: for one dimension, it returns the position of a label and the label at a position. Two arrays whose dimensions are in another order then combine with no action from the caller:

```python
b = nb.from_dense(
    np.array([[10.0, 30.0], [20.0, 40.0]]),
    {"region": np.array(["DE", "FR"]), "year": np.array([2030, 2040])},
)

(a + b).dims  # ('region', 'year')
(a + b).to_dense()
# array([[11., 33.],
#        [22., 44.]])
```

For two arrays whose labels differ, the operation raises `ValueError`:

```python
a + c
# ValueError: dimension(s) ['region'] have different labels in the two arrays;
# conform one to the other first
```

**Effect.** The frame of a result follows from the dimension names alone. An operand may be transposed, narrower than the other, or partially overlapping it, and the caller reorders no operand. Where the labels differ, the operation raises and returns no value.

### 2. A dimension does not require stored labels

A coordinate returns the position of a label and the label at a position. It does not require an array of labels for that. Where the positions run `0, 1, 2, …` over a product of axis sizes, the position is computed. An array of labels then adds memory and no information:

```python
stored = nb.StoredCoord(np.arange(10_000_000))
generated = nb.ProductCoord((10_000_000,))

stored.labels.nbytes  # 80000000
len(stored), len(generated)  # (10000000, 10000000)
stored.to_index(np.array([3])), generated.to_index(np.array([3])).ravel()
# (array([3]), array([3], dtype=int32))
```

The two coordinates return the same label. One uses 80 MB, and the other a tuple and an integer. The package therefore has three interchangeable coordinates. `StoredCoord` stores arbitrary labels. `ProductCoord` computes the positions of a full product. `SubsetCoord` computes the positions of a subset of a product, and numbers its members in code order.

| Coordinate | Stores | Use |
|---|---|---|
| `StoredCoord(labels)` | An array of labels | A dimension with arbitrary labels |
| `ProductCoord(sizes, start=0)` | Axis sizes only | A dimension over a full product |
| `SubsetCoord(codes, sizes, start=0)` | The raveled codes of the subset | A dimension over part of a product |

`StoredCoord` builds the sorted permutation for lookups at the first lookup, not at construction. A stored coordinate that only defines the extent of a dimension never builds it.

**Effect.** A `ProductCoord` over millions of positions stores only its axis sizes. A subset of a product is a coordinate of its own, not a full grid with missing cells.

### 3. Entries, or cells

Labels do not determine how many positions have a value. Each storage form suits a different density, and both implement one contract.

`SparseArray` stores an index matrix, with one row per dimension and one column per entry, and a value buffer. It stores nothing for an absent coordinate. The entries are in canonical order: sorted by their C-order ravel key, with no key repeated. Canonical order makes alignment a merge over sorted keys, not a hash join.

`DenseArray` stores an ndarray over the same labeled dimensions. It is faster above about 1% density. The [Performance](#performance) section measures the crossover: near 1% density for time. The sparse form uses less memory at every density the section measures.

Both implement the `Array` protocol. A consumer writes one piece of code and chooses the storage form by density:

```python
isinstance(a, nb.Array)  # True
```

**Effect.** The storage form is a choice of density, not of interface. A change of storage form changes no calling code.

### 4. Absence has two meanings

An array of entries requires a meaning for an absent entry. The two meanings require opposite arithmetic. A coordinate that contributes nothing is the additive identity. A sum is then the union of the entries of the two operands. A coordinate that was not modeled has no value. A sum is then the intersection: the result has no entry where either operand has none.

The data does not distinguish the two meanings. The array declares one:

```python
coords = {"region": nb.StoredCoord(np.array(["DE", "FR"]))}
de = nb.from_long(("region",), coords, {"region": np.array(["DE"])}, np.array([1.0]))
fr = nb.from_long(("region",), coords, {"region": np.array(["FR"])}, np.array([2.0]))

(de + fr).nnz  # 2: absence is the identity, and the sum is a union

unknown = de.as_unknown() + fr.as_unknown()
unknown.nnz  # 0: neither coordinate has a value in both operands
```

| | `"empty"` | `"unknown"` |
|---|---|---|
| An absent coordinate | contributes nothing | was not modeled |
| Addition aligns by | union | intersection |
| Reduction | uses the present entries | requires `skip=True` or `fill=<value>` |
| `to_dense()` | writes `0.0` | requires `fill=<value>` where a coordinate is absent |

Under `"unknown"`, an operation that requires a value at an absent coordinate raises `ValueError`:

```python
de.as_unknown().sum()
# ValueError: absence is 'unknown' and no reduction policy is given; pass
# skip=True to reduce the present entries, or fill=<value> to include the absent
# coordinates
```

A stored `0.0` is distinct from an absent coordinate under both declarations. It is a present coordinate with the value zero.

**Effect.** A missing measurement is never counted as zero. The caller declares the meaning once, on the array, not at each operation.

### 5. A question about several dimensions at once

A coordinate covers one dimension. Other questions concern a tuple of dimensions: the coordinates of an array, the coordinates two arrays share, and the coordinates an operation removes. A numbering of those coordinates is a further question. A coordinate does not return these. A member is a combination across dimensions, not a position along one.

A `Domain` is a sorted set of unique multi-indices over named dimensions, stored as raveled codes:

```python
years = nb.StoredCoord(np.array([2030, 2040, 2050]))
regions = nb.StoredCoord(np.array(["DE", "FR"]))
coords = {"year": years, "region": regions}

p = nb.from_long(
    ("year", "region"),
    coords,
    {"year": np.array([2030, 2040]), "region": np.array(["DE", "DE"])},
    np.array([1.0, 2.0]),
)
q = nb.from_long(
    ("year", "region"),
    coords,
    {"year": np.array([2040, 2050]), "region": np.array(["DE", "FR"])},
    np.array([3.0, 4.0]),
)

shared = p.domain().intersect(q.domain())
shared.size  # 1
shared.labels()  # {'year': array([2040]), 'region': array(['DE'], dtype='<U2')}
p.restrict(shared).nnz  # 1
```

A domain is an ordered set, and it defines a numbering of its members. The members can therefore form a dimension of another array. `as_coord(start)` returns the domain as a coordinate. `identity(into, coord)` pairs each member with its position along a new dimension. The two numberings are equal: a member has the same position under both.

```python
shared.as_coord(start=5)  # SubsetCoord(1 of (3, 2), start=5)
```

**Effect.** A domain supports intersection, union and difference, and numbers its members. A consumer finds which members remain after an operation, and gives them positions along a new dimension.

### 6. One result from many blocks

A result assembled from several blocks normally exists twice in memory: once as the blocks, and once as their concatenation. `EntryBuffer` is one preallocated index and value buffer that returns successive slices. The kernel functions write into a slice. A block therefore never exists as a separate object.

The [Performance](#performance) section measures the effect. Assembling sixteen blocks peaks at 1.04 times the final size. The excess is the working set of one sort-merge, not a second copy of the result.

**Effect.** The peak memory of a large result is the result plus one block, not twice the result.

---

These six concepts are the whole package. The sections below describe them in detail: the frame of a result, the operations of an array, grouping and export, and the measurements.

## Alignment

The frame of a binary result follows from the dimension names of the two operands. Every operator applies one rule:

| Operands | Result frame |
|---|---|
| Equal | The shared order |
| One a subset of the other | The wider |
| Overlapping | The dimensions of the left operand, then those only the right operand has |
| Sharing no dimension | Raises `ValueError` |

`combined_dims` applies the rule to two tuples of dimension names. A caller obtains the frame before building either operand:

```python
nb.combined_dims(("P", "Q"), ("Q", "R"))  # ('P', 'Q', 'R')
nb.combined_dims(("P",), ("Q",))
# ValueError: frames ('P',) and ('Q',) share no dimension; pass operands that
# share a dimension
```

Two frames that share no dimension have no dimension to align on. Their combination is an outer product, and the operators raise `ValueError` for it. For an outer product, the caller first expands one operand over the dimensions of the other with `expand`.

Alignment is by label. An operation on operands whose shared dimension has different labels raises `ValueError`, as [section 1](#1-a-label-is-not-a-position) shows. It does not align them by position. `conform` reconciles them: it reads an array at exactly the labels given, in the order given.

A quotient raises `ValueError` where the denominator is absent and the numerator has a value. The quotient at that coordinate is undefined: it is neither zero nor one.

```python
# ValueError: the denominator is absent at 1 coordinate(s) where the numerator
# has a value; restrict the numerator to the domain of the denominator
```

A stored zero is a value. Division by a stored zero follows floating-point arithmetic: infinity, or NaN where the numerator is also zero.

An operation on a `SparseArray` and a `DenseArray` returns a `SparseArray`. The present coordinates of the dense operand become entries, and the sparse arithmetic above applies. A product intersects presence: it has at most the entries of the sparse operand.

## Operations

Every operation below is part of the `Array` protocol, and both implementations support it.

**Arithmetic.** `+`, `-`, `*`, `/`, `**` by a number, unary `-`, and the reflected forms. The exponent of `**` is a number. An array as the exponent raises `TypeError`.

**Reductions.** `sum`, `mean`, `min` and `max`, over one named dimension or over the whole array. Each takes the `skip=` or `fill=` policy that an `"unknown"` array requires. Reducing every dimension in turn returns an array over no dimension, with one entry: the total.

**Selection and reshaping**

| Method | Returns |
|---|---|
| `sel({dim: label})` | The entries at the given labels, without the selected dimensions |
| `restrict(domain)` | The entries whose coordinate over the dimensions of the domain is a member of the domain |
| `expand(dims, coords)` | Every entry replicated over the full extent of the given dimensions |
| `conform(dims, labels)` | The array read at exactly `labels`, over `dims` |
| `transpose(*dims)` | The dimensions in the order given, or reversed when none are given |
| `rename({old: new})` | The array with its dimensions renamed |
| `shift({dim: n})` | The entries moved along a dimension; an entry moved outside the frame is removed |
| `roll({dim: n})` | The entries moved along a dimension, with wrapping at the ends |

`expand` appends its dimensions, and the result stays canonical. `transpose` gives another order. A dimension of size `k` multiplies the number of entries by `k`. The caller requests replication explicitly, and no operator performs it implicitly. `conform` takes each label once, and a repeated label raises `ValueError`.

**Reading the entries.** `coordinates(dims)` and `values()` return copies of the entries, not the buffers of the array. `domain(dims)` returns the distinct coordinates. `to_dense(fill)` returns a grid. `nnz`, `dims`, `shape` and `coords` describe the frame.

## Grouping and matrix export

`group` replaces a tuple of dimensions with one dimension, numbered by a domain. The index of a member along the new dimension is its position in the domain plus `offset`.

```python
grouped = demand.group(("year",), into="g")
grouped  # SparseArray(('g', 'region'), shape=(2, 2), nnz=3, absence='empty')
grouped.coords["g"]  # SubsetCoord(2 of (3,), start=0)
```

The grouped dimensions are a leading prefix of the canonical order, and `group` raises `ValueError` otherwise. The result is then canonical as written and requires no sort. An entry at a coordinate outside the domain is removed. A non-zero `offset` numbers the result into a wider extent, and several results then share one destination buffer and one numbering.

`to_csr` exports a two-dimensional array as CSR arrays. Canonical order sorts by row and then by column, the order CSR requires. The column indices and the values are returned as views, and only the row pointer is built:

```python
indices, values, indptr = grouped.to_csr()
# [0, 0, 1]  [5.0, 6.0, 7.0]  [0, 1, 3]
```

`to_csr` raises `ValueError` where a row position is outside the extent of the first dimension, as a non-zero `offset` can produce.

## Assembling blocks into one buffer

`EntryBuffer` is a preallocated index and value buffer that returns successive slices. A block computed into a reserved slice never exists as a separate object. Assembling several blocks then keeps one copy of the result in memory, not one copy per block plus the result.

```python
buffer = nb.EntryBuffer(ndim=2, capacity=10)
index, data = buffer.reserve(3)
index[:] = np.array([[0, 1, 2], [0, 0, 1]])
data[:] = [1.0, 2.0, 3.0]

buffer.array(
    {"a": nb.StoredCoord(np.arange(4)), "b": nb.StoredCoord(np.arange(2))}, ("a", "b")
)
# SparseArray(('a', 'b'), shape=(4, 2), nnz=3, absence='empty')
```

`buffer.array(...)` does not copy. `group`, `kernel.reduce_axis`, `kernel.gather` and `kernel.shift_axis` accept a reserved slice as the `out=` destination.

`SparseArray.from_canonical` builds an array from canonical buffers without a copy. The caller guarantees that the index is sorted with no key repeated. `nb.is_canonical(index, shape)` checks that condition. `from_canonical` does not check it: the check requires the ravel that the method avoids.

## Performance

The scripts in `benchmarks/` produce the figures below. The figures vary with the machine.

**Crossover between sparse and dense.** Addition of two 3000×3000 arrays, over a range of densities: the fraction of the cells with a value (`bench_crossover.py`).

| Density | Dense | Sparse | Dense memory | Sparse memory |
|---|---|---|---|---|
| 50.0% | 10.15 ms | 362.87 ms | 72.0 MB | 56.6 MB |
| 10.0% | 9.93 ms | 93.15 ms | 72.0 MB | 13.7 MB |
| 1.0% | 9.91 ms | 7.66 ms | 72.0 MB | 1.4 MB |
| 0.1% | 9.86 ms | 0.76 ms | 72.0 MB | 0.1 MB |

The crossover in time is near 1% density. The sparse form uses less memory at every density in the table. A dense grid is faster above about 1% density, and the package has both implementations for that reason.

**Presence encoding of a dense array.** Operations on 2000×2000 float64 arrays at 90% density, for each encoding of presence (`bench_presence.py`).

| Operation | Mask | NaN tag |
|---|---|---|
| `a + b`, absence propagating | 14.04 ms | 2.30 ms |
| `a + b`, absence as identity | 10.88 ms | 35.91 ms |
| Sum over present values | 4.66 ms | 8.61 ms |
| Storage beside 32.0 MB of values | 4.0 MB | none |

The two declarations require opposite encodings. An `"unknown"` array requires propagation, and the hardware propagates NaN. An `"empty"` array requires substitution of the identity, and a mask substitutes it in one pass. `DenseArray` therefore stores NaN at an absent coordinate under `"unknown"`, and a boolean mask under `"empty"`.

**Assembling into one buffer.** Peak memory against final memory, reducing blocks into a shared destination (`bench_assembly.py`).

| Blocks | Final | Peak | Ratio |
|---|---|---|---|
| 1 | 16.00 MB | 25.00 MB | 1.56× |
| 3 | 48.00 MB | 57.00 MB | 1.19× |
| 8 | 128.00 MB | 137.00 MB | 1.07× |
| 16 | 256.00 MB | 265.00 MB | 1.04× |

The excess is constant at 9 MB. It is the working set of one sort-merge, not a second copy of the result. The ratio therefore falls as the number of blocks grows.

## Architecture

The package has two layers.

`kernel.py` contains module-level functions over plain numpy buffers: `ravel`, `unravel`, `distinct`, `canonicalize`, `align`, `gather`, `reduce_axis`, `shift_axis`, `to_csr`, `lookup`, `first_repeat` and `is_canonical`. They take and return numpy arrays, and they use no labels or dimensions. Every array operation calls them. A compiled module with the same signatures can replace the layer.

The array layer, `SparseArray`, `DenseArray`, `Domain` and the coordinates, stores the labels and the frames, validates the arguments, and calls the kernel functions.

Four design choices are visible in the interface. Sorting uses a single int64 ravel key: an order over several dimensions is one `argsort`, not a lexsort. Alignment is a merge over sorted keys with a take-vector per operand, not a hash join. A block whose keys already ascend is not sorted again. Stored zeros are kept: a stored zero marks a present coordinate.

`nimblend.kernel` and the `.index` and `.data` of an array are internal. A consumer uses them through the array layer. The public interface is the set of names in `nimblend.__all__`, imported from the top-level module.

## Failure behavior

The package raises an exception and does not substitute another behavior. It raises `ValueError` for operands whose labels differ, whose absence declarations differ, or whose frames share no dimension. It raises `ValueError` for a repeated coordinate in a constructed array, and for a quotient where the denominator is absent. A method that takes dimension names raises `ValueError` for a name the array does not have, and for a repeated name. Under `"unknown"`, a reduction or `to_dense()` raises until the caller passes the policy. Each message reports the condition, then the action to take.

## Development

```bash
pip install -e ".[dev]"

pytest
ruff check . && ruff format --check .
```

The suite tests the contract in four ways. `tests/conformance.py` defines one `Array` contract, and both implementations run against it. `test_alignment_ladder.py` tests every pair of frames with the four arithmetic operators, and checks that the two implementations return equal results. `test_kernel_*.py` test the buffer layer. `test_boundary_vocabulary.py` scans the source of the package for names and words of a consuming layer.

## Public interface

```python
from nimblend import (
    Array,  # the contract; runtime-checkable, never constructed
    SparseArray,  # entries in canonical order
    DenseArray,  # an ndarray over labeled dimensions
    Domain,  # a set of coordinates over a tuple of dimensions
    EntryBuffer,  # one preallocated destination for several blocks
    StoredCoord,  # labels stored as an array
    ProductCoord,  # positions of a full product of axis sizes
    SubsetCoord,  # positions of a subset of a product
    from_long,  # an array from label columns and a value column
    from_dense,  # an array from a grid and its labels
    combined_dims,  # the frame of a binary result
    is_canonical,  # whether an index is sorted with no key repeated
)
```

## License

MIT. See `LICENSE`.

The citation metadata is in `CITATION.cff`.
