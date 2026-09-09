# nimblend

Labelled sparse N-dimensional arrays for Python.

The documentation site is at <https://cdgaete.github.io/nimopt/nimblend/>.

`nimblend` stores an N-dimensional array as the entries it carries rather than as a grid, and names every position by a label rather than by an offset. Two arrays combine by aligning their labels, never by matching their shapes, so an operand's dimensions may be reordered, may nest inside the other's, or may overlap it partially, and the result carries a frame determined by the dimension names alone. The package declares one dependency, `numpy`, and its vocabulary is dimensions, labels, entries and alignment.

The distinguishing property is that absence is a first-class declaration. An array states whether a coordinate it does not carry contributes nothing (`"empty"`) or was not modelled (`"unknown"`), and every operator and reduction follows from that declaration rather than from an implementation's convenience. A stored `0.0` remains distinct from an absent coordinate under both.

## Installation

From a checkout:

```bash
pip install .            # the package
pip install -e ".[dev]"  # editable, with the test and lint tooling
```

Python 3.13 or 3.14, and `numpy >= 2.5.2`. A built wheel is in `dist/`.

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

Three of the six coordinates the frame spans carry an entry. Densifying places the additive identity at the rest, because that is what `absence="empty"` declares them to contribute:

```python
demand.to_dense()
# array([[5., 0.],
#        [6., 7.],
#        [0., 0.]])
```

An array over one dimension multiplies an array over two by aligning on the dimension they share. The narrower operand supplies a factor at every coordinate of the wider frame, and the result is over the wider frame:

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

The reduction over `region` carries two entries, not three: 2050 holds no entry to sum, so the reduced array holds no entry there either.

## Why there are six concepts

A labelled array could have been one class. It is six, and the reason is that each one answers a question the one before it raises. Read in order, the table derives the library: every row states a question, and the subsection under it shows what goes wrong when the answer is missing.

| | Concept | The question it answers |
|---|---|---|
| 1 | a coordinate | Where does the label `"FR"` sit, and what sits at position 3? |
| 2 | `StoredCoord`, `ProductCoord`, `SubsetCoord` | Must a dimension's names be stored in order to be answered? |
| 3 | `SparseArray`, `DenseArray`, `Array` | Hold every cell of the grid, or only the entries that exist? |
| 4 | `absence` | Does a coordinate holding nothing mean zero, or mean unknown? |
| 5 | `Domain` | Which coordinates exist over several dimensions at once? |
| 6 | `EntryBuffer` | How is one result assembled out of many separate blocks? |

### 1. A label is not a position

numpy aligns by position. Two arrays holding the same data under different orderings are combined cell against cell, and the answer is wrong without saying so:

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

`c` holds exactly the data `a` holds — the same value against the same pair of labels — and differs only in listing its regions the other way round. Adding the two by label therefore has to give `a` doubled, `[[2, 4], [6, 8]]`. Adding the grids gives something else, because numpy pairs cell with cell and cannot see that column 0 means `"DE"` on one side and `"FR"` on the other.

An operation that aligns by label cannot make that mistake. A coordinate is what makes it possible: it answers, for one dimension, which position a label occupies and which label sits at a position. Given that, two arrays whose dimensions are merely in a different order combine without the caller doing anything:

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

And two arrays whose labels genuinely disagree are refused rather than guessed at:

```python
a + c
# ValueError: dimension(s) ['region'] carry different labels in the two arrays;
# an entry is aligned by its label, so conform one to the other first
```

**What it buys.** The frame of a result follows from the dimension names alone, so an operand may be transposed, narrower than the other, or only partly overlapping it, and nothing is lined up by hand. Where the labels cannot be reconciled the operation stops instead of returning a plausible number.

### 2. A dimension's names need not be stored

A coordinate has to answer the label question; it does not have to hold an array of labels to do so. Where positions run `0, 1, 2, …` over a product of axis sizes, the answer is arithmetic, and storing it would cost a great deal for nothing:

```python
stored = nb.StoredCoord(np.arange(10_000_000))
generated = nb.ProductCoord((10_000_000,))

stored.labels.nbytes  # 80000000
len(stored), len(generated)  # (10000000, 10000000)
stored.to_index(np.array([3])), generated.to_index(np.array([3])).ravel()
# (array([3]), array([3]))
```

The two answer alike; one costs 80 MB and the other a tuple and an integer. That is why there are three coordinates rather than one, and why they are interchangeable: `StoredCoord` holds arbitrary labels, `ProductCoord` computes the positions of a full product, and `SubsetCoord` computes the positions of a subset of one, numbering its members in code order.

| Coordinate | Holds | Suited to |
|---|---|---|
| `StoredCoord(labels)` | An array of labels | A dimension named by arbitrary labels |
| `ProductCoord(sizes, start=0)` | Axis sizes only | A dimension spanning a full product |
| `SubsetCoord(codes, sizes, start=0)` | The subset's ravelled codes | A dimension spanning part of a product |

`StoredCoord` builds the sorted permutation a lookup needs on the first lookup rather than at construction, so even a stored coordinate carried only to state a dimension's extent never pays for one.

**What it buys.** A dimension spanning millions of positions costs nothing to carry, so a subset of a product is a thing in its own right rather than a full grid with holes in it.

### 3. Entries, or cells

Naming positions says nothing about how many of them carry a value. Both answers are reasonable and neither is right everywhere, so both exist behind one contract.

`SparseArray` holds an index matrix of one row per dimension and one column per entry, beside a value buffer, and holds nothing for a coordinate it does not carry. Entries are kept in canonical order — sorted by their C-order ravel key, with no key repeated — which is what makes alignment a merge over sorted keys rather than a hash join.

`DenseArray` holds an ndarray over the same labelled dimensions, and is the faster representation once the grid is well populated. The [Performance](#performance) section measures where one overtakes the other: the crossover in time sits near one per cent density, and memory favours the sparse form well before that.

`Array` is the contract both satisfy, so a consumer writes one piece of code and chooses the representation on density rather than on capability:

```python
isinstance(a, nb.Array)  # True
```

**What it buys.** Density is a storage decision rather than an API decision, and changing it changes no calling code.

### 4. Absence has two meanings

An array holding entries rather than cells must say what a missing entry means — and there are two answers, which want opposite arithmetic. A coordinate that contributes nothing is the additive identity, so a sum over it is a union of what either operand carries. A coordinate that was never modelled has no value at all, so a sum over it is an intersection: nothing is known where either operand is silent.

Nothing in the data distinguishes the two, so the array declares it:

```python
coords = {"region": nb.StoredCoord(np.array(["DE", "FR"]))}
de = nb.from_long(("region",), coords, {"region": np.array(["DE"])}, np.array([1.0]))
fr = nb.from_long(("region",), coords, {"region": np.array(["FR"])}, np.array([2.0]))

(de + fr).nnz  # 2 — absence is the identity, so the sum is a union

unknown = de.as_unknown() + fr.as_unknown()
unknown.nnz  # 0 — nothing is known at either coordinate
```

| | `"empty"` | `"unknown"` |
|---|---|---|
| An absent coordinate means | it contributes nothing | it was not modelled |
| Addition aligns by | union | intersection |
| Reduction | uses the entries held | must state `skip=True` or `fill=<value>` |
| `to_dense()` | places `0.0` | must state `fill=<value>` where coverage is partial |

Under `"unknown"` the operations that would have to invent a value say so instead:

```python
de.as_unknown().sum()
# ValueError: this array declares absence 'unknown', so a reduction must state
# skip=True to use present entries only, or fill=<value> to count absences as
# that value
```

A stored `0.0` is distinct from an absent coordinate under both declarations: it is a coordinate that is present and worth nothing.

**What it buys.** A missing measurement is never quietly counted as a zero, and the caller states which of the two meanings applies once, on the array, rather than at every operation that reads it.

### 5. A question about several dimensions at once

A coordinate answers for one dimension. Many of the questions that arise are about a tuple of them: which coordinates does this array actually carry, which do two arrays share, which did an operation drop, and what number does each of them get? None of those can be put to a coordinate, because a member is a combination across dimensions rather than a position along one.

A `Domain` is that answer — a sorted, unique set of multi-indices over named dimensions, held as ravelled codes:

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

Because a domain is a set with an order, it also states a numbering, which is what lets its members become a dimension of something else. `as_coord(start)` reads it back as a coordinate, and `identity(into, coord)` pairs each member with its own position along a new dimension — the two agree, so a member numbered one way sits where the other puts it.

```python
shared.as_coord(start=5)  # SubsetCoord(1 of (3, 2), start=5)
```

**What it buys.** Set arithmetic over coordinates — intersection, union, difference — plus a numbering, so a consumer can ask which members survive an operation and give the survivors positions along a dimension of their own.

### 6. One result out of many blocks

A result assembled from several blocks would ordinarily exist twice: once as the blocks, and once as the concatenation of them. `EntryBuffer` is one preallocated index and value buffer that hands out successive slices, and the kernel operations write directly into a slice, so a block never exists as a separate object.

The [Performance](#performance) section measures it: assembling sixteen blocks peaks at 1.04× the final size, and the excess is one transient sort-merge rather than a second copy of the result.

**What it buys.** The peak memory of building a large result is the result plus one block, rather than twice the result.

---

Those six are the whole library. Everything below states what they do in detail: how a result's frame is decided, which operations an array answers, how a block is grouped and exported, and what the measurements show.

## Alignment

The frame a binary result carries is read from the two operands' dimension names alone, by one rule that every operator obeys.

| Operands | Result frame |
|---|---|
| Equal | The shared order |
| One nesting inside the other | The wider |
| Overlapping | The left operand's dimensions, then those only the right carries |
| Sharing no dimension | Refused |

`combined_dims` states that rule without an array to read it from, so a caller may know the answer before materialising either operand:

```python
nb.combined_dims(("P", "Q"), ("Q", "R"))  # ('P', 'Q', 'R')
nb.combined_dims(("P",), ("Q",))
# ValueError: frames ('P',) and ('Q',) share no dimension; there is nothing to
# align them on
```

Frames sharing no dimension have nothing to align on, and their combination would be an outer product no caller asked for, so it is refused rather than performed.

Alignment is by label throughout, and operands whose shared dimension carries different labels are refused rather than aligned by position, as [step 1](#1-a-label-is-not-a-position) shows. `conform` is what reconciles them: it reads an array at exactly the labels given, in the order given.

A quotient refuses an absent denominator, which is a coverage question rather than an arithmetic one: a numerator reaching a coordinate the denominator does not carry has no quotient there that is either zero or one.

```python
# ValueError: the denominator is absent at 1 coordinate(s) the numerator
# carries; a quotient there is not zero and not one, so it is refused
```

A stored zero, by contrast, is a value the array carries, so dividing by one answers what the arithmetic answers — infinity, or NaN where the numerator is zero too.

Mixing the two implementations is allowed, and every mixed operation answers a `SparseArray`: the dense operand contributes its present coordinates as entries, and the arithmetic is then the sparse arithmetic above. A product intersects presence, so it carries at most the entries the sparse operand holds.

## Operations

Every operation below is on the `Array` contract and is answered by both implementations.

**Arithmetic** — `+`, `-`, `*`, `/`, `**` (by a number), unary `-`, and the reflected forms. An exponent must be a number: raising an array by an array is not an operation the contract offers.

**Reductions** — `sum`, `mean`, `min`, `max`, each over one named dimension or over the whole array, and each taking the `skip=` / `fill=` policy an `"unknown"` array requires. Reducing every dimension in turn ends at an array over none, which carries the single entry holding the total.

**Selection and reshaping**

| Method | Answers |
|---|---|
| `sel({dim: label})` | Entries at the given labels, dropping each dimension named |
| `restrict(domain)` | The entries whose coordinate over the domain's dimensions it carries |
| `expand(dims, coords)` | Every entry replicated across the full extent of the named dimensions |
| `conform(dims, labels)` | The array read at exactly `labels`, laid out over `dims` |
| `transpose(*dims)` | The dimensions in the order given, or reversed when none are named |
| `rename({old: new})` | The array with dimensions renamed |
| `shift({dim: n})` | Entries moved along a dimension, those leaving the frame dropped |
| `roll({dim: n})` | Entries moved along a dimension, wrapping at the ends |

`expand` appends its dimensions, which keeps the result canonical; a different order is reached with `transpose`. Adding a dimension of size `k` multiplies the entry count by `k`, so replication is stated by the caller rather than implied by an operator. `conform` names each label once, since a repeat would ask one position to occupy two.

**Reading the entries out** — `coordinates(dims)` and `values()` answer the entries as copies without handing out the buffers the array owns; `domain(dims)` answers the distinct coordinates covered; `to_dense(fill)` answers a grid; `nnz`, `dims`, `shape` and `coords` answer the frame.

## Grouping and matrix export

`group` collapses a tuple of dimensions into one dimension numbered by a domain. A member's position in the domain, plus `offset`, is its index along the new dimension.

```python
grouped = demand.group(("year",), into="g")
grouped  # SparseArray(('g', 'region'), shape=(2, 2), nnz=3, absence='empty')
grouped.coords["g"]  # SubsetCoord(2 of (3,), start=0)
```

The grouped dimensions must be a leading prefix of the canonical order, which is what makes the result canonical as written rather than sorted afterwards. An entry at a coordinate the domain does not carry is not emitted. A non-zero `offset` numbers the result into an extent wider than its own members span, which is what lets several results share one destination buffer and one numbering.

A two-dimensional array exports as CSR triplets. Canonical order is sorted by row and then by column, which is CSR's own requirement, so the column indices and values are returned as views and only the row pointer is built:

```python
indices, values, indptr = grouped.to_csr()
# [0, 0, 1]  [5.0, 6.0, 7.0]  [0, 1, 3]
```

## Assembling blocks into one buffer

`EntryBuffer` is a preallocated index and value buffer handing out successive slices. A block computed directly into a reserved slice never exists as a separate object, so assembling several of them holds one copy of the result rather than one copy per block plus the result.

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

`buffer.array(...)` takes no copy, and `group`, `kernel.reduce_axis`, `kernel.gather` and `kernel.shift_axis` all accept such a slice as an `out=` destination.

`SparseArray.from_canonical` is the matching door for a caller that has built canonical buffers itself: it takes no copy, and the caller states that the index is sorted with no key repeated. `nb.is_canonical(index, shape)` answers that question where a caller cannot. Verifying it inside `from_canonical` would cost the ravel the path exists to avoid.

## Performance

The figures below are produced by the scripts in `benchmarks/` and vary with the machine.

**Where sparse overtakes dense.** Addition of two 3000×3000 arrays, sweeping the fraction of the coordinate grid that carries a value (`bench_crossover.py`):

| Density | Dense | Sparse | Dense memory | Sparse memory |
|---|---|---|---|---|
| 50.0% | 10.15 ms | 362.87 ms | 72.0 MB | 56.6 MB |
| 10.0% | 9.93 ms | 93.15 ms | 72.0 MB | 13.7 MB |
| 1.0% | 9.91 ms | 7.66 ms | 72.0 MB | 1.4 MB |
| 0.1% | 9.86 ms | 0.76 ms | 72.0 MB | 0.1 MB |

The crossover in time sits near one per cent; memory favours the sparse form well before that. A dense grid is the faster representation for a well-populated frame, which is why both implementations exist and satisfy one contract.

**Why a dense array stores what it declares.** Operations on 2000×2000 float64 arrays at 90% density, under each encoding of presence (`bench_presence.py`):

| Operation | Mask | NaN tag |
|---|---|---|
| `a + b`, absence propagating | 14.04 ms | 2.30 ms |
| `a + b`, absence as identity | 10.88 ms | 35.91 ms |
| Sum over present values | 4.66 ms | 8.61 ms |
| Storage beside 32.0 MB of values | 4.0 MB | none |

The two declarations want opposite encodings: an `"unknown"` array wants propagation, which NaN performs in the hardware, and an `"empty"` array wants substitution of the identity, which a mask performs in one pass. `DenseArray` therefore stores what it declares.

**Assembling into one buffer.** Peak against final memory while reducing blocks directly into a shared destination (`bench_assembly.py`):

| Blocks | Final | Peak | Ratio |
|---|---|---|---|
| 1 | 16.00 MB | 25.00 MB | 1.56× |
| 3 | 48.00 MB | 57.00 MB | 1.19× |
| 8 | 128.00 MB | 137.00 MB | 1.07× |
| 16 | 256.00 MB | 265.00 MB | 1.04× |

The excess is constant at 9 MB and per-block: it is the transient working set of one sort-merge, not a second copy of the result, so the ratio falls as blocks accumulate.

## Architecture

The package is two layers, and the seam between them is deliberate.

`kernel.py` is module-level functions over plain numpy buffers — `ravel`, `unravel`, `distinct`, `canonicalize`, `align`, `gather`, `reduce_axis`, `shift_axis`, `to_csr`, `lookup`, `first_repeat`, `is_canonical`. They take and return numpy arrays and know nothing of labels or dimensions. Every array operation is carried by them, so a compiled module satisfying the same signatures replaces the layer wholesale.

The array layer above — `SparseArray`, `DenseArray`, `Domain`, the coordinates — holds the labels, the frames and the refusals, and emits kernel calls in order.

Several design choices are worth naming because they show up in the interface. Sorting is on a single int64 ravel key, so ordering over several dimensions is one `argsort` rather than a lexsort. Alignment is a merge over sorted keys with a take-vector per operand, rather than a hash join. A block whose keys already ascend is copied straight through instead of being permuted, because what a sort costs is applying it. Stored zeros are kept, because a stored zero states that a coordinate is present.

`nimblend.kernel` and an array's `.index` and `.data` read like interfaces and are not: a consumer of the package reaches them through the array layer. The public interface is the names `nimblend.__all__` exports, reached through the top-level module.

## Failure behaviour

The package raises rather than substituting a different behaviour and continuing. Operands whose labels differ, whose absence declarations differ, or whose frames share no dimension are refused; a duplicate coordinate in a constructed array is refused; a quotient at a coordinate the denominator does not carry is refused; a reduction or densification that would have to invent a value for an `"unknown"` array is refused until the caller states the policy. Each message names what was seen and what the contract expects.

## Development

```bash
pip install -e ".[dev]"

pytest                       # 717 passed, 8 skipped
ruff check . && ruff format --check .
```

The suite states the contract from several directions. `tests/conformance.py` holds one `Array` contract that both implementations are run against; `test_alignment_ladder.py` sweeps every pair of frames across all four operators and asserts that the two implementations answer alike; `test_kernel_*.py` state the buffer layer's contract with the array layer above it; and `test_boundary_vocabulary.py` scans the package's own source for names and words belonging to a consuming layer, so the vocabulary stays that of a labelled array.

## Public interface

```python
from nimblend import (
    Array,  # the contract; runtime-checkable, never constructed
    SparseArray,  # entries in canonical order
    DenseArray,  # an ndarray over labelled dimensions
    Domain,  # the coordinates carried over a tuple of dimensions
    EntryBuffer,  # one preallocated destination for several blocks
    StoredCoord,  # labels held as an array
    ProductCoord,  # positions of a full product of axis sizes
    SubsetCoord,  # positions of a subset of a product
    from_long,  # an array from label columns and a value column
    from_dense,  # an array from a grid and its labels
    combined_dims,  # the frame a binary result carries
    is_canonical,  # whether an index is sorted with no key repeated
)
```

## Licence

MIT. See `LICENSE`.

Citation metadata is in `CITATION.cff`.
