import numpy as np

from nimblend import kernel


def test_lookup_returns_the_position_of_each_probe():
    keys = np.array([10, 20, 30], dtype=np.int64)
    probe = np.array([30, 10, 20], dtype=np.int64)
    assert list(kernel.lookup(keys, probe)) == [2, 0, 1]


def test_lookup_marks_an_absent_probe_with_minus_one():
    keys = np.array([10, 30], dtype=np.int64)
    probe = np.array([10, 20, 30, 40], dtype=np.int64)
    assert list(kernel.lookup(keys, probe)) == [0, -1, 1, -1]


def test_lookup_tolerates_a_repeated_probe():
    keys = np.array([5, 7], dtype=np.int64)
    probe = np.array([7, 7, 5, 7], dtype=np.int64)
    assert list(kernel.lookup(keys, probe)) == [1, 1, 0, 1]


def test_lookup_against_empty_keys_is_all_absent():
    keys = np.empty(0, dtype=np.int64)
    probe = np.array([1, 2], dtype=np.int64)
    assert list(kernel.lookup(keys, probe)) == [-1, -1]


def test_lookup_of_an_empty_probe_is_empty():
    keys = np.array([1], dtype=np.int64)
    assert kernel.lookup(keys, np.empty(0, dtype=np.int64)).size == 0


def _refuse_search(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the keys were searched")

    monkeypatch.setattr(np, "searchsorted", refuse)


def _count_searches(monkeypatch):
    calls = []
    search = np.searchsorted

    def counted(*args, **kwargs):
        calls.append(1)
        return search(*args, **kwargs)

    monkeypatch.setattr(np, "searchsorted", counted)
    return calls


def test_lookup_reads_a_table_when_range_and_keys_are_at_most_four_probes(
    monkeypatch,
):
    # 300 keys over a range of 300 and 150 probes: 300 + 300 == 4 * 150
    keys = np.arange(100, 400, dtype=np.int64)
    probe = np.arange(50, 500, 3, dtype=np.int64)
    want = np.where((probe >= 100) & (probe < 400), probe - 100, -1)
    _refuse_search(monkeypatch)
    assert np.array_equal(kernel.lookup(keys, probe), want)


def test_lookup_searches_when_range_and_keys_exceed_four_probes(monkeypatch):
    # 300 keys over a range of 300 and 149 probes: 600 > 4 * 149
    keys = np.arange(100, 400, dtype=np.int64)
    probe = np.arange(50, 497, 3, dtype=np.int64)
    want = np.where((probe >= 100) & (probe < 400), probe - 100, -1)
    calls = _count_searches(monkeypatch)
    assert np.array_equal(kernel.lookup(keys, probe), want)
    assert calls


def test_lookup_table_covers_only_the_range_of_the_keys():
    # a table from 0 to 10**15 would not fit in memory
    keys = np.array([10**15, 10**15 + 2], dtype=np.int64)
    probe = np.array([10**15 + 2, 10**15 + 1, 10**15, 5, 2 * 10**15], dtype=np.int64)
    assert list(kernel.lookup(keys, probe)) == [1, -1, 0, -1, -1]


def test_lookup_accepts_an_int32_probe_below_keys_beyond_the_int32_range():
    keys = np.array([2**40, 2**40 + 1], dtype=np.int64)
    probe = np.array([7, 7, 7, 7], dtype=np.int32)
    assert list(kernel.lookup(keys, probe)) == [-1, -1, -1, -1]


def test_lookup_matches_a_dictionary_on_randomised_keys():
    rng = np.random.default_rng(21)
    table_cases = search_cases = 0
    for _ in range(400):
        low = int(rng.integers(0, 10**12))
        width = int(rng.integers(1, 2000))
        keys = np.unique(rng.integers(low, low + width, int(rng.integers(1, 300))))
        probe = rng.integers(low - 50, low + width + 50, int(rng.integers(0, 900)))
        at = {int(key): i for i, key in enumerate(keys)}
        want = [at.get(int(p), -1) for p in probe]
        assert list(kernel.lookup(keys, probe)) == want
        if int(keys[-1] - keys[0]) + 1 + keys.size <= 4 * probe.size:
            table_cases += 1
        else:
            search_cases += 1
    assert table_cases > 50 and search_cases > 50
