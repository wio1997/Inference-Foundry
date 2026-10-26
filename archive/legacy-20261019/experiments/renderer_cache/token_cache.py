"""Bounded deterministic input-token memoization for a fixed tokenizer instance.

Experimental; not installed in the serving path. Callers must create a new
cache when tokenizer configuration changes and still construct per-request
prompt objects and run normal validation after a hit.
"""
from collections import OrderedDict
from threading import Lock
import sys

class TextTokenCache:
    def __init__(self, *, max_entries=128, max_bytes=128 * 1024 * 1024):
        if max_entries < 0 or max_bytes < 0:
            raise ValueError('negative cache capacity')
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._rows = OrderedDict()
        self._bytes = 0
        self._lock = Lock()

    def encode(self, text, kwargs, tokenize):
        # Offset and non-scalar option paths retain original execution.
        if (not isinstance(text, str) or kwargs.get('return_offsets_mapping')
                or any(type(v) not in (type(None), bool, int, float, str)
                       for v in kwargs.values())):
            return tuple(tokenize(text, **kwargs)['input_ids'])
        options = tuple(sorted((k, type(v).__name__, v) for k, v in kwargs.items()))
        key = (text, options)
        with self._lock:
            row = self._rows.get(key)
            if row is not None:
                self._rows.move_to_end(key)
                return row[0]
        ids = tuple(tokenize(text, **kwargs)['input_ids'])
        cost = (sys.getsizeof(text) + sys.getsizeof(key) + sys.getsizeof(options)
                + sum(sys.getsizeof(x) + sum(sys.getsizeof(v) for v in x) for x in options)
                + sys.getsizeof(ids) + sum(sys.getsizeof(x) for x in ids) + 512)
        if not self.max_entries or cost > self.max_bytes:
            return ids
        with self._lock:
            # Duplicate concurrent misses may finish in either order.
            if key in self._rows:
                self._rows.move_to_end(key)
                return self._rows[key][0]
            self._rows[key] = (ids, cost)
            self._bytes += cost
            while len(self._rows) > self.max_entries or self._bytes > self.max_bytes:
                _, (_, removed) = self._rows.popitem(last=False)
                self._bytes -= removed
        return ids
