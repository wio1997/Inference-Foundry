"""Integer-only layouts for original request rows in a persistent NPU graph.

Physical state order is stable across replays. Original metadata order is
restored by ``output_rows``; inactive states have no query rows. Only the
reserved final state receives padding. No allocation or scheduling lives here.
"""
from dataclasses import dataclass


def validate_hidden_capacity(capacity, max_tokens, *, c2_logical_rows=None):
    """Check the existing graph/scratch geometry before allocating resources."""
    if (type(capacity) is not int or type(max_tokens) is not int
            or not 1 <= capacity <= max_tokens):
        raise ValueError('original hidden graph capacity must fit the cache context')
    if c2_logical_rows is not None and (
            type(c2_logical_rows) is not int or c2_logical_rows < 2 * capacity):
        raise ValueError('hidden graph scratch C2 plane is smaller than 2*capacity')


@dataclass(frozen=True)
class HiddenGraphLayout:
    capacity: int
    lengths: tuple
    starts: tuple
    cu_query: tuple
    chunks: tuple
    output_rows: tuple
    slots: tuple
    original_lengths: tuple

    @property
    def values(self):
        tokens = tuple(token for chunk in self.chunks for token in chunk)
        used = tuple(start + length for start, length in zip(self.starts, self.lengths))
        requests = tuple(slot for slot, length in enumerate(self.lengths) for _ in range(length))
        positions = tuple(position for start, length in zip(self.starts, self.lengths)
                          for position in range(start, start + length))
        return tokens + used + self.cu_query + requests + positions


def plan_hidden_rows(chunks, slots, starts, *, requests, capacity, max_tokens,
                     query_limit, vocab_size, noise_token):
    """Validate the entire host layout before staging or changing history."""
    if (any(type(value) is not int or value < 1 for value in
            (requests, capacity, max_tokens, query_limit, vocab_size))
            or requests < 2 or capacity > max_tokens
            or type(noise_token) is not int or not 0 <= noise_token < vocab_size):
        raise ValueError('invalid original hidden graph dimensions or padding token')
    if len(chunks) != len(slots) or len(starts) != requests or len(set(slots)) != len(slots):
        raise ValueError('original hidden graph requires distinct request/state rows')
    if any(type(slot) is not int or not 0 <= slot < requests - 1 for slot in slots):
        raise ValueError('original request cannot use the private scratch state')
    if any(type(start) is not int or not 0 <= start <= max_tokens for start in starts):
        raise ValueError('original hidden graph has invalid state positions')
    if any(not 1 <= len(chunk) <= query_limit
           or any(type(token) is not int or not 0 <= token < vocab_size for token in chunk)
           for chunk in chunks):
        raise ValueError('original hidden graph requires complete valid query prefixes')
    total = sum(map(len, chunks))
    if total > capacity:
        raise ValueError('original query rows exceed the selected graph capacity')
    if any(starts[slot] + len(chunk) > max_tokens for slot, chunk in zip(slots, chunks)):
        raise ValueError('original hidden graph query exceeds context capacity')
    rows = [()] * requests
    positions = [0] * requests
    for slot, chunk in zip(slots, chunks):
        rows[slot] = tuple(chunk)
        positions[slot] = starts[slot]
    rows[-1] = (noise_token,) * (capacity - total)
    lengths = tuple(map(len, rows))
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    output = tuple(row for slot in slots for row in range(cu[slot], cu[slot + 1]))
    # An empty DP still executes a real scratch token for the original sampler.
    if not output:
        output = (cu[-2],)
    return HiddenGraphLayout(capacity, lengths, tuple(positions), tuple(cu),
                             tuple(rows), output, tuple(slots), tuple(map(len, chunks)))
