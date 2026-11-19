"""Plan ratio-one causal LightningIndexer groups for a contiguous C2 request."""

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryGroup:
    first_query_row: int
    query_rows: int
    key_prefix_rows: int

    @property
    def query_slice(self):
        end = self.first_query_row + 2 * self.query_rows
        return slice(self.first_query_row, end, 2)


def c2_causal_groups(prefix_tokens, query_rows):
    """Cover every row with visible keys using at most two gather/call groups.

    The input represents absolute positions prefix_tokens through
    prefix_tokens + query_rows - 1 for one request. The K source is
    already compressed in the same order as the original C2 scorer.
    """
    if type(prefix_tokens) is not int or type(query_rows) is not int:
        raise TypeError('Prefix and query lengths must be integer token counts')
    if prefix_tokens < 0 or query_rows < 0:
        raise ValueError('Prefix and query lengths must be nonnegative')

    groups = []
    for first in range(min(query_rows, 2)):
        count = (query_rows - first + 1) // 2
        last_row = first + 2 * (count - 1)
        keys = (prefix_tokens + last_row + 1) // 2
        # The parity subsequence increases the visible K length by one per
        # query. A leading row with zero visible keys can stay all-invalid.
        visible_rows = min(count, keys)
        if visible_rows:
            groups.append(QueryGroup(first + 2 * (count - visible_rows),
                                     visible_rows, keys))
    return tuple(groups)
