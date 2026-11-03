#!/usr/bin/env python3
"""CPU address arithmetic for splitting a row-major two-projection output."""
T, width = 2, 512
row_stride = 2 * width
for name, base in [('kv', 0), ('score', width)]:
    expected = base + row_stride  # address of [1, 0] in a column-split view
    packed_assumption = base + width  # existing kernel's row * HEAD_DIM rule
    assert expected != packed_assumption
    print(name, 'view_row1_col0=', expected, 'packed_kernel_row1_col0=', packed_assumption)
assert T * 1024 * 4 == 8192
