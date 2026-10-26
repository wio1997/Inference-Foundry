# Run618 independent oracle revision review

**SCOPED PASS for the revised pure row mapper and20 fixtures; overflow/type admission is not yet complete. NOT LIVE-READY.**

Reviewed script SHA256 5170085fec8efa2091e26582401834ff3fb5ee11e24fee258ab418ab4730679f; output SHA256 053b5129335be8edb5f034e5a2f5427f87f1b4226af0e310a2ae6e3840984777. Independently executed CPU logic with only top-level output-write/print calls removed. The reconstructed result exactly equals saved JSON:7 arithmetic successes and13 expected rejections. Run617 original output SHA remains4a89f21c379d484d5c22bf203fe078b4112950d7d39394634e91543faf10ba65, confirming prior evidence was not overwritten.

Corrected issues: tiling.h is pinned and sparse width2176 is rejected against2048; actual block-table presence is required in both branches; sparse floats, bools and out-of-int32 values are rejected; scalar inputs are bounded integers; row byte ranges must fit both storage and signed64 bound. The scope now explicitly says pure mapping helper, selected tiling guards and toy geometry, not complete ABI validation. Dense/sparse formulas and duplicate-preserving output are unchanged and correct within the declared supported subset.

Two residual limits prevent saying “all type/overflow issues are fixed”:

1. block_table entries are checked only for Python int type, not signed-int32 range. Independent challenge kv_len10/q_len2/query0/block_size4/table[0,2**40,2]/capacityNone returned[4398046511105,4398046511106,4398046511107,8]. Such a table cannot represent the native int32 payload. Require signed-int32 values for every supplied entry; selected negative blocks remain invalid. If this helper deliberately delegates that to a later invocation schema, record that delegation explicitly and do not call this a complete payload guard.

2. Bounded scalar operands do not imply bounded C++ intermediate sums. For the existing base fixture with win_right=INT32_MAX, the helper returns[13,14,15,36,37], while K-Q+j+win_right exceeds signed-int32 before the native Min. The source contract cannot be extended through signed overflow using Python arbitrary-precision arithmetic. Reject dense inputs whose corresponding intermediate expression is outside the native signed range, or explicitly restrict the declared invocation subset to a proved safe range. Add a boundary-negative fixture. Current realistic32K/window values are not implicated by this challenge.

The final byte-address bound is conservative and sound for positive admitted geometry; it does not fix earlier sequence arithmetic or table-payload representation. Optional slot capacity continues to mean symbolic mapping only until actual cache capacity/view bounds are supplied. The module intro's “set” is a minor naming mismatch: it returns an ordered list with multiplicity, as required.

No new excessive performance/Bound claim is present in the JSON scope. Metadata content/coverage, live row generations, exact optional-argument branch, supported dtype/head/block geometry and loaded binary/tiling joins remain external gates, as Run616/617 state. The actual-W0 eligible rows, physical HBM bytes and strict Resource/Scheduling floors remain null. Formal Current571.681tok/s is unchanged.
