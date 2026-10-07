# H13 diagnostic v2 — staged only

V1 files/results preserved. V2 changes only the diagnostic, not production7bb35edf. Actual original prepare and runner snapshot bytes are retained verbatim as file prefixes; additions have consistent class-method indentation and no trailing whitespace.

Changes: byte comparison now transfers with detach().cpu() **before** contiguous()/view(uint8). Both banks must have equal descriptor sets and identical entry.input_addresses for each descriptor; capture witness records these lists. Returned output pointer sets must be disjoint; an alias rejects this diagnostic rather than being silently cloned or accepted. Capture/runtime witnesses include actual pid and TP rank for controller ownership binding.

Warmup layout witnesses record padded_num_tokens, replace_allreduce, tp_size/rank, fast_gate, hidden/router input geometry. Both modes must actually cover [2,6144]/[2,256], pad16, TP16, non-SP and their owned rank; merely observing mode0/1 no longer establishes applicability. Geometry/pointer failures abort before selector installation. Expected dynamic scope is still externally frozen by controller.

`check_production_equivalence_CPU.py` directly compares AST of the two helpers and prepare with production7bb35edf, normalizing **only the renamed prepare function identifier**. All three PASS; function bodies, docstrings, arguments and expressions are equal. This caught the earlier generated docstring indentation difference; v2 now uses the exact original method lines with proper class indentation rather than treating that mismatch as equivalent.

`check_shim_CPU.py` PASS additionally injects input-pointer mismatch, cross-bank shared output and wrong pad geometry: all rejected. It checks capture pid/rank and input records. Remaining tests/limitations from v1 apply: actual wrapper/runner/control AST, real mmap/filesystem, synthetic tensor/graph/native allocation. Numerical Torch CPU3888 is separate; this oracle does not prove numerical or device behavior.

Actual model source ends with `hidden_states, _ = self.norm(hidden_states, residual)` then returns hidden_states (Deepseek model1514–1517). No explicit fixed persistent output buffer is returned there. That supports—but does not by itself prove—private graph output under independent pools: the installed norm allocation implementation is not fully traced here. Output-disjoint assertion is therefore a **diagnostic admission condition**, not a universal model theorem. If it fails, preserve evidence and stop; do not change output ownership or relax it automatically. Identical top-level input addresses likewise do not prove every nested attention metadata pointer: existing persistent SFA contracts remain required.

No service/installation/device request. H12 physical observer cleanup is common baseline setup controlled separately, not an H13 gain. Candidate remains correctness-only until Root freezes any controller. Original Run277 failure stays unchanged.
