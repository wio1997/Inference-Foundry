# Run281 — boot preparation failure

Both candidate and the single bounded recovery failed before ACL initialization/model loading because the epoch-local `runtime_bundle/native_acl_lifecycle.py` was omitted. Zero PD requests. Failed raw/spec preserved. The recovery restored prepare54e/runner4c/RoPEb6 source but did not establish a healthy D instance; actual terminal stack is unknown, not ON. P249 unaffected. H13 device legality/performance not tested.

This avoidable controller preparation failure is fixed offline in fresh Run282 by copying/pinning the original boot entry into both epochs and validating it before source/device actions; Run281 is never replayed. H6/H5 previous performance evidence remains retained.
