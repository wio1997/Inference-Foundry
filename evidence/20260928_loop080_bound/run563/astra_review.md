# Run563 independent Astra High source review

Verdict: **source-only patch gate PASS; live controller not yet admitted.** Astra inspected the source patch, identified three initial restore/provenance gaps, then independently verified their correction. It reran the temporary-copy `patch_gate.py` with exit0. Reviewed patcher SHA256: `5a0e38402b5a0e6d1f7133c3d61079f53da85326e4dbe5b859aff5798923e9d3`. No production source was installed and no inference service ran.

The patch matches fixed original SHAs, compiles proposed files, uses unique anchors, and restores only fixed source paths/original bytes after checking known patched bytes and candidate/patcher/helper identity. Every restored file is rehashed before success is recorded.

The proposed selected-cohort A0/B/A1 test can falsify an unchanged observed trajectory. It cannot certify complete fixed `W₀` from the existing Runtime trace alone: KV and full mutable state, Draft context, request semantic-key/entry-state alignment and all8 client admission still need evidence. B's extra HCCL/Host work prevents attributing timing deltas to pause alone. The live controller requires offline proof, locking, timeouts/cleanup, all8 complete files, exact client correctness and fixed-source restoration before any acquisition.
