# Run562 independent Astra High review

Verdict: **source-only component PASS; live admission REJECT.** Candidate SHA256 `169a6ad7f77698c4c5172c2aa755130fe15e7426f610458532624e2efef87e49`. Astra independently inspected the revised source and reran `source_gate.py` inside `vllm-ascend26-dsv4f-w4a8`; exit code 0. No inference service was run and no source was installed.

The earlier rank-local pause-branch collective-order flaw and post-parking Host-mirror termination flaw are repaired in the source. The final delta checks the full SHA256 digest as eight 32-bit words, requires an initialized eight-rank default group, and covers continuation branch, final trajectory and group-size negatives. No new source-level blocker was found for this narrow component gate.

Live remains rejected because the outer RPC transaction must prove that all eight workers own the same method/generation and can fail closed on rank-local exceptions; this class has no actual normal API publication, client permit lineage, final Scheduler settlement or complete semantic fixed-`W₀` control. The exported sampled/count trajectory is only part of that control. Formal Current and every finite Bound endpoint remain unchanged.
