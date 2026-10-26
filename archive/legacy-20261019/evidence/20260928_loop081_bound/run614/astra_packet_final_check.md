# Run614 revised packet final design check

**SCOPED PASS for the revised design wording; NOT LIVE-READY.** This is a read-only review of script SHA256 f5e2074f1ac18f37928ed6d8f4aee22562faedf94e8e0d64d9429865297d0483 and gate SHA256 73ed975c9bab9eb47ea877741f223e62e9eda1de94c56d76010ef259c3d85f79. All six evidence input hashes were independently rechecked.

The central wording problems are corrected: allocation lifetime/content/row-writer generations are separate; snapshots must follow existing producer/consumer stream order before overwrite; immutable submission ownership must join native flow rather than Host-cycle bins; the read domain is explicitly ABI-semantic; fresh-work admission is separate from Scheduling identity; valid reuse or query overwrite can preserve a PASS Scheduling packet while yielding NO_WITNESS/UNKNOWN for conditional W-minus.

Three interpretations must remain explicit when implementing these fields:

- selection.fallback NO_WITNESS applies to the fresh unoverwritten context-row witness, consistently with bound_admission.scheduling_packet; it must not discard otherwise sound scheduling evidence.
- Validating the loaded sparse-op implementation is necessary for a physical first-read claim, but is not by itself sufficient to measure the first load timestamp or compulsory HBM traffic. The currently planned metadata establishes only the logical ABI read domain. Keep physical timing unknown absent its own evidence.
- The existing predecessor/successor and “any mismatch INVALID” requirements retain the original preflight's meaning: actual token/state/value lineage, not mere markers; semantic mismatches under a declared same-state comparator, not natural independent-run acceptance/hash differences.

The earlier six-part review remains an implementation gate (astra_packet_preflight.md SHA256 a26d3d4ccb21b2a096cba3cab00855f068b61e2bc902adedca33c09125f97faa). Concrete storage-lifetime acquisition, metadata content versions, loaded ABI semantics, predecessor ENTRY_UNKNOWN handling, exact request-row/dispatch identity, bounded metadata byte/count limits, overflow behavior, negative fixtures and the same-state correctness comparator are still to be implemented and reviewed. Merely listing these fields does not make them measured. In particular, broad entry-credit completeness cannot be obtained by snapshotting one cache.

No live service or NPU workload was run. The revised gate retains design_not_live_armed status, formal Current571.681 tok/s and all six strict traffic/capacity/endpoint/gap fields null. It introduces no numerical Bound or performance gain.
