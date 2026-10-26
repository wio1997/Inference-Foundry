# Run557 — warmup completed, original controller invalid at client scope gate

Status: **INVALID for the planned 48+48 acquisition.** The original controller returned exit1 after all48 warmup requests completed, before phase transition to measured. No measured cohort5→6 exists in this Run; no current→Bound cost or TPS may be inferred. Cleanup stopped the service, restored the stage-two and stage-one patches, matched all borrowed source/script SHA files and verified eight NPUs idle.

The direct cause was an exact client-summary `scope` string mismatch: `loop080_frontier_client.py` appended the c12 release-lineage qualification, while the reused Loop079 client validator still required the old string. The old validator rejected at `admit_phase` before inspecting the otherwise complete request files. The Run remains invalid even though a corrected offline replay passes.

After the run, `loop080_frontier_phase_validate.py` was copied from the pinned old validator with only its three exact scope strings changed. Against the saved Run557 warmup data, the new validator admitted48/48 requests, `loop080_frontier_client_validate.py` admitted36 conservative c12 eligible-release sets with peak12 and zero unique-parent certificates, and the phase barrier replay admitted32 rank/cohort records. These posthoc checks isolate the integration bug; they do not fabricate a missing measured phase. Astra independently reviewed the exact diff and the revised Run558 controller before retry.

Formal Current remains 571.681 tok/s; strict Resource/Hardware, Scheduling and Product endpoints remain null.
