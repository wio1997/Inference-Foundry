# GLM-RUN-0008

REJECT this deployment under the legal mixed E2E contract. Both restarted engines (Graph/MTP3, budget16384, gmu0.92) passed local334→128; mixed P requests128+16 completed and expected400 passed, but all three D requests failed with HTTP200, native in-band SSE InternalServerError500, DONE, no usage/finish. Four valid inference requests total400 outputs; failed D partial chunks receive zero effective token credit. Native DCP recv allocation in MTP merged draft failed520MiB on NPU14 (D_failure.log4918/5012/5014), cascading to EngineDeadError. Allocation location is established; complete memory-pressure causality and exact tensor shape remain unknown.

Capability phase exit1, no timeout; controller terminal failed/dead, gateway stopped -15 with six acquire/release trace pairs. In-band engine failures were missed by router backend_failure telemetry; all three leases falsely marked false. This is a control-plane observability/fault-isolation gap to fix while preserving raw response bytes, not a native request success.

Diagnostic failure is real evidence, not an illegal test contract; canonical manifest verdict REJECT distinguishes the unchanged capability_summary's generic INVALID. No formal KEEP, SLO or hardware capacity claim. New Run9 retains verified idle P and restarts only D at known functional budget128; no old queue replay.

Raw logs/source snapshots and original Zcode reduction remain on server. GPT audit clarifies target-global eager versus draft enforce_eager and text chunks versus tokens. See reduction and manifest artifact hashes.
