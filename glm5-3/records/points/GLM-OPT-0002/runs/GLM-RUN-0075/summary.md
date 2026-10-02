# GLM-RUN-0075

REJECT: native DSA-CP + SP + DCP16 candidate failed in profile_run, before KV residency/Graph/health or public inference. 0 requests, 0 effective public outputs. First native 507034/vector timeout at prepare_inputs_event synchronization; exact-PID device logs identify fault MoeDistributeDispatchV2 with concurrent aiv_all_gather_bfloat16_t. This localizes the failed chain, not its unique cause or the hardware limit.

Two CPU fullCLI/backend/source/SDK0 checks and exact D69 API2/NPU32 cleanup completed. D75 API2/workers subsequently exited natively; readonly v2 audit verifies both NPU sets empty and original controller dead, no additional signals. Terminal audit VALID23742B SHA95b24bd0f41d4a88d3abf11fd10e004d85378ed554656a26c54fac309a604e88; exact-PID resource/driver v2 VALID29068B SHA4f6bdfe0e0bead5b9334a36effefb7db501d13db4be63029f86032bec192da80. Progress sampler third sample lost its live API and produced no Result: INVALID, two earlier raw samples retained. First driver sampler substring matched old filenames via timestamp: scoped reduction INVALID, corrected v2 retained.

Run76 changes only shared-expert multistream overlap to false in the native GPU config; new cohort/engineIDs/cache salts. No native operator/math edits. Current remains None; no DSA-CP gain/fullAPI/capacity/KEEP claim.
