# Initial P prefill-MC2 startup failure

The four P node logs in the initial startup/profile attempt all reported the same failure. The dispatch operator failed during tiling because the EP communication buffer was too small: the operator calculated a 3,297 MB requirement, while the reported HCCL/CCL buffer was 400 MB. The error was `561002` from `npu_moe_distribute_dispatch_v2` calling `aclnnMoeDistributeDispatchV4`, followed by an HCCL window-size check failure.

The failing shape/configuration was `maxBs=512`, hidden size 6,144, EP64, four local routed experts, zero shared experts, and top-k 8. The captured logs contain 33 matching occurrences per node log; these are repeated log lines, not 33 separate startup attempts.

The installed `model_runner_v1.py` `profile_run` entry at lines 3714–3730 runs a prefill dummy forward at MC2 capacity when the selected method at that capacity is MC2/FUSED_MC2 and the runner's maximum token count exceeds that capacity. This places the failure during initial profiling, before ordinary serving requests.

The evidence establishes that the initial 400 MB buffer was insufficient for this shape. It does not establish a general EP64 MC2 incompatibility, nor does it establish that a later 4,096 MB `HCCL_BUFFSIZE` setting will succeed or what KV capacity will remain afterward. The later startup attempt is outside this evidence record.

Source archive SHA-256: `d29966c1548d987b34e0c7fd5d8c291480d7f504ae7f17fa85d0652154611e5e`.
