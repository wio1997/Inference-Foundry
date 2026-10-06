# Reset167 — repair native correctness evidence and restore owned D

Same branch/HEAD/rules as Reset166. H4 remains the only candidate. Run250 B2 rank12 combined gate failed; source restored, owned D request1 remains blocked. Short matched A1/B1/A2 signal is not PERF_KEEP. Natural-complete output74A versus8B is MIXED, not code E2E gain.

Distinguishing missing evidence: native HCCL full-output raw-byte/input preservation including padding NaN/Inf/signed-zero; old torch.equal gate cannot distinguish NaN self-inequality. Existing source explicitly rejects strided stock inputs, so micro stock reference normalizes those inputs; candidate normalizes internally. Corrected model warm gate compares full padded output before unpad, separately records input bytes/NaNs/format and collectively reduces status before any rank throws.

Hypothesis: all_gather_into_tensor preserves actual HCCL payload; failure may be diagnostic NaN false-negative, not established. Minimal standalone16rank HCCL20cases/rank, no weights, tiny tensors, no profile. Existing unique controller retires only failed owned Run250 D, leaves P unchanged. If micro fails -> stock model recovery only; no further performance. If micro passes -> one recovery D load with diagnostic selector, stock default; four modes with warm8 +2 timed8 +one complete arithmetic thinking_token_budget0, max96 natural EOS (same dynamic API request each side, not a config scan). Count all errors and outputs; full-output trajectory must match before E2E gain. Finish stock on-disk source and mode0.

Decision table: full native byte mismatch -> REJECT and restore stock; byte pass but output trajectory mismatch/perf unstable -> INCONCLUSIVE/MIXED, no Current; repeatable short and matched complete gain -> scoped code evidence, full API/stable SLA still unaccepted and product CurrentNone. Recovery itself is no performance gain. No large workload/new profile/parameter scan. Astra independent gate review supports diagnostic repair; NaN explanation remains unproved for the actual old rank12 values.

本阶段没有新增代码级性能 KEEP。
