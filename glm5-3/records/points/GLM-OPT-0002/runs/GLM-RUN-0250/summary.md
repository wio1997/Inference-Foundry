# Run250 — bounded MoE gather comparison, failed diagnostic gate

H4 only; [Reset166](../../research/decode_path_20261006/RESET166.md). Actual Zcode Job GLM53-GATHER-AB-20261006 handed off to the unique controller. Its initial running Result is not final execution success. Controller subsequently failed, preserved in `state.json`, `failure.json`, `compare.phase.json` and `compare.log`.

P Run249 untouched; D reloaded once with diagnostic selector, same DP1/TP16/PP1/EP16/DCP16, K1, eager, kernels, model and profiler-OFF. Candidate CRLF SHA26e52eef…; original54e8ac6d…; shim310c47b2…. Modes at globally idle boundaries A1/B1/A2/B2.

Short A1/B1/A2 medians265.398/240.723/256.630ms/token, identical8 IDs and chunks1/2/2/2/1,2334 external KV hits. Complete A1/A2 output74 tokens, B1 output8, all final answer2: unequal work, **MIXED**, no complete code E2E gain.

B2 warm request timed out180s. Native log rank12 worker319627 hit `assert equal and torch.equal(before, hidden_states)`. Fifteen other mode3 witnesses exist; rank12 witness is absent. The combined predicate does not identify output versus input failure. `torch.equal` is not bitwise and fails NaN self-comparison; finalize had already unpadded, so its output gate compared only one real row. Actual rank12 operands were not recorded. NaN false-negative is plausible, not proved for that occurrence. See `failure_raw_excerpt.txt` and independent gate review.

Controller finally restored original on-disk source, but resetting mode failed because the owned D request remained blocked. Recovery was separately authorized/executed as [Run251](../GLM-RUN-0251/summary.md), not a rewrite or silent retry of Run250. No PERF_KEEP, CurrentNone. 本阶段没有新增代码级性能 KEEP。
