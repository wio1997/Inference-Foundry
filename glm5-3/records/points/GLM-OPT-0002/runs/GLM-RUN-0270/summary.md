# GLM-RUN-0270 — failed observer during memory profile

Native first failure08:28:41Z is the observer’s nonexistent `_EXTRA_CTX.num_actual_tokens`, called through `profile_run → drafter.dummy_run`, before graph capture or any candidate request. All16 capability witnesses were eligible/use_cuda_graph/async, which is only static capability evidence. Preserve original failed state/spec/raw; no performance result.

One planned H6/H5/target FULL/eager MTP recovery completed08:37:11Z with exact2334/8 golden, both roles health/idle/all16, H11off. CurrentNone. See [raw reduction](initialization_reduced.json), [guards](guards_after.json), [stack](retained_stack.json). Subsequent correction uses actual runtime kwargs rather than moving the missing field to another namespace.
