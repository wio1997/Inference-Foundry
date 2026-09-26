# Astra High independent Loop064 review

Scope: read-only review at repository HEAD 07cb354 with Run280 immediate source installed. No source, launcher or service was modified; no NPU work was executed. This file is an independent review input for Sol, not a KEEP decision.

## Facts and implementation assessment

The frozen enable_dsa_cp path uses AscendDSACPImpl._forward in attention/context_parallel/dsa_cp.py. Its c4 indexer cache is updated before the fork. Main compressor/scatter reads gathered hidden and its own state/cache/metadata; indexer query/QLI reads local hidden/qr and indexer key/scale cache. Neither branch consumes the other's output. Sparse attention is the real join. Source-level semantic independence is high confidence; native workspace/Graph allocator independence is not yet proven.

The installed patch now excludes get_forward_context().is_draft_model, requires model.layers.<0..42>, and scope=one selects layer 2, the first c4 layer in the frozen config. Aux waits for main before compressor; main joins aux before sparse attention. No new HCCL call or source-level collective reorder is introduced. Local keepalive retains branch outputs and compressor metadata through the join. The once-per-layer visible storage audit is useful but does not prove native workspace or capture allocator non-aliasing.

Immediate reverses the original serial order: main compressor then indexer query/QLI. Overlap uses the same launch order and aux stream, moving the main wait to the real attention join. Immediate versus overlap therefore tests this synchronization edge. Original A0 versus immediate includes order/cache/stream effects; original A0 versus overlap tests net product value. Do not attribute A0-to-immediate solely to stream overhead.

## Run280/281 evidence requirements

The current launcher performs warmup48 and bench12 with one-use alias logs. Successful exit, exact output length and runtime gates establish functional execution and readiness only. They do not establish same-state numerical equivalence, actual replay overlap or a Product E2E performance gain.

Each mode needs independent capture identity: source SHA, mode/scope, target layer/rank, FULL Graph replay count and actual selected branch. Python logs can come from warmup or capture and cannot prove replay execution. Device evidence must show the selected compressor/scatter and indexer QLI on their expected streams and show the dependency before sparse attention. Compare fork-to-join, complete Target/cycle and next collective arrival across all eight ranks. If the two tasks remain serialized because of AICore or scheduler constraints, record this result rather than inferring concurrency from stream assignment.

Correctness should execute the candidate outputs, use the established same-state A/B/A KV-restored control and self-replay noise floor, and include actual main/indexer write pages, TopK, attention/logits and acceptance/counts. Visible buffer alias checks cannot replace these gates. Timing must disable heavy checks and profiling; one-layer cross-service TPS differences are too trajectory-sensitive to identify this small intervention. Do not multiply a one-layer observation by 21 without all-layer measurement. Keep the running Run280 unchanged and classify its evidence by the measurements it actually produces.

## Resource and scheduling bounds

The latest model correctly separates a physical relaxation using defensible capacity upper bounds from an attainable engineering allocation demonstrated under measured concurrency. Current task bytes, advertised capacity, achieved microbenchmark bandwidth and profiler HCCL occupancy are different quantities. Run267 arrival skew makes first-collective task duration unsuitable as transport cost. Run266 union/gap values remain observed schedule descriptions, not removable time.

One additional DAG refinement is necessary before calling the nested graph a semantic Scheduling-aware lower bound: q_local currently bundles q_a -> qr_local -> q_b -> RMS/RoPE, but indexer query only needs qr_local, not the completed main query. Likewise the indexer_qli node bundles indexer query/weights preparation with QLI; only QLI's key read must wait for indexer_cache. Treating the entire bundled nodes as required predecessors adds false serialization to an unrestricted ideal schedule. Split qr production, main-query completion, indexer-query preparation, indexer-cache completion and QLI. The existing coarse DAG remains valid as a model of the restricted current/prototype schedules, if labeled accordingly. It must not close broader scheduling space.

For this restricted fork, the no-contention local screen is delta <= min(T_indexer_branch, T_main_compressor_scatter), using same-path branch spans and comparable resources. The real executable gain also includes contention, event overhead and rank rendezvous. Run256's 1.283ms sum for the 21 c4 main-compressor tasks is a descriptive scale, not an exposed-saving or rigorous scheduling bound. No numeric TPS ceiling is established.

## Algorithmic cycle-count audit and correction of initial concern

A generic fixed-cohort count bound should use remaining outputs, not automatically all client outputs. For cohort c and slot i, let R_ci=max(0,1024-I_ci), where I_ci is the count already produced before this Runtime. Since one target cycle emits at most 8 useful tokens per slot, C_c >= max_i ceil(R_ci/8), and C_total >= sum_c max_i ceil(R_ci/8). This ignores parking delay/acceptance loss and is deliberately loose. The weaker total-output equivalent is ceil(sum_ci R_ci/96).

For the actual frozen Run99 path, the existing number 512 IS supported; the initial concern is resolved by evidence, not by changing that number. All 96 measured rank/cohort records (rank0..7, cohort5..16) have handoff_before_model_forward=true, generated_output_counts=[1024]*12 and pass=true. Rank0 cohorts5..8/9..12/13..16 sum to 1217/1212/1206 observed cycles; each group's loose formula gives 512. Current borrowed model_runner_v1.py lines3374-3394 enforces max_tokens1024/greedy/ignore-EOS and passes initial_output_counts=[0]*12 to FixedCohortServing. runtime/fixed_serving.py defines remaining=1024-initial and checks generated==remaining. Thus all 49152 useful outputs belong to these Runtime cycles on this path. Clarify this premise in the executable model rather than declaring 512 invalid. The count bound says nothing about attainable acceptance or TPS.

Evidence: evidence/20260924_loop036_metadata/run99/runtime/rank<0..7>_cohort<5..16>.json; runtime/fixed_serving.py; borrowed worker/model_runner_v1.py. This review did not rerun Run99.

## Next highest-information action

After Run280, the next independent-capture overlap diagnostic is justified for feasibility. Before any formal performance claim, obtain candidate-consumed numerical evidence and a narrowly scoped device timeline distinguishing (a) legal actual overlap, (b) stream-assigned but serialized execution, and (c) overlap offset by resource contention or late-rank waiting. That discriminator decides whether all21 layers are worth testing. If branch contention defeats this fork, the next source-backed choice is local-query versus hidden-AllGather overlap, or splitting qr/main-query/indexer-query as above; neither requires assuming the current whole operators are semantic barriers. Sol retains route and KEEP/REJECT/PIVOT authority.

Confidence: semantic branch independence and frozen Run99 count accounting high; stream/Graph/workspace safety pending actual replay; net Target/cycle/E2E gain unknown; Hardware/Resource and unrestricted Scheduling-aware numerical ceilings unknown.
