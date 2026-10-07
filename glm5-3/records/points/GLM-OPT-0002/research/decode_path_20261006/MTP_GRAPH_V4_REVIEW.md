# Independent final MTP Graph v4 combination review

2026-10-07. Offline only; no service/run/candidate or frozen-run edits.

Verified actual files:
- proposer: `19d31822fbf2ceff0ac62d1ee241a91de5813314a5f193fa12798dffc73b7f96`
- runner: `4c90054ad40f4c430c2562bc1603aacb497684429988bf1a60edce585d6d2923`
- observer v5: `dfb76312046d6e8bcf4f353e86de134f0ca73f004419f29ba11e17a3056069c0`

## Verdict

**No new concrete source blocker found within the exact-NoopOffloader scoped path.** The new guard closes the unresolved non-noop offloader applicability condition from v3; it does not make the wrappers' replay fences equivalent.

Personally checked v3→v4 proposer and observer-v4→v5 diffs. Production change is the actual NoopOffloader/get_offloader import and exact-class static predicate; runner remains the previously reviewed single-outer-wrapper change. Observer adds only corresponding guard/import and actual offloader class in capability evidence.

## Constructor/lifecycle proof

Actual runner creates and installs its configured offloader at line368 (`set_offloader(create_offloader(self.offload_config))`), before drafter construction at514. Therefore the predicate at proposer init observes the configured worker singleton rather than merely assuming the module's initial default. Later load_model calls `get_offloader().post_init()` at4164; inspected BaseOffloader.post_init returns without replacement, and NoopOffloader inherits it.

Actual `offloader_sources/vllm_base.py`: exact NoopOffloader only returns modules unchanged; inherited sync_prev_onload and join_after_forward are no-ops. `type(...) is NoopOffloader` correctly rejects subclasses that could override those methods. It preserves native eager behavior for excluded configurations instead of silently removing a non-noop dependency.

The static gate is a worker-lifetime assumption, not protection against arbitrary external singleton replacement. No runtime replacement path was found in the inspected runner after line368. All-rank capability class evidence confirms the state again after loading. If offloader replacement is introduced later, revalidate/revoke the graph capability at that boundary; do not claim the present patch supports it.

## Fence distinction retained

Outer ACLGraph retains its capture-before offloader sync and capture-after join; its replay lacks Breakable's pre-replay offloader sync. V4 is safe to investigate because the admitted exact no-op implementation gives that absent call no dependency to enforce. Neither implementation nor review claims the fences themselves are equal. Target wrapper, other drafters, CPU staging/count/KV/error ordering and H6/H5 remain unchanged.

## Test interpretation / next boundary

Read CPU results:18 static exclusions including3 non-noop/subclass counterexamples,1024 runtime,19 pointer and2 constructor cases; nested24 scope/2 order cases; observer16 lifecycle/9 fallback/4 dummy cases. They support class gating, host composition and observer control flow. Model/graph/allocator remain doubles; they do not prove capture, HCCL or numerical output.

The next meaningful question remains the single bounded device capture/replay correctness diagnostic under the frozen exact-noop scope after owned recovery/reset. No performance Run, parameter scan or automatic retry follows from this review. Run271's nested-capture failure is preserved as evidence; no claim that it proved model numerical failure. This is a source review and route recommendation, not deployment authorization or PERF_KEEP.
