# Run621 independent Astra High preflight

Verdict: **SCOPED LIVE-READY for one guarded descriptor-only acquisition**, using the revised controller below. This is not approval of typed row-generation/read-domain instrumentation, a scheduling intervention, a formal TPS claim, or a non-null Bound. No service, model workload, NPU experiment, patch install or restore was run during this review.

## Reviewed identities

- Patch: `scripts/loop081_cache_descriptor_patch_run621.py` SHA256 `8e7ecebb0214c25204fc332308d6cd11b8db74fdb3c98ec2cd11794fe71455dc`.
- Patch CPU selftest script: `dbeca7ce9ffad9b66938389b114c1a76ce04847a269f84cb9fed4710ed76a426`; JSON `a38f9c85a4151140fb07e2e7fabc2c91d7ca918cea75d3f0c29b8fc73fb8d8ce`.
- Nine-source patch check: `718921320c40de96385ffb1588a6b6124b0b51b98d2f01d32167292ac3650c04`.
- Revised controller `scripts/run_loop081_cache_descriptor_run621.sh`: `7b07871fbc6506d646f7c71cba9a750e6ed38e67cf98ac8028a5ea9b4f6fac7f`.
- Revised validator: `d1882b2f4f39609e6843a198d8b7ba83c0870e320af137ee6f3a2d7a1ea379be`; validator selftest script `3b6be4a633ee752ef1db5374b15ceb8b7363fbb1f30fa41e919a6a3326de4302`; 14-negative JSON `a6a44244fc20041ab0290a44958af81b13c09f5db290a4503cb3039d0cf7fa76`.

Independently reconstructed all nine patched source strings in memory and verified the serialized check SHA against the CPU selftest manifest. Independently executed the validator positive plus 14 negative fixture portion without its output-writing section: all passed. Revised controller passes `bash -n`.

## Execution and observer safety

1. Run621 adds no device tensor-value extraction, clone/copy, explicit synchronization, HCCL, model invocation or Graph operation. Its descriptor records retain integers/strings/lists, not tensor/storage references after the helper returns. Shape/stride/storage pointers and offsets describe the existing objects; no allocation-lifetime extension is intentionally introduced. Host Python allocations, storage wrapper construction, imports and descriptor lookup still cost time. Observer neutrality is not measured; this remains diagnostic.
2. `_extreme_served_cohorts` is updated before runtime construction/Run621 arm. The arm requires existing Product arm, cohort exactly 5, and `EXTREME_RUN621_DESCRIPTOR=1`. Controller completes/validates 48 warmup requests before creating Product arm, then sends the measured 48. Only handoff `_execute` entries with cycle index 64/65 append, with cap two. If the actual cohort/entry pattern differs, admission fails; no inference from wall-clock Host bins is needed.
3. Capture occurs **before** existing host-mirror commit/count-copy/common-metadata refresh. It identifies the source SWA cache owner at pre-Draft entry, not the actual scatter/attention ABI call. Common descriptors may describe the preceding metadata object state. No row values, freshness or first-read eligibility may be inferred from them.
4. The flush adds records to the existing Product payload after the existing `torch.npu.synchronize()` following runtime `.run()`. No new synchronization is inserted. Actual file I/O remains after that synchronization.
5. CPU fake selftest does not exercise the NPU-format branch. The actual serving container `vllm-ascend26-dsv4f-w4a8` was checked read-only: `_format.py` SHA `8abe7bea6817f25d2360aee783afae16fe7fe2aaa46a04e7b7b865e7f2039488` wraps original get_npu_format with an IntEnum conversion. Installed `libtorch_npu.so` SHA `83fb9a0eb249aef6bca7f8463047fcb4d3062cc4849f17ddf31a5e050c838842`: native entry 0x18e59c0 tails into 0x16d3950; ordinary NPU/storage path obtains storage descriptor via 0x182aaa0 and reads format at offset 280. Inspected ordinary native path has no device copy or synchronization call. This is supporting static evidence, not a full measurement of dispatch overhead or every possible subclass path. The helper does not suppress errors; unsupported real tensors fail the guarded run.

## Gates and corrections

The first controller revision omitted imported `loop081_dispatch_patch_run606.py` from SCRIPTS and did not bind current generated patch manifest to CPU selftest. Both are fixed: revised SCRIPTS includes the import, and the current check SHA must equal `source_manifest_sha256`. Revised validator also checks positive data pointer/nonnegative view offset and compares both across cycles; the two new negative fixtures reject changes.

Run606 controller reuse is valid only through this reviewed Run621 adaptation: fresh run621 output/state, exclusive lock, Run621 tag and environment, unchanged guarded 48+48 product, exact 96 POSTs, Basis then Product then dispatch then descriptor admission, and owned-process stop. All eight rank/cohort5 files must have exactly two ordered cycle64/65 records, expected layer43/44/45 cache tensors, and descriptor checks. Missing/non-tensor cache or missing metadata is rejection, not zero traffic.

Install checks original source hashes and saves all original bytes/manifest before atomic writes. Restore requires verified process/service/NPU stop, valid backups/path/source set, and current source matching original or patched SHA. Controller cleanup compares all nine original source hashes and script hashes afterward. `--offline-confirmed` alone is only caller assertion; do not bypass controller stop/idle gates. Any failed stop/restore/source comparison invalidates completion and requires explicit recovery; never overwrite unexplained drift.

## What a pass establishes / does not establish

A pass establishes two observations of cache-owner/storage/view descriptors and format in the actual guarded workload, across all eight ranks. Equality of owner id, storage cdata/base/view pointer and geometry across two observations does **not** prove uninterrupted allocation lifetime: allocator/address reuse between observations remains possible. Fixture name `storage_generation_changed` tests cdata inequality only, not a true generation counter. Device identity, metadata content and common/state eligibility remain facts to inspect in raw records; current validator is not a complete typed ABI/schema proof.

No actual scatter/reader tensor pointer, writer epoch, row version, metadata snapshot, loaded attention-binary binding, native-task flow, first physical read or compulsory serial edge is established. No fresh-W-minus witness or absence follows. Successful acquisition can narrow the practical observer design by revealing actual cache format/geometry/ownership, but supplies no wall-time elimination credit.

**Strict Resource floor = null; strict Scheduling floor = null; Product E2E Bound = null; numerical Current-to-Bound gap = null. Current Formal remains 571.681 tok/s.**
