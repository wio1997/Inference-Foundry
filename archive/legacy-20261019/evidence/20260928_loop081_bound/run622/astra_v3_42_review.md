# V3.42 independent Astra High dual-Bound review

**SCOPED PASS for observed geometry and conditional payload arithmetic.** No strict Resource/Hardware, Scheduling/Execution or Product endpoint is established. This is not approval of a new live observer. No service/NPU work or model/source edit was performed by this review.

## Final identity and fail-closed checks

- Final generator SHA256: **6b842763f2581d61975cfe65392bade4861bf4e197180fc6ce0bcfc02f0108a7**
- Output SHA256: **41b52c3dd7b63924e930b3f0e56bb0f9f4289cb10a8ef63105b5d547a6041c02**
- Model recheck JSON SHA256: **4a3e3626077c055c5ecd5d6fa67cbe174d2b33f839a7a2d952ce2586597c25dd**
- Geometry recheck JSON SHA256: **40f49381e26b4ab50dbdbb9869b1c06b1514a95446d2c85a3503c61a9d6c2af1**

All five input pins matched. Executing the generator with only final filesystem write/print omitted reproduced the saved JSON byte-for-byte. All inherited V3.41 fields are unchanged except model_revision.

Initial generator6b65d466 checked only four flat strict endpoints. Review identified unguarded new nested aliases/certificates. Final6b842763 adds the nested null checks; output is unchanged. Independently injected1.0 into each of **four flat plus nine nested** floor/ceiling/gap, compulsory-byte, capacity and DAG-placeholder fields. All13 mutations were rejected with the expected endpoint guard. No original or revised output actually contained a promoted endpoint.

## Independent all8 geometry

Rehashed all six Run621 admission links and the eight raw cohort5 Product files against Product admission. Recomputed **16 snapshots /48 cache observations**, exact rank/cycle64–65/layer43–45 coverage, device/dtype/format/shape/stride, view offsets, descriptor-to-raw identity and within-rank storage-interval non-overlap.

| Quantity | Independent arithmetic / result |
|---|---:|
| Each observed BF16 ND cache |34090×32×1×512×2 =1,117,061,120 bytes|
| Three caches per rank |3,351,183,360 bytes|
| Sum of all8 observed storage capacities |26,809,466,880 bytes|
| One logical row |512×2 =1024 bytes|
| Two cycles × three layers × one row |6144 bytes|
| Hypothetical pre/post copies |12,288 bytes/rank|
| Block-table logical view / backing storage |1,572,864 /2,097,152 bytes|
| Slot-map logical view / backing storage |384 /33,152 bytes|

The two samples match the admitted descriptors. Matching samples do not certify uninterrupted allocation lifetime; identical virtual addresses on different ranks do not imply shared storage. Common slot and state target-slot descriptors match at those samples, not necessarily at native invocation or after metadata refresh.

## Resource and Scheduling interpretation

V3.42 correctly calls these observed **tensor storage capacities**, not physical allocator reservation, compulsory HBM traffic, memory bandwidth or per-cycle bytes.12KiB is hypothetical selected-row retained payload; it proves neither valid selected rows nor full observer cost or available live memory under the128KiB operand cap. Metadata/generation tags, copy temporaries, buffering and alignment remain separate obligations. The full block-table view exceeds that cap and selective capture must follow the actual ABI/branch; storage size alone cannot establish that selection is sufficient.

Descriptors were captured before common metadata refresh. No device slot/index values, row contents, last-writer generations, eligible-reader execution coverage or actual native reader identity follows. Consequently no legal overlap, mandatory serialization, critical-path floor or fresh-W-minus claim is made.

The model keeps both flat and nested Resource, Scheduling and Product endpoints null. Actual selected rows and total observer cost also remain null. Run621's diagnostic604.151tok/s is not promoted or transferred to Run99. Formal Current stays **571.681tok/s**; numeric Current-to-credible-limit gap remains null.

## Next evidence and limitations

The recommended bounded actual-invocation snapshot is consistent with the evidence: preserve exact ABI arguments, branch-dependent selected metadata and valid cache row pre/post generations on original streams; join within the same acquisition to native attention and predecessor/successor. It still requires independent implementation preflight, correctness and cleanup. Physical reads, mathematical necessity and exact-board cumulative C-plus/B remain separate certificates.

No remaining model-update blocker found after the nested-gate correction. Geometry uncertainty was reduced; the trustworthy whole-system performance interval was not numerically narrowed by this update.
