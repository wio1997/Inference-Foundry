# C32 native mapped-host IO and MemCache integration

Deployment remains based on the Ascend documentation reviewed in
`results/memcache-docs-L09-tested-runtime-reconciliation-v6.json`: Meta starts
first, separate contribution processes provide the pool, and model clients use
`dram_size=0GB`, `max_dram_size=160GB`, and `device_sdma`. The SDK is 1.3.0.
The four providers still contribute 136, 112, 160, and 160 GiB. The deployment
guides establish this process/configuration pattern; they do not establish
compatibility between SGLang's native host allocator and its storage wrapper.

The first native candidate failed during P startup. Offload had already mapped
its entire host slab into the SDK global address manager. SGLang subsequently
called MemCache `register_buffer` on its K subrange; this was a different mapping
from the existing slab, and the SDK rejected the overlap. Both failures and the
tested L10 recovery are retained. No end-to-end result was accepted from that
candidate. Identical mapping replay is permitted by the SDK; partial subrange
registration is the specific failing case here.

The prerequisite investigation used the exact shipped SDK source, commit
`a085a7c905a51e8cd55c2ef16c74ffe95ded31bc`. LOCAL Offload uses an exclusive first
2 TiB virtual-address region beginning at `0x280040000000`. Normal MemCache
storage allocations use the following region. The combined native fixture
checked those address bounds and linear DVA translation before reading data.
An ordinary pinned buffer retained standard registration and passed its control
roundtrip. Thirty native mapped objects passed two complete 71,024,640-byte
storage roundtrips around 82 native copy calls. Ten complete BF16 H2D/D2H tensor
comparisons and the single-layer isolation check passed. All 31 owned test keys
were removed, SDK close returned zero, and the original processes remained.
See `results/C32-L12C-combined-MC-native-mapping-reuse-qualified-v1.json`.

`glm53_C32_native_MC_mapping_reuse_v1.py` changes only the storage wrapper's
registration of three exact native host buffers. It requires P34/35, explicit
feature flags, the pinned original wrapper source, pool object identity,
page-first split layout, page size 128, contiguous CPU BF16 storage, 634 pages,
79 packed K/V layers, and 78 indexer layers. Both buffer endpoints must already
have the expected linear Offload mapping. An unknown buffer in the reserved
region fails the guard. Ordinary pointers call the original registration
method. Storage bookkeeping, warmup, SDK implementation, payload transfer,
cleanup, and pool configuration retain their original code paths.

The packed MTP adapter preserves the existing target/draft host views and the
original AscendC copy helper. The optional extension is built from the exact
SDK source and mounted as one read-only neighboring file in candidate P
containers. Core SDK, driver, image, and D/Meta services are unchanged. A private
same-inode loader alias was used only by the standalone prerequisite fixture;
production containers use the normal SDK path. Neither uses LD_PRELOAD or a
second core SDK instance.

The candidate startup and C32 short cohort passed:
`results/sglang-C32-L12C-native-startup-v1.json` and
`results/sglang-C32-L12C-native-functional-v1.json`. These establish 64 actual
SDMA workers, 32 P extension mappings and three guarded buffer reuses per P
worker, unchanged P/D KV capacity, retained D graph/operation configuration,
and 64 successful short backend requests. The two 80K semantic/cache cases and
post-request native evidence also passed; see
`results/sglang-C32-L12C-native-readiness-v1.json`. The complete qualification
covers 76 successful backend requests, 2,952 output tokens, both full 581-page
cross-P-DP storage hits, all 32 actual P native packed-MTP first uses, all 32 D
SDMA destinations, and productive D graph counter changes. Both P nodes had
eight model-side cache readers and 40 observed successful batch-get calls each;
the observed trace counts were balanced with no failed calls. Cross-DP whole
output hashes differ, as they did for the recovered L10 reference; the required
fact answers pass. Raw outputs retain those differences. No new formal
performance result is established by these functional qualifications.

Production qualification and the single allowed canonical C32 measurement are
tracked by `loops/L12D-C32-native-HiCache-existing-mapping-integration-v1.json`.
The L12C prerequisite loop is accepted only for the combined SDK/native fixture.

## Accepted C32 formal result (2026-10-19)

The single allowed canonical C32 cohort completed and was accepted as a same-case
native P cache integration result. The accepted measurement is
`results/sglang-C32-L12D-native-HiCache-v1-measurement-annotated-v2.json`
(source v1 SHA `39e31dc3cfb97b5d59553dce69137b01cfcf6160c1ae95837ad6a6fd9a0ffa3d`,
annotated v2 SHA `bab30fb0e0cdec779eb26e833142549e3bdd0b45b4b841f3fe9ba42a519173d7`).
It records 128 formal requests, 320 backend HTTP attempts, 76,992 successful
output tokens (76,800 formal), and a 196.98668286 s full clock excluding the 32
warm logical pairs: 389.8740711045039 output TPS, TTFT P50
17.757243875414133 s, TPOT P50 36.65860450806373 ms.

Against the qualified L10 result this is +26.20757259854496% TPS,
-40.51670798525417% TTFT P50 and +0.2936857162019413% TPOT P50, so the native P
path kept TPOT stable while lowering TTFT. Against the accepted vLLM C32
reference it remains -26.551457885694774% TPS, -16.089087065792963% TTFT P50 and
+91.40219972003096% TPOT P50. D decode is therefore the largest remaining gap.
Aggregate SLA is false: both the TTFT and TPOT thresholds fail. The result is one
cohort versus qualified historical L10, not global optimality and not a universal
causal or precision claim.

Native post-run evidence (`results/sglang-C32-L12D-native-HiCache-v1-native-v1.json`,
SHA `3e021ee4763250b4b611a77df2b6ea84dc3ae293f23e7fca7f9ab4f1c694c1d6`) covers
32 P MemCache readers, 32 D SDMA destinations, 3,458 PD write batches,
2,808,364,105,728 typed descriptor bytes (physical wire bytes remain null), all 32
productive Graph ranks and 261/261 complete metric pairs. The original raw client
observation remains 127/128 with one D-storage null; it was source-qualified, not
backfilled.

The cache path distinction remains: SGLang performs MC G2H followed by the native
H2D stage, while vLLM uses direct HBM `batch_get_into_layers`. The accepted result
does not claim blanket capability or precision parity. L13 is preparing a D-only
two-step diagnostic and is not runtime-authorized.
