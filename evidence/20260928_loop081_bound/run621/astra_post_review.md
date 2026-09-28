# Run621 independent Astra High POST review (including Run622)

**Verdict: PASS, scoped actual-workload descriptor acquisition, guarded recovery, and conditional payload arithmetic.** No service or device workload was rerun. This closes observed cache geometry/format uncertainty, not row provenance or a Scheduling Bound.

## Independently verified provenance and recovery

- All nine cleanup exit fields are zero. `live/cleanup_status.txt` SHA256: `cd06ecab09b860d341c260eb9fe671532b7543ef42559f84d61672f69147157a`.
- Source before/after lists are byte-identical, as are script lists; every listed current source/script/dataset hash was independently recomputed and matches. Before-list SHAs: source `0ffbd1e0a84b1addf9c53a685414a8335d569ec5f8a3fe5777c3093df68d06ae`; scripts `fe52e6cfa1fb5e15e6379be4d646f88c7af4796227a1cd03350175256643d292`.
- Live patch check SHA `718921320c40de96385ffb1588a6b6124b0b51b98d2f01d32167292ac3650c04` matches preflight. Ignoring action labels, check/install/restore equal the saved patch manifest. Reviewed controller `7b07871f...`, patch `8e7ecebb...`, validator `d1882b2f...` occur in the recorded script manifest.
- Final stopped log reports no service processes and eight idle-HBM values below6144 MiB. POST counts before and after stop both96. These are recorded recovery facts, not a fresh real-time server-state claim.
- All281 Product raw-hash links and all64 dispatch raw-hash links were independently recomputed and match.

| Admission | SHA256 |
|---|---|
| client_two_phase_admitted | `5e7eba24dbbae40e324382cfdb33cede25866d1178d40d20f3c11b8ffcbb1f8f` |
| diagnostic_compact_basis_all8_admitted | `9ae555a68df9432742db90754a2d84a856699687ec9008d681eff408f99b11cf` |
| product_identity_pass | `2b39f6e5ea790545446e14eaf7402ff3a84b30870d5693460e5ffe49ddbfe916` |
| scoped_dispatch_identity_pass | `753a6f20ab298fa5d9b07623ac9d0615e048dea41d60520a28fc62e06217048b` |
| all8_actual_W0_two_cycle_cache_descriptors_only_pass | `0e78e7b9258de04aa40f6b2ef95e89fb0dd9f6281b431161a613e34be8d43424` |

Gates establish their stated diagnostic scopes, not a new stock-oracle correctness campaign or equality to another run's acceptance trajectory. Run621 measured cohorts5–8 have286/298/292/281 cycles. Do not join its pointers to Run610/611 native events merely through equal rank/layer/cycle labels: this is a new process/workload realization.

## Actual cache observations

Independently validated all eight raw `product/rank{0..7}_cohort5.json` files, each containing cycle64/65 and layers43/44/45. Across all48 cache observations:

- Device exactly `npu:{rank}`; BF16; format2/ND; shape `[34090,32,1,512]`; stride `[16384,512,512,1]`; block32; zero view offset; data pointer equals storage base.
- Storage bytes exactly shape product times2: **1,117,061,120 bytes/cache**. This is reported tensor storage capacity, not allocator physical reservation or traffic.
- Prefixes are `mtp.0/1/2.self_attn.swa_cache`. Three storage intervals are disjoint within each rank; I explicitly checked interval non-overlap, stronger than merely different cdata/base pairs. Equal numerical addresses across ranks do not imply inter-device sharing.
- Owner/storage/view descriptors match at the two observations; model owner id also matches. This does not prove uninterrupted lifetime or absence of legal reuse/overwrite.
- In every one of16 snapshots, common slot_mapping equals state target_slot_mapping in its complete descriptor. That establishes sampled view/storage alias, not actual native scatter/reader operand identity.

Capture remains pre-Draft entry, before common-metadata refresh. No cache row contents or metadata values were captured.

Raw cohort5 SHAs, rank order:

```
0 c10a9be435c9bddb7f3c2cf6956b3ae9c63320cc5c1a10844648c911628cd2ab
1 52e17f52ff1701c31397bd0b201b6b629227dec7e6d5315ec8252627e14e6578
2 cb89847265286cbf92d0d790b8d998f7cea647e08c943b5acdd633efcf959311
3 449ae684eb13d6ed0f176dc44ef4cc0cd788a5f2dd20be836b359abb7c0fca0f
4 cb3d781875d07f74300484db5735e1bc9c6b4d4e5d06bb2ca608db59656f8721
5 95056ac8d05f86ccf659690aef98c8c4b5ffe9e0a112f0dbd7d3aa3aa90176b8
6 492f7cc4e8e29a146b0faecc59803379286b45485d98938b4ea11d15eb053953
7 a592c9138f73445a96d06ae0b31baec1cacabf3f54ab314bd49b49dc469925e6
```

## Run622 arithmetic and scratch limits

For this observed contiguous ND cache, one physical row is512 BF16 = **1024 bytes**; one block32768 bytes; capacity1,090,880 rows. For valid physical slot p, byte interval is `[base+1024*p, base+1024*(p+1))`, conditional on the actual ABI tensor being this view and `0<=p<1090880`. It is not an eligible-row or HBM-traffic claim.

Run622's example is correct: two cycles × three layers × one selected row = six selected row instances; one retained snapshot6144 bytes; pre/post retained snapshots **12,288 bytes =12 KiB/rank**. That is payload arithmetic only. It does not prove valid/meaningful selected rows, that six rows suffice for a witness, a live allocation, actual copy traffic, or total observer footprint under128 KiB. Add row indices, metadata, generation tags, alignment, native-copy temporaries and buffering. The128 KiB comparison is a proposed payload cap, not measured headroom.

Slot-map logical view384 bytes vs backing33,152 bytes. Full block-table view `12*32768*4=1,572,864 bytes` =1.5 MiB (1.573 decimal MB); backing2,097,152 bytes =2 MiB (2.097 decimal MB). Avoid ambiguous "2 MB". Other common metadata: query_start_loc52 logical bytes/backing72; seq_lens48/backing64. Full common compact payload is1,573,348 bytes/sample. State positions/input ids add768/384 bytes if needed. Count common/state slot alias once only when snapshot time and required generation coincide.

Full three-cache payload is3,351,183,360 bytes/rank/sample; all-eight observed storage sum26,809,466,880 bytes. Neither is compulsory traffic. The Run622 full-block-table rejection flag is a design decision to favor selected metadata, not experimental proof that selective capture is sufficient: indices must derive from the actual branch and reader ABI. Live spare HBM, scratch reuse, workspace and timing perturbation remain unmeasured; post-stop free memory does not certify live headroom.

The reducer's distinct (cdata,base) tuple gate alone would not establish arbitrary interval non-overlap; actual Run621 intervals were independently checked here and do not overlap. No revision is required for these pinned observations, but do not generalize that gate as an allocation/lifetime proof.

## Missing evidence / Bound impact

Next typed observation must bind actual writer/reader call tensors to storage/views; preserve relevant metadata generations through original-stream snapshots before mutation; track row writers and lawful reuse/overwrite; and join host/native tasks within the same acquisition. Loaded binary/ABI path, first eligible read and first physical read are separate claims. Lifetime evidence must not be manufactured by prolonging production ownership. A selected-row witness may certify a particular edge; its absence cannot certify absence of all dependencies.

Run621/622 remove a geometry/format unknown and provide conditional scratch arithmetic. They establish no compulsory traffic, eliminated wall time or fresh-W-minus witness. Client604.151 tok/s is diagnostic, not formal improvement.

**Current Formal571.681 tok/s remains unchanged. Strict Resource floor, strict Scheduling floor, Product E2E Bound and numerical Current-to-Bound gap remain null.**

Run622 exact reviewed SHA256: reducer `ad5573fd7d3fa0422cfec9afcde3efcdb8c71418ec08882d02650e2d2e1fc9f5`; output `8b0519762f0f43fd36a704c8bbca4038f242a059376e9d54eeab86234a353e09`. Independently executed only the read/compute portion, excluding output writes, and compared the whole result object to the saved JSON: exact match.
