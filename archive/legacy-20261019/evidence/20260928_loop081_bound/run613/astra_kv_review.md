# Run613 independent KV ownership review

SCOPED PASS for exported CPU/native ownership; NOT a tensor-generation, mandatory dependency, or timing-bound certificate. This was read-only source/trace analysis; no service or device experiment. Formal Current remains 571.681 tok/s. Strict Resource/Hardware, Scheduling/Execution, Product E2E endpoints and numeric Current-to-limit gap remain null.

## What the existing trace proves

All eight ranks have three extreme::dspark_model occurrences, each containing six CPU aclnnScatterNdUpdateSk events. The first three have no draft-layer scope; the last three lie inside vllm::dsa_forward and layer43, layer44, layer45 respectively. Repeated nested hook scopes with the same label collapse to a label set, not multiple layer executions. Each query layer also contains one CPU aclnnSparseAttnSharedkv.

I independently joined all 144 scatter and 72 attention CPU calls to unique exported native events using explicit torch_to_npu flow: exact Decimal CPU (pid,tid,ts) equals flow start; flow ID joins a unique finish; finish (pid,tid,ts) joins a unique native X record with the expected operator name. No connection_id or six-by-six temporal grouping was used. All joined native records have Model Id 4294967295 and physical stream47. This is explicit exporter association, not independent proof of tensor arguments or profiler clock calibration.

The flow associations agree exactly with all candidate task IDs in Sol's ledger. Those candidate pairings can be promoted to exported-flow ownership. Rank5 occurrence0's six native scatters all start after occurrence1's Target Host start in the exported timeline. Host-cycle timestamp bins are therefore invalid for native ownership. This does not establish overlap with the next Target's device work or an avoidable wait. Logical cycles64–66 remain nominal pending a cycle/launch-generation witness. Run610 remains INVALID, with observer, carry-in and stop-fence caveats.

## Source attribution and remaining uncertainty

llm_base_proposer.py:1328–1343 calls build_model_inputs_first_pass before draft forward. dflash_proposer.py:267 passes shared context states, positions and per-layer context slots to precompute_and_store_context_kv. deepseek_v4_dspark.py:209–224 independently projects those context states with each layer's wkv/norm/RoPE, then writes attn.dsa_attn.swa_cache_layer.kv_cache; lines231–232 later execute query layers. device_op.py:710 dispatches the non-A5 scatter.

The first three trace events are strongly source-consistent with the three context KV writes. Their individual layer43/44/45 assignment still relies on source loop order: no context-layer tag or call stack was recorded. The latter three have measured query-layer scope ownership, consistent with each layer's SWA query KV write. The prefill path (dsa_v1.py:2086) and decode path (:2414) both write SWA; this capture does not directly tag which branch supplied an individual call. Do not relabel all six as context writes or Target-cache writes.

ops/dsa.py:227–231 builds query cache from self.swa_cache_layer.kv_cache; device_op.py:821 unpacks the SWA element. For compress_ratio<=1, query attention receives ori_kv=swa_kv_cache (dsa_v1.py:2097 or :2540). This establishes intended source-level aliasing and identifies SparseAttnSharedkv as the first attention consumer in that path. It does NOT measure live storage address/allocation generation, cross-layer aliasing, byte range, slot validity, actual read set, or which writer supplied a consumed row after query overwrites.

CPU args contain only Sequence number/Fwd thread id; native args provide task/stream/model IDs and connection_id. There are no tensor addresses or generation/slot payloads. Run606 slot intersections must not be transferred to Run610's different W0. Equal numeric slots alone never prove a hazard.

The three context projections read shared context without consuming each other's projected outputs in this source. Thus the all-context-before-layer43 order has a potential scheduling restriction beyond source dataflow. Actual storage aliasing, scratch lifetimes, streams and resource contention still constrain legal overlap. No saving follows from source independence. A query only needs its actually-read generations ready; current whole-loop order is not a universal lower bound.

## Smallest missing probe

For one bounded pair of adjacent active ordinary cycles in a new admitted same-W0 packet, record all8 typed context-scatter, query-scatter and immediately-pre-attention witnesses: rank/cohort/cycle/layer/role, cache object token, storage allocation-generation token, storage base+offset/nbytes, dtype/shape/stride, slot-buffer generation, valid rows, stream and launch correlation token. Record actual device slot payload plus consumer block-table/seq-length/window/mask/sparse-index metadata into a small preallocated witness buffer, copying it to Host after the window. Track write generations separately from tensor _version; pointers can be reused.

Derive per-row last writer including query overwrites and invalid/parked rows. Include predecessor retained-KV state and one successor consumer. Link typed scopes to native flows while retaining existing stream/event waits. Add no synchronization or algorithm/model-work change. This resolves true read-after-write/overwrite/lifetime edges; it does not itself yield attainable overlap time. A later minimal same-state schedule intervention tests one legal relaxation and contention, with A0/probe/A1 for observer transfer. No repeat of Run610 is needed solely to recover its existing flow inventory.

Historical retrieval: performance_knowledge.py search "DSpark context KV overlap", source commit db3beef223e0b5acd81ccccb444600c1b22aac8a. R02 warns to verify the loaded V1 proposer/active path. R20's DP2TP4 local-Q/all-gather overlap yielded ~59ms TTFT from ~157ms overlap in exact32K short serving, supporting contention/E2E gates, not a transferable speedup. Current DP1TP8 runtime and fixed acceptance require their own evidence.

## Evidence identity

Run613 ledger SHA256 9bfb19294d64b6b3d4e233cccdcdd8a3ddc09db7fbc421a3f10e388cffcce267. Run611 parse manifest SHA256 b51eac01dcdc9f183544d51d20040f371f5ffb23a686ded7372b2cb366c84e83. The manifest identifies full trace paths. Direct source hashes below describe current installed files; acquisition manifests pin only a subset, so new hashes are not retroactive proof of every Run610 imported byte.

| Installed vllm_ascend source | SHA256 |
|---|---|
| models/deepseek_v4_dspark.py | b459fa373e89da1722085cffa63a9959502dd29fb1a0cf4b8f073e431d811867 |
| models/deepseek_v4.py | 11dd3e983dc1d628bf42a95581f9617861471b1f658739db44c20df886147247 |
| spec_decode/dflash_proposer.py | 6be46559f9f869861c676efb4531b0c01eef8a0445521ecadf5c1fec1c4412f8 |
| spec_decode/dspark_proposer.py | e9163996db794c5fba777ea55a5374283780a2fd3404778f5369caea76f9344f |
| spec_decode/llm_base_proposer.py | e3e1ff579e67f845f155c983294ae5546e6ee76ba330098d7867e870c3f9f6e4 |
| attention/dsa_v1.py | 88a0cd69dbbba476d0ac179431568590e6e876134390fba77f22721069f491cb |
| device/device_op.py | 67dda24f28de847f9aee87db9714e5d2293ebb46309ad13152538cf63ca5f1b4 |
| ops/dsa.py | 3abeb11a3d82f3919d67f4f7c5ab404faa476c2eb8455d2eca777dc1712811fc |
| utils.py | d56362aa59a94a25010dc967572e394f46735407ae73d51b09f9948eb89d830c |

| Rank | trace_view.json SHA256 |
|---|---|
| 0 | 969b7b32c1fdb09bbe9bd59ca2302789fe92ac936473233bba609e8e124077a5 |
| 1 | 629f58a8c789d4dc30f4f896d9b8550a7c5e661ccd49048af7ed993b630ce771 |
| 2 | 9bc89e028804efffdc9f7bfdf2c4580fef7bf6568af1b31d65b1c98f77b86d9b |
| 3 | 02db2c7233681152fdc5323e9f250876e93f6af9221c3b8187ff15d1f568fbd1 |
| 4 | ecbd9f1d2d51e28fd8a5b5ceccd3d6199b3a4a42117afc03b864da575ab97b59 |
| 5 | d97778699dff2df705ea0b1b923a2505118b638ba7ce7d6dbff8dc557e7033a1 |
| 6 | 39de8055f8651ddea6cc9154cee14de7a806f7d9201f3a8965d41846fc3f496a |
| 7 | 5fd084806a970389062706cfdbbbca490b35d6285562281a4011b8791165647e |

Each identity below is CPU trace index/native trace index/native task ID. All stream47 / Model UINT_MAX. Ordinal is Host occurrence index, not a certified cycle number. Scatter1–3 lack context-layer identity; scatter4–6 and attention1–3 map to measured layer43/44/45 scopes.

| Rank/ordinal | Six scatter identities | Three attention identities |
|---|---|---|
| 0/0 | 2052/53303/44847; 2129/53355/44873; 2206/53407/44899; 2390/53491/44934; 2842/53673/44995; 3294/53855/45056 | 2398/53493/44935; 2850/53675/44996; 3302/53857/45057 |
| 0/1 | 5976/62869/45603; 6053/62921/45629; 6130/62973/45655; 6314/63057/45690; 6766/63239/45751; 7218/63421/45812 | 6322/63059/45691; 6774/63241/45752; 7226/63423/45813 |
| 0/2 | 9900/72435/46359; 9977/72487/46385; 10054/72539/46411; 10238/72623/46446; 10690/72805/46507; 11142/72987/46568 | 10246/72625/46447; 10698/72807/46508; 11150/72989/46569 |
| 1/0 | 2052/53297/44831; 2129/53349/44857; 2206/53401/44883; 2390/53485/44918; 2842/53667/44979; 3294/53849/45040 | 2398/53487/44919; 2850/53669/44980; 3302/53851/45041 |
| 1/1 | 5976/62863/45587; 6053/62915/45613; 6130/62967/45639; 6314/63051/45674; 6766/63233/45735; 7218/63415/45796 | 6322/63053/45675; 6774/63235/45736; 7226/63417/45797 |
| 1/2 | 9900/72429/46343; 9977/72481/46369; 10054/72533/46395; 10238/72617/46430; 10690/72799/46491; 11142/72979/46552 | 10246/72619/46431; 10698/72801/46492; 11150/72983/46553 |
| 2/0 | 2052/53298/44847; 2129/53350/44873; 2206/53402/44899; 2390/53486/44934; 2842/53668/44995; 3294/53850/45056 | 2398/53488/44935; 2850/53670/44996; 3302/53852/45057 |
| 2/1 | 5976/62864/45603; 6053/62916/45629; 6130/62968/45655; 6314/63052/45690; 6766/63234/45751; 7218/63416/45812 | 6322/63054/45691; 6774/63236/45752; 7226/63418/45813 |
| 2/2 | 9900/72430/46359; 9977/72482/46385; 10054/72534/46411; 10238/72618/46446; 10690/72800/46507; 11142/72982/46568 | 10246/72620/46447; 10698/72802/46508; 11150/72984/46569 |
| 3/0 | 2052/53326/44844; 2129/53378/44870; 2206/53430/44896; 2390/53514/44931; 2842/53696/44992; 3294/53878/45053 | 2398/53516/44932; 2850/53698/44993; 3302/53880/45054 |
| 3/1 | 5976/62892/45600; 6053/62944/45626; 6130/62996/45652; 6314/63080/45687; 6766/63262/45748; 7218/63444/45809 | 6322/63082/45688; 6774/63264/45749; 7226/63446/45810 |
| 3/2 | 9900/72458/46356; 9977/72510/46382; 10054/72562/46408; 10238/72646/46443; 10690/72828/46504; 11142/73010/46565 | 10246/72648/46444; 10698/72830/46505; 11150/73012/46566 |
| 4/0 | 2052/53345/44847; 2129/53397/44873; 2206/53449/44899; 2390/53533/44934; 2842/53715/44995; 3294/53897/45056 | 2398/53535/44935; 2850/53717/44996; 3302/53899/45057 |
| 4/1 | 5976/62911/45603; 6053/62963/45629; 6130/63015/45655; 6314/63099/45690; 6766/63281/45751; 7218/63463/45812 | 6322/63101/45691; 6774/63283/45752; 7226/63465/45813 |
| 4/2 | 9900/72477/46359; 9977/72529/46385; 10054/72581/46411; 10238/72665/46446; 10690/72847/46507; 11142/73029/46568 | 10246/72667/46447; 10698/72849/46508; 11150/73031/46569 |
| 5/0 | 2052/53351/44847; 2129/53403/44873; 2206/53455/44899; 2390/53539/44934; 2842/53721/44995; 3294/53903/45056 | 2398/53541/44935; 2850/53723/44996; 3302/53905/45057 |
| 5/1 | 5976/62917/45603; 6053/62969/45629; 6130/63021/45655; 6314/63105/45690; 6766/63287/45751; 7218/63469/45812 | 6322/63107/45691; 6774/63289/45752; 7226/63471/45813 |
| 5/2 | 9900/72483/46359; 9977/72535/46385; 10054/72587/46411; 10238/72671/46446; 10690/72853/46507; 11142/73035/46568 | 10246/72673/46447; 10698/72855/46508; 11150/73037/46569 |
| 6/0 | 2052/53392/44844; 2129/53444/44870; 2206/53496/44896; 2390/53580/44931; 2842/53762/44992; 3294/53944/45053 | 2398/53582/44932; 2850/53764/44993; 3302/53946/45054 |
| 6/1 | 5976/62958/45600; 6053/63010/45626; 6130/63062/45652; 6314/63146/45687; 6766/63328/45748; 7218/63510/45809 | 6322/63148/45688; 6774/63330/45749; 7226/63512/45810 |
| 6/2 | 9900/72524/46356; 9977/72576/46382; 10054/72628/46408; 10238/72712/46443; 10690/72894/46504; 11142/73076/46565 | 10246/72714/46444; 10698/72896/46505; 11150/73078/46566 |
| 7/0 | 2052/53367/44846; 2129/53419/44872; 2206/53471/44898; 2390/53555/44933; 2842/53737/44994; 3294/53917/45055 | 2398/53557/44934; 2850/53739/44995; 3302/53919/45056 |
| 7/1 | 5976/62929/45602; 6053/62981/45628; 6130/63033/45654; 6314/63117/45689; 6766/63299/45750; 7218/63479/45811 | 6322/63119/45690; 6774/63301/45751; 7226/63481/45812 |
| 7/2 | 9900/72495/46358; 9977/72547/46384; 10054/72599/46410; 10238/72683/46445; 10690/72865/46506; 11142/73045/46567 | 10246/72685/46446; 10698/72867/46507; 11150/73047/46568 |

## Sol flow reducer recheck

Directly reviewed reducer SHA256 a1d64d08573f52c533fc300b30654df2dd3c335018616f5fba34b56967c91793 and output SHA256 dc17ab40f85d38d2d5f87d921d44d1b1f4debb51dea6aceddd9b9ae0ad9c99b9. All 216 independently obtained native task identities above agree with its output, including rank7. SCOPED PASS for this frozen input. My independent scope checks additionally require same pid/tid and full CPU-duration containment inside layer/model scopes; they reproduce the reducer's layer tags. The reducer currently checks timestamp containment without pid/tid for layer tags and uses float-derived proposer boundaries: harden those for future trace reuse, but neither changes this acquisition's admitted associations. First-reader byte/generation and exact logical-cycle ownership remain unclosed as described above.
