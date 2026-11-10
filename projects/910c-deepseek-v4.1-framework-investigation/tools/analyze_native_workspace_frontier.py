#!/usr/bin/env python3
"""Bound workspace ownership and conditional capacity; reuse old matched prices."""
from pathlib import Path
from hashlib import sha256
import json

OUT=Path('audit/execution-search')

def write(name, value):
 (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def main():
 retrieval=json.loads((OUT/'native-workspace-upstream-retrieval.json').read_text())
 ranges={
 'framework/OpCommand.cpp':[[1,184]],'framework/OpCommand.h':[[1,181]],
 'framework/OpParamMaker.h':[[223,245],[283,403]],
 'framework/OpParamMaker.cpp':[[295,390],[497,531],[732,780],[850,904]],
 'framework/NPUDefine.h':[[53,105]],'framework/NPUDefine.cpp':[[1,61]],
 'aten/common/TensorFactories.cpp':[[119,124],[169,221]],
 'framework/utils/OpPreparation.cpp':[[415,451]],
 'core/npu/NPUCachingAllocator.cpp':[[45,120],[1213,1303],[1524,1588],[1637,1657],[1969,2080],[2187,2300],[2475,2566],[2570,2602],[2614,2695],[2918,2958],[3084,3110],[3180,3267],[3370,3423],[3730,3938],[4055,4080]],
 'core/npu/NPUStream.cpp':[[417,446],[588,598],[735,788],[830,858]],
 'core/npu/NPUQueue.cpp':[[1,85],[430,459],[491,511]],
 'core/npu/NPUGraph.cpp':[[150,284],[299,374]],
 'core/npu/register/OptionsManager.cpp':[[113,139],[166,177],[639,712]],
 'core/npu/NPUWorkspaceAllocator.cpp':[[1,105]],
 'framework/OpCmdHelper.cpp':[[1,143]],
 }
 sources=[]
 for r in retrieval['files']:
  p=OUT/'native-workspace-upstream'/r['path'];b=p.read_bytes();assert len(b.decode())==r['expected_characters']
  sources.append(dict(**r,local_path=str(p),sha256=sha256(b).hexdigest(),bytes=len(b),reviewed_ranges=ranges.get(r['path'].removeprefix('torch_npu/csrc/'),[])))
 backend=json.loads((OUT/'native-workspace-backend-records.json').read_text())
 assert len(backend['headers'])==9 and all(r['matches_reference'] for r in backend['headers'])
 assert len(backend['workers'])==16
 review=dict(status='BOUNDED_REFERENCE_SOURCE_REVIEW',reference_commit=retrieval['ref'],sources=sources,installed_header_matches=backend['headers'],qualification='Version.py declared commit and 9 byte-exact installed headers; .so current disk SHA/maps separately catalogued. Not binary/source equivalence or dynamic option-state proof. Empty ranges mean saved identity/reference only; inclusion of a file does not mean its whole body was read.',actual_native_entry='native-workspace-entry.json',source_errors=['Guessed op_api_common.cpp not present; real macro is header.', 'Guessed OpCommandBase sources not present; actual implementation is OpParamMaker.', 'EmptyTensor.cpp was a CPU registration locator; selected NPU empty is TensorFactories.cpp169-221.'],whole_chain_complete=False)
 write('native-workspace-source-review.json',review)
 formulas=[]
 for M,top,E in [(3,6,24),(24,6,24),(3,3,8)]:
  for cap in [65536,min(65536,16*M*top)]:
   payload=cap*5120*3
   other=((M+255)//256)*256*top*4+16*16*E*4*2+cap*4*2+16*4*16+(E+16)*4*16
   formulas.append(dict(M=M,topK=top,local_experts=E,conditional_capacity=cap,two_payload_banks_bytes=payload,cocWorkspace_bytes=payload+other,system_prefix_bytes=16*1024*1024,total_formula_if_coc_dominates_bytes=16*1024*1024+payload+other,not_measured=True,guard='All16 sender row count <=M for this invocation, capacity/clipping semantics and alignment preserved; actual workspace is 16MiB + max(cocWorkspace,initRoutingWorkspace). API result/system prefix/inner scratch/pool high-water not observed. M3/top6 is conditional example, not declared actual target geometry.'))
 price=json.loads((OUT/'dfc-gemm-tile-profile.json').read_text())
 selected=[]
 for x in price['conditional_values']:
  if x['kind']=='common_suffix' and (x['common_critical_stage_fraction'],x['removable_or_hidden_fraction'],x['added_us_per_layer']) in [(1,1,0),(.25,.5,5)]:
   selected.append({k:v for k,v in x.items() if k!='requests'})
 frontend=dict(status='BOUNDED_NATIVE_WORKSPACE_OWNERSHIP_AND_CAPACITY_FRONTIER',whole_chain_complete=False,every_replacement_bitwise=True,report='原生workspace寿命与融合容量判断.md',source_review='native-workspace-source-review.json',catalog='native-workspace-catalog.json',backend_records='native-workspace-backend-records.json',
  actual_path=['actual EXEC_NPU_CMD GetWorkspaceSize/executor','Byte at::empty -> selected TensorFactories empty -> NPUCachingAllocator.allocate -> getCurrentNPUStreamNoWait -> stream-keyed block pool','scope-local tensor releases before handler; local_raw_delete -> free(Block), cache insertion or recorded-other-stream events, no implicit device completion','raw pointer/std::function captured; OpCommand Run direct or queued COMPILE_AND_EXECUTE -> copied ExecuteParas/customHandler -> ExecFunc -> opApi captured aclstream','kernel entry/DFC WorkspaceInfo and original GMM/activation/combine/flag last readers; keep old source layout and real stream order','graph capture stream/model-specific private pool, replay ExecuteAsync using recorded addresses, releasePool refcount after graph use; tensor destruction != physical release'],
  source_policy=dict(initial_worker_environments=backend['workers'],default_if_not_overridden=dict(TASK_QUEUE_ENABLE=1,PER_STREAM_QUEUE=0,ASCEND_LAUNCH_BLOCKING=0,MULTI_STREAM_MEMORY_REUSE=2),guards=['TASK_QUEUE dynamic atomic override or sync-launch stream may select direct path; initial /proc environment is not current cached OptionsManager state.','Graph capture rejects queue mode2/default/external capture stream; captured native custom handler allowed; replay may use a different current stream.','CheckForceUncached selects physical malloc/free outside capture; not inferred active from absence of environment.','NPU empty has deterministic fill only if all3 runtime flags true; no blanket zero-fill saving.','Dedicated NPUWorkspaceAllocator is a different explicit unsafe_empty_workspace(size,stream) path; macro at::empty does not select it.']),
  ownership_findings=['Caching allocator retains stream-keyed blocks; free/requested counters and Tensor scope do not show physical release. Same-stream submission order can allow reuse after previously submitted device work; no global fence was found in the selected allocate/deleter path.','The macro captures a raw workspace address, not its Tensor. Queued callback copying retains std::function, not a missing workspace Tensor lease. This alone does not prove an observed fault; interleaving/allocation/current binary runtime details are not measured. New fusion/aux-stream work must hold an explicit last-reader lease or prove complete ordering.','Recorded foreign streams create events; capture may defer these. recordStream cannot be replaced by merely recording that an API callback returned.','Private pools can share by explicit handle/refcount; replay does not rerun host empty/GetWorkspaceSize per native node. Cross-layer or graph-bucket pools cannot be assumed independent or all already shared.','Selected DFC holds input/scale/C1-C2/A-activation/counters/flags through routing+MMA+activation+combine/peer protocol. QLI output candidate/index persists beyond scratch, and current arch22 empty LD epilogue still touches reset/init/prefetch/sync. No arbitrary all-workspace alias from C++ scope or empty payload.'],
  conditional_capacity_formulas=formulas,
  old_matched_parent_price=dict(source='dfc-gemm-tile-profile.json',sha256=sha256((OUT/'dfc-gemm-tile-profile.json').read_bytes()).hexdigest(),reference_conditions=selected,scope='2560 actual old C8 first-token DFC tasks/all16 and8 typed groups, closed old clock endpoints. Reused conditional whole native suffix prices only; no new capacity->latency conversion, no sum with QLI or other parents.',one_percent_phase_threshold=price['one_percent_stage_thresholds']),
  database_limit='Only rank0 original DB memory/schema inspected: no allocation/free or PYTORCH_API table. CANN workspace APIs are duration observations over whole DB, not exact bytes, allocator time or selected160 DFC native first-token matches. E.g.InnerDFC853 API vs160 selected native calls prevents direct attribution.',
  confirmed_implementable=[],
  stopped=[dict(recipe='Per-layer960MiB/QLI declaration multiplied by40 or16 = newly freed VRAM',reason='Same-stream caching/private pools and already aliased payloads; no actual physical-peak/disjoint-allocation evidence.'),dict(recipe='Per-native malloc/free saving multiplied by captured decode iterations',reason='Native host allocation/query callbacks execute during capture, replay uses recorded addresses; zero recurring host-empty credit for those nodes.'),dict(recipe='Use current scope end or callback return as workspace last-device-read',reason='No device-completion proof; raw pointer plus cached/graph memory can remain live.'),dict(recipe='Remove all workspace zeroing by replacing macro with dedicated allocator',reason='No unconditional workspace zeroing in selected source; dedicated allocator is a separate path and has its own grow synchronization.')],
  pending=[dict(candidate='Per-bucket/shape-bounded DFC capacity and safe shared scratch for larger joint fusion',gap='Global all16 sender row maximum/clipping for each eager/padded/dummy target/draft/prefill; actual GetWorkspaceSize & inner routing max; allstream/system/HCCL last readers; pool pointer/physical peak and retained prefill graph pools; per-primitive bit and net resource costs.'),dict(candidate='QLI scratch contraction + stable address/index/panel plan',gap='Actual CANN LibAPI/system prefix, execution tiling/workspace result and scratch pointer map; arch22 LD epilogue touches; graph-bucket union and source reader leases. Source99.023MiB vs4.6875MiB not measured release.'),dict(candidate='Aux-stream/co-resident/megakernel schedule using the reduced scratch',gap='Lease and queue/current-stream integration plus capture/replay address stability, peer flags and exception retirement; new space only valuable if removes actual capacity constraint or improves critical schedule. DFC24/48 launch corrected, no free4/8 grant.'),dict(candidate='Graph/host query cache and allocation coalescing',gap='Actual misses/capture/rebuild cohorts, executor/converted-parameter/version keys, exclusive CPU cost and overlap; whole-DB API durations cannot become per-request savings.')],
  next_entry='smla-metadata-delivery-entry.json',new_NPU_work=0,new_profile_collection=0,model_loads=0,service_requests=0,remote_writes=[],service_changes=[])
 write('native-workspace-frontier.json',frontend)
 write('native-workspace-exploration-notes.json',dict(status='READ_ONLY_SOURCE_REVIEW_NOT_A_NATIVE_TEST',scope='No Torch/NPU imports, model calls, service requests, profiles or remote writes. Guessed missing paths and CPU EmptyTensor locator are excluded from selected implementation proof.',source_recovery='24 exact saved references; terminal output truncation was recovered by bounded rereads, not claimed as unseen source coverage. AddFile extra newline corrected exactly once and all expected character counts asserted. Nine installed headers SHA-match reference copies.',qualification=review['qualification'],new_NPU_work=0))
 print(json.dumps(dict(source_references=len(sources),reviewed_nonempty=sum(bool(x['reviewed_ranges']) for x in sources),installed_header_matches=9,capacity_examples=len(formulas),old_price_conditions=len(selected),new_primitives=0)))

if __name__=='__main__':main()
