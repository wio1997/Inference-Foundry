#!/usr/bin/env python3
"""Record actual packet producer/readers, bounded branch witnesses and paid parent."""
from hashlib import sha256
from pathlib import Path
import json

OUT=Path('audit/execution-search')

def main():
 base=json.loads((OUT/'smla-metadata-execution-identity.json').read_text());dep=json.loads((OUT/'smla-schedule-dependencies.json').read_text());by={r['runtime_path']:r for r in base['sources']+dep['sources']};assert len(by)==21
 ranges={
 'sparse_flash_mla_metadata_aicpu.cpp':[[25,1638]],'sparse_flash_mla_metadata_aicpu.h':[[24,398]],
 'sparse_flash_mla_metadata_check.h':[[365,407]],'sparse_flash_mla_torch_adpt.h':[[24,89],[156,182]],
 'sparse_flash_mla_kernel_metadata.h':[[20,67]],'sparse_flash_mla_arch22_metadata.h':[[24,80]],
 'sparse_flash_mla.cpp':[[1,131]],'sparse_flash_mla_tiling.cpp':[[374,397],[2240,2330],[2380,2436],[2480,2498]],
 'sparse_flash_mla_common_arch22.h':[[40,55],[247,320]],
 'sparse_flash_mla_csa_kernel.h':[[400,458],[470,599],[736,979]],
 'sparse_flash_mla_swa_kernel.h':[[511,648],[809,1006]],
 'sparse_flash_mla_csa_block_vector.h':[[850,884]],'sparse_flash_mla_swa_block_vector.h':[[1040,1080]],
 'sparse_flash_mla_csa_block_cube.h':[[410,565]],
 'sparse_flash_mla_template_tiling_key.h':[[26,109]],
 'aclnn_sparse_flash_mla_metadata.cpp':[[1,208]],
 'dsa_v41.py':[[879,918],[1060,1098],[1106,1220]],
 }
 rows=[]
 for rp,r in by.items():
  p=Path(r['local_path']);b=p.read_bytes();assert sha256(b).hexdigest()==r['runtime_sha256'];n=len(b.decode().splitlines())
  spans=[[a,min(z,n)] for a,z in ranges.get(p.name,[])]
  # l0op producer shares basename with the kernel entry, so path determines body.
  if '/op_host/op_api/sparse_flash_mla_metadata.cpp' in rp:spans=[[1,n]]
  assert all(1<=a<=z<=n for a,z in spans)
  rows.append(dict(runtime_path=rp,path=str(p),sha256=r['runtime_sha256'],lines=n,reviewed_ranges=spans,scope='Only stated actual read ranges; whole source/header inclusion or SHA is not full coverage.'))
 review=dict(status='BOUNDED_ACTUAL_SMLA_PACKET_PRODUCER_AND_READERS',source_pins=rows,source_count=len(rows),reviewed_nonempty=sum(bool(r['reviewed_ranges']) for r in rows),whole_chain_complete=False,qualification='8 earlier actual source matches plus13 current dependencies (9 matched,4 newly retrieved); same owned171 container/image, no loaded binary/source equivalence or runtime deterministic getter claim.')
 (OUT/'smla-schedule-source-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
 cpu=json.loads((OUT/'smla-schedule-cpu-a1/result.json').read_text());paid=json.loads((OUT/'smla-address-plan-paid-price.json').read_text())
 selected=[{k:s[k] for k in ['K','sparse_rate','added_us_per_native_call','plan_charged_byte_multiplier','max_percent','min_percent']} for s in paid['scenarios'] if s['K'] in [256,512] and s['sparse_rate']==.75 and s['added_us_per_native_call']==5]
 frontier=dict(status='SMLA_NATIVE_METADATA_SCHEDULE_AND_CONSUMER_BOUNDARIES_BOUNDED',whole_chain_complete=False,every_replacement_bitwise=True,report='SMLA分核包与原顺序交付判断.md',source_review='smla-schedule-source-review.json',CPU_fixture='smla-schedule-cpu-a1/result.json',paid_parent='smla-address-plan-paid-price.json',
  correction=dict(old='arch22 entry does not pass FLASH_DECODE to SMLAType',result='WITHDRAW_WRONG_REASON; entry does pass FLASH_DECODE. Actual host GET_TPL_TILING_KEY selects0 and arch22 key selection only compiles0; BATCH_CONSISTENCY is the flag not forwarded/consumed in SMLAType. Six compiled keys retain prior identity. No automatic FD admission.'),
  packet=dict(allocated_int32_words=1024,defined_struct_words=900,defined_struct_bytes=3600,tail_unwritten_words=124,FA_core_capacity=36,FD_vec_capacity=72,FA_words_per_slot=9,FD_words_per_slot=8,arch22_directly_consumed_FA_fields=['enable','BN2/M/S2 start','BN2/M/S2 end'],arch22_directly_consumed_FD_fields=[],core_attrs='ACLNN queries actual platform Cube/Vector counts and SocLongVersion then passes attrs to AICPU;36/72 are packet max capacities, not actual launch counts.',bit_gate='Keep original full metadata output/copy until independently bit-qualified; CPU900-word equality is not1024-word primitive bit acceptance.'),
  actual_deliveries=['dsa_v41 _publish_task already shares packet by smla:c0/c1/c2 in same batch; persistent output copy and task/event publication remain. No per38-layer duplicate metadata generation credit.','ACLNN does input checks/contiguous for9optional tensors, current platform attrs and ACL_OPT_DETERMINISTIC==3; AICPU Prepare/ParamsCheck/ParamsInit -> cost/split -> GenMetadata. No selected-index values or block-table physical addresses are inputs.','Default ASCEND910 metadata schedules whole rows (MbaseG64 forCSA andori_sparse; denseSWA can cover4query rows), block/force assignment enabled only via supportFd (batch consistency here).','CSA/SWA Init parse FA bounds; Process reads enable. ConstInfo holds them while query lengths/page tables/TopK remain live in loops. Global shared packet lease ends only after last layer/rank metadata read; this is not a lease for other live arrays.','CSA independently clips causal selected count and binary-searches first negative TopK; metadata512 capacity/ceil count cannot substitute actual validK. EmptyKV retains zero-output/preload draining; no-query caller zeroes packet and may still produce cache.','CSA first/last loop preloads3 RunInfo records and2extra flushes; V0/C1/V1/C2/V2 cross flags and3-buffer KV ring have actual consumers. FD descriptors and FA extra2words are not consumed by selected arch22 kernels, but are defined output and cannot be silently removed under whole primitive bit gate.'],
  branch_witness=dict(scope='Original scalar bodies323-1631 with mocked tensor/entry ABI, not device gate',cases=cpu['cases'],disposition='Stop only flag-only level3->safe crossS2/FD recipe. q3 level3 emits6FA/3FD/48vec vs default3FA/noFD; q24 fixture did not split. Selected arch22 FD0/no FD descriptor consumer prevents blanket acceptance; not a measured service fault and no statement frozen service runs level3.'),
  source_workspace=dict(CSA_KV_ring_buffers=3,per_core_ring_BF16_bytes=3*512*512*2,conditional_C24_ring_bytes=24*3*512*512*2,FD_staging_arch22_source_bytes=0,misleading_unused_increment='CSA Init final offset +=4 buffers is not consumed by a following region. Host requests3 buffers; fourth-bank/12MiB-deletion claim has zero basis.',physical_release='No credit; actual ACLNN/LibAPI/system/pool/graph high-water and all readers remain required per loop033.'),
  paid_parent_selected=selected,max_K512_plan_payload_MiB=paid['max_C8_K512_plan_payload_MiB'],
  stopped=[dict(recipe='New metadata reuse per each of38layers',reason='Already shared by perbatch ratio keys; original producer/packet identity and event semantics retained. Extra duplicate38-call saving0.'),dict(recipe='Only trim packet FD/tail/unused scalar offset for independent1% gain',reason='No original FD payload computation to remove from selected FD0 body, tail is unwritten, flags/padding primitive gate remains; previous entire native-prefix112cycle free max.81838% scope remains valid only that sample. No need isolate more tiny same-budget plans.'),dict(recipe='Only set level3 and assume same arithmetic safe crossS2',reason='Original scalar q3 fixture emits split/FD data, selected arch22FD0 consumes no FD descriptors and does not forward batch-consistency arithmetic; requires new complete reduction/bit gate, not flag-only.')],
  confirmed_implementable=[],
  pending=[dict(candidate='Original-order selectedKV address/transfer plan + sparse Cube direct read + first nonfinal FP32 partial',gap='Keep original packet, MMA/softmax/FP16-BF16 boundaries. Need actual validK and matching sparse Cube issue/transaction rate, ND2NZ original L1 layout/padding/pair fallback, page/index/KV source generations, masks/ring/flags, full primitive bits and net cost.'),dict(candidate='Host BatchFrame/capture producer fusion or conditional C1-C2 packet sharing',gap='Default fully capped CPU fixtures900 words equal only; need complete effective intervals/query/core/Soc/deterministic/key/error obligations, untouched tail and full1024-output bit qualification, event/bank/last readers, actual host parent costs. No claim existing device-only sub-budget clips larger parent.'),dict(candidate='DSpark ori_sparse direct delivery',gap='Separate per-query topk_length/indices/invalid rows, metadata MbaseG vs denseSWA4queries, old graph persistent buffer/readiness and original primitive geometry; no C8 causal cost extrapolation.'),dict(candidate='FD/source qualification when level3 selected',gap='Actual per-call runtime deterministic level/key and loaded source agreement; complete legal new producer+consumer reduction and bit/cost before enabling. Current frozen baseline not identified as level3.')],
  next_entry='smla-direct-kv-entry.json',new_NPU_work=0,new_profile_collection=0,model_loads=0,service_requests=0,remote_writes=[],service_changes=[])
 (OUT/'smla-schedule-frontier.json').write_text(json.dumps(frontier,ensure_ascii=False,indent=2)+'\n')
 notes=dict(previous_goal_turn='progress',revalidated_HEAD='3e5f158f39edb7ae14b8406152f55ac5fc046f16',current_remote='baai-ssh --check171 hostname confirmed; source/catalog reads only via python -B -I -S',range_validation='All manifest ranges bounded before finish; body extraction untouched and source SHA current-matched. Guessed common.h/old frozen metadata directory/tiling.h paths absent; actual common_arch22 and live metadata/API catalog used.',correction=frontier['correction'],cost_scope='54 conditional paid scenarios/6 old model reproductions;13 negative request values retained. No new native performance or source identity as completeness claim.',local_CPU='clang++ only; no Torch/backend imports, no remote build/file write/model request.',whole_chain_complete=False)
 (OUT/'smla-schedule-exploration-notes.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(sources=len(rows),reviewed_nonempty=review['reviewed_nonempty'],CPU_cases=len(cpu['cases'])-1,paid_scenarios=len(paid['scenarios']),new_native_primitives=0)))

if __name__=='__main__':main()
