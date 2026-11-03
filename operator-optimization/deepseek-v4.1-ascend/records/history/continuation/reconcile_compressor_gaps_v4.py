from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import hashlib,json

R=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
N=Path(__file__).parent;OUT=N/'profile-first-20261031'


def main():
    comp=json.loads((OUT/'compressor-profile-summary.json').read_text())
    receipt_path=N/'archived-ep3-gap-evidence166.json';receipt=json.loads(receipt_path.read_text());assert receipt['returncode']==0
    raw=json.loads(receipt['stdout']);assert raw['database_sha256']=='084b8718888e33271435ec79d7573c95ca1615f3e7484c807ea24590fad234c4' and raw['database_stat_unchanged']
    gaps=[]
    for x in raw['items']:
        a=x['api_a']['pytorch_api_candidates'];b=x['api_b']['pytorch_api_candidates']
        assert len(a)==len(b)==1 and a[0]['globalTid']==b[0]['globalTid']
        gaps.append({'label':x['label'],'device_exported_gap_ns':x['gap_ns'],
                     'all_task_record_union_ns':x['all_record_union_ns'],'overlap_task_count':x['overlap_task_count'],
                     'event_wait_like_records':x['event_wait_like_records'],'all_name_aggregates':x['all_name_aggregates'],
                     'same_host_thread_API_end_to_next_start_ns':int(b[0]['startNs'])-int(a[0]['endNs']),
                     'host_API_names':[a[0]['name'],b[0]['name']],
                     'host_device_clocks_subtracted':False,
                     'scope':'one archived rank; exported records only; short/long examples not equal-work experiments'})
    refs=['agents/fusion/F17-COMPRESSOR-DUAL-PROJECTION-PIPELINE.md','agents/fusion/COMPRESSOR-TILE-DEPENDENCY.md',
          'agents/fusion/COMPRESSOR-RING-WITNESS.md','agents/fusion/F19-LATENT-FORK-PUBLICATION-CUT.md',
          'agents/fusion/compressor-ring-conflict-witness.json','agents/fusion/f17-layer2-run2-evidence.json']
    base={'generated_utc':datetime.now(timezone.utc).isoformat(),'overall_complete':False,'whole_modules_closed':0,
          'new_target_source_read':False,'new_device_or_inference_work':False,
          'remote_work':'read-only archived DB SHA verification and SQLite TASK/API queries; original DB stat unchanged',
          'profile':comp,'gap_comparisons':gaps,'raw_full_result_server':raw['full_result'],
          'receipt':str(receipt_path),'receipt_sha256':hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
          'reused_local_contracts':[{'path':str(R/p),'sha256':hashlib.sha256((R/p).read_bytes()).hexdigest()} for p in refs],
          'scoped_decisions':[
              {'id':'PF-F17-CONCAT-STRIDE','status':'EXCLUDED_NAIVE_VARIANT_IN_FROZEN_HELPER_DOMAIN',
               'statement':'For T>1, slicing a row-major[T,1024] fused projection into two[T,512] column views yields row stride1024. Existing pool/ring helper requires contiguous row stride512; cannot pass these views unchanged.',
               'limits':'Does not exclude two contiguous outputs, stride-aware consumers or unpacking; no measured performance gain.'},
              {'id':'PF-F18-STATE-ORDER','status':'RETAIN_CONDITIONAL_WAR_AND_WAW_DEPENDENCIES',
               'statement':'Pool may read old ring values before update; tail new writes can target the same32-slot ring. Preserve request/page/slot/version ordering and original FP32 projected values.',
               'limits':'Witnesses are hypothetical source semantics, not observed conflicts; disjoint states may permit overlap, not universal whole-kernel serialization.'},
              {'id':'PF-F18-POOL-NORM','status':'CONDITIONAL_BOUNDARY_WITH_CROSS_CHANNEL_REDUCTION',
               'statement':'Channel-specific weighted pool and subsequent512-channel RMS require complete pooled statistics; input per-token sums of squares alone do not determine pooled RMS. Preserve pooled BF16 rounding and zero/incomplete rows.',
               'limits':'Does not preclude exact channel accumulation or fused row reduction; resource/consumer closure remains open.'},
              {'id':'PF-F19-DUAL-PUBLISH','status':'RETAIN_BOTH_PERSISTENT_OUTPUT_VERSIONS',
               'statement':'Index key and scale must be mutually consistent before QLI; long-KV before its attention reader. Index projection reads pre-long-RoPE latent, so cross-stream long mutation needs actual read completion or nonmutating representation.',
               'limits':'No actual slot/tensor/event identity measured; static source layer consumers do not prove every row is read.'},
              {'id':'PF-F19-VALID-ROWS','status':'OPEN_RUNTIME_DOMAIN_GAP',
               'statement':'Pool output is token-aligned with completed-group rows, separate physical compressed slots and source positions. Cannot infer valid completed rows or simply remove half postprocessing from physical M or ratio2.',
               'limits':'A compact valid-row variant needs actual validity/mapping, zero-row observability, duplicate slots, scales, cache lifetime and graph-address contracts.'},
              {'id':'PF-GAP-LABELS','status':'SIMPLE_WAIT_OR_LAUNCH_GAP_EXPLANATION_NOT_ESTABLISHED',
               'statement':'Long F22/F17 gaps lack EVENT WAIT-like TASK labels in this DB; short F22 interval overlaps WAIT records on other streams. Host API gaps differ from device gaps and are not directly convertible.',
               'limits':'No physical idle, absent communication, runtime scheduling cause, or removable time is proved; unexported work/other processes and capture contracts remain open.'},
          ],
          'joint_C2_interface_model':{'domain':'Conditional row-independent projections before equal-size gather; preserve SWA BF16, both compressor FP32 outputs, ring/metadata/layout and numerical contracts',
                                       'original_BF16_hidden_bytes_per_row':5120*2,
                                       'SWA_BF16_plus_two_FP32_projection_bytes_per_row':512*2+2*512*4,
                                       'logical_feature_payload_ratio':2,
                                       'example_original_bytes_per_rank_1023rows':1023*5120*2,
                                       'example_joint_bytes_per_rank_1023rows':1023*(512*2+2*512*4),
                                       'limits':'Pure conditional interface byte accounting, not network/HBM traffic, a universal lower bound, measured savings or2x speedup. Packing/call overhead, weight/row ownership and changed-M numerical effects unproved. C1 is a separate graph.'}}
    (OUT/'compressor-gap-reconciliation-v4.json').write_text(json.dumps(base,ensure_ascii=False,indent=2)+'\n')
    ledger=json.loads((OUT/'candidate-profile-ledger-v3.json').read_text())
    for c in ledger['candidates']:
        if c['id'] not in ['F17','F18','F19','F22','F32']:continue
        c.setdefault('evidence_files',[]).append('compressor-gap-reconciliation-v4.json')
        if c['id']=='F17':
            c['profile_review_state']='ALL_RETAINED_C2_C1_ROLE_SCOPES_REVIEWED_LOCAL_LAYOUT_CONTRACT_REUSED'
            c['scoped_detail']=comp['C2_sources'];c['next_profile_or_contract_action']='保留双FP32输出与ring消费者；直接将[T,1024]列切片交现有contiguous helper的变体不适用；联合F32沟通域与布局资源账。'
        elif c['id']=='F18':
            c['profile_review_state']='C2_POOL_UPDATE_NORM_PROFILE_REVIEWED_STATE_AND_ROUNDING_CONTRACT_REUSED'
            c['scoped_detail']={'C2':comp['C2_sources'],'valid_completed_rows_measured':False};c['next_profile_or_contract_action']='保持old-read→new-write、尾部写写版本、BF16池化与全行RMS；复核实际有效完成行metadata，不用ratio或物理M推省计算比例。'
        elif c['id']=='F19':
            c['profile_review_state']='LATENT_FORK_PROFILE_INTAKE_AND_PERSISTENT_CONSUMER_CONTRACT_REUSED'
            c['scoped_detail']={'C2':comp['C2_sources'],'C1':comp['C1_sources'],'runtime_tensor_cache_slot_edge_proved':False};c['next_profile_or_contract_action']='保留index key+scale、long-KV两发布和原latent读完成；位置/slot/完成行分域，继续复用F11既有profile核对双写候选。'
        elif c['id']=='F22':
            c['scoped_detail']['archived_DB_gap_review']=gaps[:2];c['next_profile_or_contract_action']='已补长/短间隔全TASK查询。长间隔未由WAIT标签解释，不认定真实idle/可删CPU开销；保留调度/采集完整性缺口，继续其他候选与数值合同。'
        elif c['id']=='F32':
            c['scoped_detail']['C2_joint_interface_model']=base['joint_C2_interface_model']
        c['candidate_closed']=False;c['legal_fusion_proved']=False
    ledger['review_v4_utc']=base['generated_utc'];ledger['overall_complete']=False
    (OUT/'candidate-profile-ledger-v4.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n')
    modules=json.loads((OUT/'module-profile-ledger-v3.json').read_text())
    for m in modules['modules']:
        if m['id'] in ['M07','M08','M09','M10','M13','M23','M24']:
            m['profile_entry_points'].append('compressor-gap-reconciliation-v4.json')
            m['review_state']='PARTIAL_FULL_RETAINED_ROLE_PROFILE_AND_LOCAL_STATE_CONTRACT; runtime/branch/resource closure remains open'
    (OUT/'module-profile-ledger-v4.json').write_text(json.dumps(modules,ensure_ascii=False,indent=2)+'\n')
    summary={'generated_utc':base['generated_utc'],'candidate_count':32,'module_surface_count':26,
             'C2_scopes':comp['C2_sources']['scope_count'],'C1_scopes':comp['C1_sources']['scope_count'],
             'archived_DB_SHA_verified':raw['database_sha256'],'gap_comparisons':gaps,
             'review_states':dict(Counter(c['profile_review_state'] for c in ledger['candidates'])),
             'local_contracts_reused':len(refs),'whole_modules_closed':0,'overall_complete':False,
             'scope_warning':'Review states are not candidate or module completion percentages.'}
    (OUT/'compressor-gap-summary-v4.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
