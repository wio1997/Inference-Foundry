from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import json,hashlib

R=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
N=Path(__file__).parent;OUT=N/'profile-first-20261031'
SRC=R/'whole-scope-20261030/frozen-source-snapshot'


def main():
    p=OUT/'cache-merge-layout-summary.json';summary=json.loads(p.read_text())
    receipt_path=N/'archived-ep3-cache-merge-evidence166.json';receipt=json.loads(receipt_path.read_text());assert receipt['returncode']==0
    db=json.loads(receipt['stdout']);assert db['database_sha256']=='084b8718888e33271435ec79d7573c95ca1615f3e7484c807ea24590fad234c4' and db['database_stat_unchanged']
    by_label={x['label']:x for x in db['items']};assert len(by_label)==7
    bindings=[]
    for label,x in by_label.items():
        assert len(x['api_candidates'])==1,(label,'API not unique')
        a=x['api_candidates'][0]
        bindings.append({'label':label,'API':a['name'],'callchain_id':a['callchainId'],'selected_frames':a['selected_frames'],
                         'frame_display_truncated':a['frame_display_truncated'],'exact_task_rows':x['exact_task_rows']})
    merge=next(x for x in bindings if x['label']=='F23_MERGE_L36')
    assert any('fused_moe.py(240): forward' in f['frame'] for f in merge['selected_frames'])
    assert len(db['F15']['communication_ops'])==1
    comm=db['F15']['communication_ops'][0];assert 'alltoall' in comm['operation_name'].lower() and comm['rankSize']==8
    source_specs=[
      (SRC/'vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py','2f42ce56a7636bf618204dbc2c1aeb2a2c95fed69fe9e5bcfac995354a00beb4',[(157,246),(277,357)]),
      (SRC/'vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py','05893188402fe37229bf72d75088d66193c417d5d502daafd3a55e1bfedaa60c',[(400,417),(665,679)]),
      (R/'sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py','101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91',[(262,330)]),
    ]
    sources=[]
    for path,pin,ranges in source_specs:
        raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==pin
        lines=raw.decode().splitlines()
        sources.append({'path':str(path),'sha256':pin,'reviewed_ranges':ranges,
                        'bounded_excerpts':[{'start':a,'end':b,'lines':[f'{i+1}: {lines[i]}' for i in range(a-1,b)]} for a,b in ranges]})
    t,h,d,tail,width_bytes=1023,64,512,64,2
    full=t*h*d*width_bytes;rotated=t*h*tail*width_bytes
    base={'generated_utc':datetime.now(timezone.utc).isoformat(),'whole_modules_closed':0,'overall_complete':False,
          'new_device_or_inference_work':False,'source_gate':str(OUT/'CACHE-MERGE-CODE-QUESTION-GATE.json'),
          'profile_summary':summary,'selected_callsite_bindings':bindings,'source_provenance':sources,
          'DB_SHA':db['database_sha256'],'DB_receipt':str(receipt_path),'full_DB_query_server_path':db['full_result'],
          'F23_observed_variant':{
              'scope':'Selected EP3/L36 recorded callsite, interpreted in the pinned frozen implementation; not a historical code-hash attestation for all calls.',
              'recorded_merge':'aclnnForeachAddListV2 at AscendMoERunner.forward:240',
              'frozen_graph':['routed/shared finalization and optional reductions','runner output scaling and routed output transform','torch._foreach_add([shared_output],[fused_output])[0]','final reduction/truncation hook','optional zero-expert output hook','Tensor returned to model; legacy tuple branch bypassed','decoder HC post'],
              'legacy_tuple_variant':'Retained as another source branch, not the graph for this pinned runner-return path.',
              'runtime_reduction_policy':'Ascend helper uses static layer sequence/early-reduction policy and context-aware custom ops. _fused_output_is_reduced depends on MoECommType; do not freeze a capture-time boolean as all-replay truth.',
              'scalar_value_bound_from_profile':False,
              'conditions_for_merge_into_HC':['preserve actual scaling/transform before add','preserve add output dtype rounding before HC FP32 math','retain or include post-add reduction, truncation and zero-expert effects','preserve shared/routed reduction domains and original HC accumulation order','all input tiles and residual/coefficients actually ready; no unproven external alias/consumer removal'],
          },
          'F11_contract':{
              'scopes_with_full_signature_chain':summary['F11']['source_scopes_with_key_projection'],
              'selected_API_stack_bindings':5,
              'preserve':['key128D BF16 projection and RMS/partial-RoPE boundaries','native dynamic quantization plus FP32-to-FP16 scale cast','both INT8 key and FP16 scale cache writes','real page stride/block table/slot mapping/invalid slots','original allowed outcomes for duplicate slots; consistent publication before actual QLI readers'],
              'unresolved':['runtime tensor/slot identity and duplicate indices','actual primitive rounding and loaded binary mapping','ready visibility, cache lifetime and cross-iteration aliases'],
          },
          'F14_F15_contract':{
              'retained_scope_candidates':summary['F14_F15'],
              'selected_full_DB_AlltoAll':comm,
              'sidecar_zero_communication_interpretation':'EXTRACTION_ASSOCIATION_GAP; selected full DB has AlltoAll and host _project_output callstack.',
              'joint_variant':'Inverse-RoPE plus valid-row padding and send-layout permutation; preserve token/head ownership, BF16 boundary, padding and communication ordering.',
              'example_domain':{'local_valid_rows':t,'heads':h,'head_dim':d,'rotated_tail_dim':tail,'dtype_bytes':width_bytes},
              'conditional_logical_IO_model':{
                  'assumptions':'Existing inverse-RoPE touches tail only; permutation materializes full tensor once. Model only, not measured traffic.',
                  'full_tensor_bytes':full,'rotated_tail_bytes':rotated,
                  'tail_read_reduction_if_fused_and_original_inplace_state_preserved':rotated,
                  'tail_read_plus_write_reduction_if_original_intermediate_is_unobserved':2*rotated,
                  'full_tensor_traffic_cannot_be_claimed_as_removed_by_tail_rotation_fusion':True,
                  'limits':'Keep original in-place state if alias readers require it; HBM/L2 traffic, overlapping work and latency gain unknown.'},
              'conditions':['inverse-RoPE pairing and BF16 rounding retained','padding rows written as original zeros without invalid cosine reads','same row/head permutation and group/collective protocol','stable projected output and metadata/cos-sin lifetimes','all original alias/state consumers preserved or proved unobservable'],
          },
          'limits':['Full sidecar candidate coverage is not full raw export/module/branch closure.',
                    'Only seven selected task/API/callchain anchors and one selected communication interval are bound to originalDB.',
                    'Frozen-source semantics and measured submission locations are separate evidence kinds.',
                    'No source branch or symbolic byte model is a measured fusion legality or speedup result.']}
    (OUT/'cache-merge-layout-reconciliation-v5.json').write_text(json.dumps(base,ensure_ascii=False,indent=2)+'\n')
    ledger=json.loads((OUT/'candidate-profile-ledger-v4.json').read_text())
    for c in ledger['candidates']:
        if c['id'] not in ['F11','F14','F15','F23']:continue
        c.setdefault('evidence_files',[]).append('cache-merge-layout-reconciliation-v5.json')
        if c['id']=='F11':
            c['profile_review_state']='ALL_RETAINED_KEY_CACHE_CHAINS_AND_SELECTED_UPDATE_KEYS_STACKS_REVIEWED'
            c['scoped_detail']=base['F11_contract'];c['next_profile_or_contract_action']='保留native quant和FP16 scale两个cache接口、stride与重复slot允许结果；实际slot/tensor/跨步版本仍是运行证据缺口。'
        elif c['id']=='F14':
            c['profile_review_state']='ALL_RETAINED_POSTATTENTION_ROPE_LAYOUT_CANDIDATES_REVIEWED'
            c['scoped_detail']=base['F14_F15_contract'];c['next_profile_or_contract_action']='与F15联合考察inverse-RoPE/有效行padding/send-layout；只按tail读写条件账，不把整矩阵流量或大段通信时间算可省。'
        elif c['id']=='F15':
            c['profile_review_state']='POSTATTENTION_CP_LAYOUT_PROFILE_AND_SELECTED_ALLTOALL_STACKS_REVIEWED'
            c['scoped_detail']=base['F14_F15_contract'];c['next_profile_or_contract_action']='侧车通信缺失已用原DB补正所选sample；保留TP8行/head所有权、pad、AlltoAll和输出shape/地址，完整跨rank集合与实际张量边仍未证。'
        else:
            c['profile_review_state']='SELECTED_RUNNER_FOREACH_RETURN_BRANCH_BOUND_AND_SOURCE_CONTRACT_REFINED'
            c['scoped_detail']=base['F23_observed_variant'];c['next_profile_or_contract_action']='按runner内部foreach→posthooks→Tensor→HC图继续，legacy tuple另列；不能跨缩放/归约/BF16舍入直接合并，不把capture动态comm标志当静态真值。'
        c['candidate_closed']=False;c['legal_fusion_proved']=False
    ledger['review_v5_utc']=base['generated_utc'];ledger['overall_complete']=False
    (OUT/'candidate-profile-ledger-v5.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n')
    modules=json.loads((OUT/'module-profile-ledger-v4.json').read_text())
    for m in modules['modules']:
        if m['id'] in ['M06','M09','M10','M11','M12','M13','M22','M23','M24']:
            m['profile_entry_points'].append('cache-merge-layout-reconciliation-v5.json')
            m['review_state']='PARTIAL_FULL_RETAINED_CHAIN_QUERY_AND_SELECTED_STACK_BRANCH_CONTRACT; module closure open'
    (OUT/'module-profile-ledger-v5.json').write_text(json.dumps(modules,ensure_ascii=False,indent=2)+'\n')
    report={'generated_utc':base['generated_utc'],'retained_scopes':summary['retained_scopes'],'F11_key_chains':128,
            'F23_merge_HC_pairs':1280,'post_attention_layout_intervals':1280,'exact_task_API_stack_bindings':7,
            'selected_AlltoAll_rankSize':comm['rankSize'],'runner_return_contract':'Tensor path under pinned implementation; legacy tuple separately retained',
            'review_states':dict(Counter(c['profile_review_state'] for c in ledger['candidates'])),
            'new_target_source_files':3,'whole_modules_closed':0,'overall_complete':False}
    (OUT/'cache-merge-layout-summary-v5.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
