"""Bounded profile supplement and fixed-source role ledger, no imports of model code."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json

R = Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
OUT = Path(__file__).parent/'profile-first-20261031'
SRC = R/'sources/vllm-workspace/vllm-ascend/vllm_ascend'
PINS = {
 'models/deepseek_v41/model.py':('101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91',[(450,490),(512,562),(760,768),(880,930)]),
 'attention/dsa_v41.py':('3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd',[(289,410),(497,600),(690,716)]),
 'attention/context_parallel/dsa_v41_cp.py':('039ec9d18e0a8a4aa39f9d5b4bea15204cea3f397a3fce97a5097b266bc9822e',[(173,227)]),
}


def main():
    sources=[]
    for rel,(expected,ranges) in PINS.items():
        p=SRC/rel;raw=p.read_bytes();actual=hashlib.sha256(raw).hexdigest()
        assert actual==expected,(rel,actual)
        lines=raw.decode().splitlines()
        sources.append({'path':str(p),'sha256':actual,'reviewed_ranges':ranges,
                        'bounded_excerpts':[{ 'start':a,'end':b,'lines':[f'{i+1}: {lines[i]}' for i in range(a-1,b)]} for a,b in ranges]})
    config_path=R/'evidence/model-config.json'
    config=json.loads(config_path.read_text());config=config.get('text_config',config)
    n=config['num_hidden_layers'];ratios=config['compress_ratios'][:n];kv=set(config['kv_source_layer_ids']);index=set(config['index_source_layer_ids'])
    assert n==len(ratios)==40
    roles=[{'layer':i,'compress_ratio':ratios[i],'is_kv_source':i in kv,'is_index_source':i in index,
            'extra_global_raw_hidden_consumer_in_reviewed_path':'C1_COMPRESSOR' if i in kv and ratios[i]==1 else 'C2_FP32_COMPRESSOR_WKV_WGATE' if i in kv else 'NONE_BEYOND_SWA_KV_IN_REVIEWED_PATH',
            'input_gather_replacement':'RETAIN_OR_EQUIVALENTLY_REORGANIZE_COMPRESSOR_CONSUMER' if i in kv else 'CONDITIONAL_LOCAL_Q_AND_PROJECTED_SWA_KV_COMMUNICATION_CANDIDATE'} for i in range(n)]
    side=R/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
    f10=[];matches=0
    for line in side.open():
        obj=json.loads(line)
        if obj.get('kind')!='rank' or obj['data']['ep']!=3:continue
        for layer in obj['data']['layers']:
            if layer['layer_scope_id']!='2128-42':continue
            matches+=1
            for group in layer['operator_groups']:
                for t in group['tasks']:
                    c=t['csv'];ins=c['Input Shapes'].strip().strip('"').split(';');outs=c['Output Shapes'].strip().strip('"').split(';')
                    if c['Type'] not in ['MatMulV2','MatMulV3','Cast','Mul'] or '1023,32' not in ins+outs:continue
                    f10.append({'task_id':c['Task ID'],'stream':c['Stream ID'],'type':c['Type'],'flow':t['flow_id'],
                                'input_shapes':c['Input Shapes'],'output_shapes':c['Output Shapes'],
                                'input_dtypes':c['Input Data Types'],'output_dtypes':c['Output Data Types'],
                                'start_ns':int(Decimal(t['device_start_us'])*1000),'duration_ns':int(Decimal(t['trace_duration_us'])*1000)})
    assert matches==1
    f10.sort(key=lambda x:x['start_ns'])
    base={'generated_utc':datetime.now(timezone.utc).isoformat(),'overall_complete':False,'whole_modules_closed':0,
          'new_device_work':False,'target_source_review':'3 explicitly gated files; reviewed body ranges only; no repository-wide audit',
          'new_source_paths':list(PINS),'source_provenance':sources,
          'config_path':str(config_path),'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
          'scope':'Frozen legacy CP Python path and declared config; historical tensor/instance identity and opaque primitive semantics remain separate.',
          'delegated_contracts':['F06-CP-SHARED-INPUT-ROW-DOMAINS.md for the multistream Q/KV wrapper branch; full primitive effects remain open'],
          'roles':roles,'role_counts':dict(Counter(x['extra_global_raw_hidden_consumer_in_reviewed_path'] for x in roles)),
          'F10_all_stream_candidates':f10,
          'profile_limits':'Cast/Mul matching still uses shape/time/stream; numerical scalar and tensor edges not measured.',
          'conditional_payload_model':{'preconditions':['equal-size allgather semantics','8ranks','1023 physical rows per rank retained','BF16 interface','projection before communication preserves each row numerical contract','all other value consumers and padding/row order retained'],
                                       'original_interface_elements_per_rank':1023*5120,'projected_swa_kv_elements_per_rank':1023*512,
                                       'original_logical_bytes_per_rank':1023*5120*2,'projected_swa_kv_logical_bytes_per_rank':1023*512*2,
                                       'feature_width_ratio':10,
                                       'meaning':'Conditional interface payload only, not measured network bytes, universal lower bound or10x speedup.'},
          'branch_contracts':[
              {'id':'PF-F32-NO-METADATA','guard':'get_forward_context().attn_metadata is None',
               'fact':'Reviewed DSA forward zeros the supplied output and returns before Q/KV preparation; no positive projected-KV payload model applies. Outer model collectives and startup ordering remain separate obligations.'},
              {'id':'PF-F32-NONEMPTY','guard':'query_hidden has rows; cache_hidden has rows',
               'fact':'Local query feeds Q and indexer weights; global cache_hidden feeds SWA KV and additionally compressed-source generation iff is_kv_source.'},
              {'id':'PF-F32-EMPTY-QUERY','guard':'query_hidden has zero rows; cache_hidden has rows',
               'fact':'_update_caches still projects global SWA KV and writes compressed source iff is_kv_source. No local Q is not permission to discard global cache work.'},
              {'id':'PF-F32-CACHE-EMPTY','guard':'cache_hidden has zero rows',
               'fact':'No nonempty global cache payload is modeled; preserve local query, output shape and other branch behavior. No positive gather saving inferred.'},
              {'id':'PF-F32-SHAPE-CONSUMERS','guard':'reviewed CP forward',
               'fact':'Sparse index selection and output projection retain hidden/qr row extents and cache metadata. Changing value representation must preserve these extents and graph-stable output address.'},
          ],
          'candidate_relations':[
              {'from':'F32:replace-hidden-gather-by-SWA-KV-only','relation':'requires','to':'preserve_or_reorganize_compressor_hidden_consumer',
               'scope':'KV-source layers2/8/14/20 in frozen40-layer config; links F17/F19/F08/F06',
               'reason':'SWA projection alone does not provide the raw hidden input consumed by C1 or C2 compressor.'},
              {'from':'F32','relation':'overlaps','to':'F06','scope':'shared local/global input and row ownership',
               'reason':'Do not add shared-input IO savings to communication replacement savings without recomputing the joint graph.'},
          ]}
    (OUT/'projection-source-contract.json').write_text(json.dumps(base,ensure_ascii=False,indent=2)+'\n')
    profile=json.loads((OUT/'projection-pipeline-summary.json').read_text())
    ledger=json.loads((OUT/'candidate-profile-ledger-v2.json').read_text())
    states={
      'F06':'SELECTED_ROW_DOMAIN_PROFILE_AND_LOCAL_CONTRACT_REVIEWED',
      'F10':'SELECTED_PROJECTION_STACK_AND_CAST_SCALE_CANDIDATES_REVIEWED',
      'F16':'ALL_RETAINED_TARGET_SCOPE_PAIRS_REVIEWED_NUMERICAL_BOUNDARY_OPEN',
      'F22':'ALL_RETAINED_SHARED_EXPERT_CHAINS_REVIEWED_INTERNAL_READY_OPEN',
      'F32':'SELECTED_GATHER_PROJECTION_PROFILE_AND_ROLE_DEPENDENT_CONTRACT_REVIEWED',
    }
    for c in ledger['candidates']:
        if c['id'] not in states:continue
        c['profile_review_state']=states[c['id']]
        c['evidence_files']=['projection-pipeline-summary.json','projection-source-contract.json','PROJECTION-CODE-QUESTION-GATE.json']
        if c['id']=='F06':
            c['scoped_detail']=profile['F06_F32'];c['next_profile_or_contract_action']='当前物理行域不相等，直接共用同一个M矩阵拼权重的变体不适用；保留重映射、重算及投影前移变体，按行别名/数值/通信合同复核。'
        elif c['id']=='F10':
            c['scoped_detail']={'all_stream_selected_scope_candidates':f10,'scalar_value_measured':False,'tensor_edges_proved':False}
            c['next_profile_or_contract_action']='保留BF16投影舍入、后续FP32缩放和非packed FP16接口；复用已有有限值等价条件及跨分支scale移动反例，不把形状链当数值相等。'
        elif c['id']=='F16':
            c['scoped_detail']=profile['F16'];c['next_profile_or_contract_action']='1280对已导出间隔20–41ns，不能以消除大段两投影间空洞解释收益；继续中间BF16/布局/访存及collective的联合语义与资源账。'
        elif c['id']=='F22':
            c['scoped_detail']=profile['F22'];c['next_profile_or_contract_action']='已有DequantSwigluQuant不计新增融合；进一步查activation→down的最长间隔所含导出任务/事件，区分观察空洞与真实依赖；保持量化scale和半区分块合同。'
        elif c['id']=='F32':
            c['scoped_detail']={'profile':profile['F06_F32'],'role_counts':base['role_counts'],'role_contract':'projection-source-contract.json'}
            c['next_profile_or_contract_action']='36个非KV-source层与4个KV-source层分开考察；前者仍需行映射/数值/输出地址合同，后者替换hidden gather必须同时保留或重组F17/F19压缩消费者。'
        c['target_source_gate']='Specific profile gap recorded; relevant local source/precision contracts reviewed or reused. No whole-repository scan.'
        c['candidate_closed']=False;c['legal_fusion_proved']=False
    ledger['projection_review_utc']=base['generated_utc'];ledger['overall_complete']=False
    (OUT/'candidate-profile-ledger-v3.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n')
    modules=json.loads((OUT/'module-profile-ledger-v2.json').read_text())
    for m in modules['modules']:
        if m['id'] in ['M07','M08','M09','M10','M11','M13','M23','M24']:
            m['profile_entry_points']+=['projection-pipeline-summary.json','projection-source-contract.json']
            m['review_state']='PARTIAL_PROFILE_PIPELINE_AND_GATED_LOCAL_CONTRACT_REVIEW; complete branch/consumer/resource closure open'
    (OUT/'module-profile-ledger-v3.json').write_text(json.dumps(modules,ensure_ascii=False,indent=2)+'\n')
    summary={'generated_utc':base['generated_utc'],'candidate_count':32,'module_surface_count':26,
             'review_states':dict(Counter(c['profile_review_state'] for c in ledger['candidates'])),
             'F16_pairs':profile['F16']['paired_scopes'],'F16_gap_ns':profile['F16']['gap_ns'],
             'F22_chains':profile['F22']['three_task_scopes'],'F10_all_stream_candidate_count':len(f10),
             'frozen_target_role_counts':base['role_counts'],'new_target_source_files':3,
             'scope_warning':'Counts describe evidence review maturity, not optimization or module completion.',
             'whole_modules_closed':0,'overall_complete':False}
    (OUT/'projection-review-summary-v3.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    main()
