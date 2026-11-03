"""Bind current V4.1 QB semantics to native owners/CPU callchains.

This corrects v74's legacy-forward applicability error without discarding its
valid conditional mathematics. No target execution, imports or source changes.
"""
from collections import Counter,defaultdict
from copy import deepcopy
from pathlib import Path
import ast,hashlib,sqlite3,time
from finalize_origin_coverage_v9 import N,O,W,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_state_engram_v62 import fullkey
from analyze_projection_ceiling_v61 import stat


def rows_for_ids(c,table,columns,field,ids):
    out=[];ids=sorted(set(ids))
    for i in range(0,len(ids),800):
        batch=ids[i:i+800]
        out+=c.execute('select '+columns+' from '+table+' where '+field+' in('+','.join('?'for _ in batch)+')',batch).fetchall()
    return out


def main():
    assert not(O/'V41-QB-native-source-binding-v75.json').exists()
    source_root=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend'
    files=[
        (N/'original-source-cache-v53/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py','101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91'),
        (source_root/'models/deepseek_v41/dspark.py','fcee256905928917154d8dac94d6611bb83d02da441e8aa154a525ef51e6a601'),
        (source_root/'attention/context_parallel/dsa_v41_cp.py','039ec9d18e0a8a4aa39f9d5b4bea15204cea3f397a3fce97a5097b266bc9822e'),
        (N/'attention-operator-source-v60/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py','3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd'),
        (N/'projection-operator-source-v61/vllm_ascend/attention/context_parallel/dsa_cp.py','7e8dd93eeda988866138342523ecfa7cc9c42ce67669d4b1507ee8c5f3452305'),
    ]
    pins=[];texts=[];trees=[]
    for p,sha in files:
        assert hashlib.sha256(p.read_bytes()).hexdigest()==sha,p
        s=p.read_text();texts.append(s);trees.append(ast.parse(s));pins.append(dict(path=str(p),sha256=sha,new_source_file_obtained=False))
    assert read(O/'M21-original-static-index-hash-validation-v41.json')['source_pins']['vllm-ascend/vllm_ascend/attention/context_parallel/dsa_v41_cp.py']==files[2][1]
    def class_node(tree,name):return next(x for x in ast.walk(tree)if isinstance(x,ast.ClassDef)and x.name==name)
    def method(node,name):return next(x for x in node.body if isinstance(x,ast.FunctionDef)and x.name==name)
    model=class_node(trees[0],'DeepseekV41Attention');draft=class_node(trees[1],'DeepseekV41DSparkAttention')
    cp=class_node(trees[2],'AscendDSAV41CPImpl');rep=class_node(trees[2],'_ReplicatedCacheImpl');base=class_node(trees[3],'AscendDSAV41Impl')
    assert [ast.unparse(x)for x in cp.bases]==['_ReplicatedCacheImpl']
    assert [ast.unparse(x)for x in rep.bases]==['AscendDSAV41Impl']and not base.bases
    assert 'torch.ops.vllm.dsa_v41_forward' in ast.unparse(method(model,'forward'))
    assert any(isinstance(x,ast.Assign)and ast.unparse(x)=='forward = DeepseekV41Attention.forward'for x in draft.body)
    assert 'get_v41_cp_classes()[1]'in ast.unparse(method(model,'__init__'))and'get_v41_cp_classes()[1]'in ast.unparse(method(draft,'__init__'))
    assert '_forward'not in {x.name for x in base.body if isinstance(x,ast.FunctionDef)}
    assert 'apply_dsa_q_rms'not in texts[3]and'q_norm_without_weight'not in texts[3]
    preprocess=method(base,'_preprocess');multi=method(base,'multistream_preprocess')
    assert 'attn.q_norm(attn.wq_a(query_hidden))'in ast.unparse(preprocess)
    assert 'attn.wq_b(qr)'in ast.unparse(preprocess)
    assert 'qr = attn.q_norm(q_a)'in ast.unparse(multi)
    assert 'q = wq_b.matmul' in ast.unparse(multi)
    assert 'self._apply_rotary(attn, q, cos, sin)'in ast.unparse(preprocess)
    assert 'impl._forward_o_proj' in ast.unparse(method(cp,'_project_output'))
    assert 'apply_dsa_q_rms' in texts[4]
    contract=dict(source_files=pins,
                  prior_cp_file_pin_evidence='M21-original-static-index-hash-validation-v41.json',
                  target_forward_entry=dict(class_name='DeepseekV41Attention',lines=[760,763],calls='torch.ops.vllm.dsa_v41_forward'),
                  draft_forward_entry=dict(class_name='DeepseekV41DSparkAttention',lines=[102,134],alias='DeepseekV41Attention.forward'),
                  V41_CP_MRO=['AscendDSAV41CPImpl','_ReplicatedCacheImpl','AscendDSAV41Impl'],
                  active_chain=['target/draft V41 forward','dsa_v41_forward→attn.v41_impl.forward','AscendDSAV41Impl.forward','CP _prepare_inputs_and_caches','AscendDSAV41Impl._preprocess','ordinary or multistream Q_A→RMS1280→quant→Q_B BF16','partial64 query RoPE'],
                  both_V41_preprocess_branches_have_no_post_QB_head_RMS=True,
                  legacy_CP_forward_is_not_in_V41_query_chain=True,
                  legacy_object_reused_for_output_projection_only_in_CP_project_output=True,
                  alternative_PCP_or_plain_V41_factory_paths_inherit_V41_preprocess=True,
                  v74_source_applicability_error='Used legacy dsa_cp.AscendDSACPImpl._forward QB→headRMS512→RoPE as if it constrained current V4.1 query preparation. Target/draft forward dispatches to independent V4.1 preprocess; legacy output projector reuse does not activate legacy query _forward.',
                  post_QB_head_RMS_mandatory_in_frozen_V41_case=False,
                  legacy_head_RMS_math_remains_valid_only_in_its_own_program_domain=True,
                  loaded_native_ELF_attested=False,new_remote_source_read=False,overall_complete=False)
    # Select all4367QB native executions by exact existing functional/signature keys.
    regimes=read(O/'operator-shape-ceiling-ledger-v55.json')['observed_regimes']
    selected=[r for r in regimes if r['module_candidates']==['M07']and r['operator']=='QuantBatchMatmulV3']
    assert len(selected)==8 and sum(r['executions']for r in selected)==4367
    lookup={fullkey(r['module_candidates'],r['operator'],r['role'],r['origin'],r['context'],r['signature']):r for r in selected}
    assigns=defaultdict(dict)
    for a in read(CACHE/'analysis/native-PMU-complete-v18/resource-functional-attribution-and-uncertainty-v20-v3.json')['execution_assignments']:assigns[a[0]][a[1]]=a
    records=[];unbound=[];graphs=[];frame_counts=Counter();contexts=Counter();origin=Counter();databases=[];parent_sources=Counter()
    for db in range(32):
        started=time.monotonic();e=read(CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json')
        taskmap={t['task_record_rowid']:t for t in e['TASK_records']};defs={d['globalTaskId']:d for d in e['COMPUTE_definitions']}
        chosen=[]
        for tid,a in assigns[db].items():
            t=taskmap[tid];d=defs[t['globalTaskId']];key=fullkey(a[3],d['opType_text'],a[5],a[2],a[6],d)
            if key in lookup:chosen.append((t,d,lookup[key]))
        rpath=Path(e['database_path']);i=max(i for i,x in enumerate(rpath.parts)if x=='analysis');path=CACHE/Path(*rpath.parts[i:])
        c=sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True);c.execute('pragma query_only=1')
        query_started=time.monotonic();c.set_progress_handler(lambda:time.monotonic()-query_started>20,10000)
        nameids=[r[0]for r in c.execute("select id from STRING_IDS where value='npu::npu_quant_matmul'")]
        parents=rows_for_ids(c,'PYTORCH_API','rowid,startNs,endNs,globalTid,callchainId,inputShapes','name',nameids)
        parents=[r for r in parents if r[4]is not None]
        frame_rows=rows_for_ids(c,'PYTORCH_CALLCHAINS','id,stackDepth,stack','id',[r[4]for r in parents])
        string_ids=[r[2]for r in frame_rows]+[r[5]for r in parents if r[5]is not None]
        strings=dict(rows_for_ids(c,'STRING_IDS','id,value','id',string_ids));chains=defaultdict(list)
        for chain,depth,sid in frame_rows:chains[chain].append((depth,strings[sid]))
        parent_index=defaultdict(list)
        for row in parents:
            pid,start,end,tid,chain,shape=row
            parent_index[tid].append(dict(PYTORCH_API_rowid=pid,startNs=int(start),endNs=int(end),globalTid=tid,callchainId=chain,input_shapes=strings.get(shape),frames=[v for _,v in sorted(chains[chain])]))
        con_to_set=defaultdict(set);api_by_set=defaultdict(list)
        for r in e['CONNECTION_IDS']:con_to_set[r['connectionId']].add(r['id'])
        for r in e['linked_PYTORCH_APIs']:api_by_set[r['connectionId']].append(r)
        for task,d,r in chosen:
            origin[r['origin']]+=1
            if r['origin']=='GRAPH':
                graphs.append(dict(database_index=db,TASK_rowid=task['task_record_rowid'],regime_id=r['regime_id'],scope='Existing graph-role/template/source inference retained; no new per-replayed-task CPU frame or phase binding invented.'));continue
            candidates={a['api_rowid']:a for setid in con_to_set[task['connectionId']]for a in api_by_set[setid]if a['name_text']=='aclnnQuantMatmulWeightNz'}
            if len(candidates)!=1:
                unbound.append(dict(database_index=db,TASK_rowid=task['task_record_rowid'],reason='nonunique_native_API_owner',candidate_count=len(candidates)));continue
            owner=next(iter(candidates.values()))
            ps=[p for p in parent_index[owner['globalTid']]if p['startNs']<=owner['startNs']and owner['endNs']<=p['endNs']]
            if len(ps)!=1:
                unbound.append(dict(database_index=db,TASK_rowid=task['task_record_rowid'],reason='nonunique_enclosing_npu_quant_API',candidate_count=len(ps)));continue
            p=ps[0];frames=p['frames']
            v41=[s for s in frames if '/attention/dsa_v41.py('in s]
            legacy=[s for s in frames if '/attention/context_parallel/dsa_cp.py('in s and s.endswith(': _forward')]
            assert v41 and not legacy,(db,task['task_record_rowid'],frames)
            assert any(s.endswith(': _preprocess')for s in v41)
            assert any(s.endswith(': forward')for s in v41)and any(s.endswith(': dsa_v41_forward')for s in v41)
            query_frames=[s for s in v41 if s.endswith(': multistream_preprocess')or s.endswith(': _preprocess')]
            for frame in query_frames:frame_counts[frame]+=1
            contexts[r['context']]+=1
            for frame in frames:
                if '/models/deepseek_v41/dspark.py('in frame:parent_sources['with_DSpark_frame']+=1;break
            records.append(dict(database_index=db,TASK_rowid=task['task_record_rowid'],regime_id=r['regime_id'],context=r['context'],signature=d,native_API_table='PYTORCH_API',native_API_rowid=owner['api_rowid'],native_API_name=owner['name_text'],native_owner_connection_SET=owner['connectionId'],native_task_connectionId=task['connectionId'],owning_CPU_quant_API_rowid=p['PYTORCH_API_rowid'],owning_CPU_quant_API_name='npu::npu_quant_matmul',callchainId=p['callchainId'],input_shapes=p['input_shapes'],operator_relevant_frames=[s for s in frames if '/attention/'in s or '/models/deepseek_v41/'in s],V41_query_path_proved=True,legacy_query_forward_in_chain=False,source_program_post_QB_head_RMS_required=False,ownership_scope='TASK runtime connection→CONNECTION_IDS set→exact aclnn API; enclosing same-thread npu quant operator interval supplies its captured CPU callchain. Not temporal adjacent pairing.'))
        c.close();databases.append(dict(database_index=db,parent_quant_APIs=len(parents),selected_QB_tasks=len(chosen),elapsed_seconds=round(time.monotonic()-started,6)))
    assert origin=={'GRAPH':3200,'DIRECT':1167}
    assert len(records)+len(unbound)==1167
    # This audit reports any unmatched source binding honestly; do not promote it.
    write(O/'V41-query-operator-path-contract-v75.json',contract)
    write(O/'V41-QB-native-source-binding-v75.json',dict(native_QB_total=4367,direct_native_QB=1167,graph_native_QB=3200,bound_direct_TASKs=len(records),unbound_direct_TASKs=unbound,records=records,graph_records=graphs,direct_source_frame_counts=dict(frame_counts),direct_contexts=dict(contexts),DSpark_frame_population=dict(parent_sources),databases=databases,all_bound_direct_use_V41_preprocess=True,legacy_head_RMS_native_work_added=0,global_functional_assignments_or_phase_bindings_changed=False,overall_complete=False))
    # Re-price ONLY the native V4.1 QB→tail-RoPE path; old head-RMS diagrams
    # remain available as explicitly legacy-domain examples.
    models=[]
    for m in [2,3,800,816,830,834]:
        H,K,out,tail=64,1280,512,64
        full=m*H*out;rot=m*H*tail
        weights=K*H*out;channel=2*H*out;row=4*m;input_q=m*K;cos_sin=8*m*tail
        base=weights+channel+row+input_q+2*full+4*rot+cos_sin
        joint=base-4*rot
        compute=2*m*K*H*out;cube=compute/752000000
        models.append(dict(M=m,query_heads=H,head_width=out,rotated_tail=tail,QB_INT8_direct_dense_ops=compute,INT8_weight_bytes=weights,BF16_channel_scale_bytes=channel,FLOAT_row_scale_bytes=row,INT8_QR_input_bytes=input_q,cos_sin_reference_bytes=cos_sin,full_BF16_query_output_bytes=2*full,separate_QB_RoPE_reference_bytes=base,joint_QB_RoPE_reference_bytes=joint,tail_interface_reference_budget_bytes=4*rot,head_RMS512_work_in_active_V41_source=0,Cube_INT8_752TOP_reference_us=cube,separate_HBM_1p6TB_reference_us=base/1600000,joint_HBM_1p6TB_reference_us=joint/1600000,separate_two_resource_reference_us=max(cube,base/1600000),joint_two_resource_reference_us=max(cube,joint/1600000),Vector_capacity_and_quant_postprocess_reference=None,actual_HBM_or_request_gain=None,domain='Cold raw inputs/weights/scales/C-S and fresh complete BF16 query output;direct dense INT8 GEMM as stated algorithm. No universal lower bound or physically observed byte count.'))
    # Output N tiling may retain each rotary partner on the same producer;
    # query attention still consumes full512 for the score dot product.
    tile_cases=[]
    for tile in [16,32,64,128,256,512]:
        split_pairs=0;tail_pairs=0
        for h in range(64):
            for d in range(448,512,2):
                a=h*512+d;b=a+1;tail_pairs+=1
                split_pairs+=a//tile!=b//tile
        assert split_pairs==0
        tile_cases.append(dict(output_column_tile=tile,rotary_pairs=tail_pairs,split_rotary_pairs=split_pairs,illustrative_supported_layout_not_native_tiling_attestation=True))
    resource=dict(status='CONDITIONAL_CEILING',shared_Cube_mode_model='If direct INT8 QB and BF16 attention both use the same Cube pool, with mutually exclusive per-core modes and stated peak ratios, T_Cube >= F_INT8/P_INT8_upper + F_BF16/P_BF16_upper;not merely max of those two isolated demands.',required_conditions=['Same physical Cube pool and matching per-die/count/clock/dense rate domain','All counted MAC work necessary in the declared direct algorithms','No alternative nonCube implementation included in this restricted domain','No overlap double counting and no native attention dtype assumed from shape'],more_general_model='For a certified attainable-capacity outer region, derive a nonnegative dual resource price;the supplied two mode sum is conditional on no simultaneously higher mixed-mode capacity.',full_bound='T>=max(L_sharedCube,Q_HBM_necessary/B_HBM_upper,L_Vector_special_and_conversion,L_network_or_dependency_if_proved). Unknown typed terms remain symbolic.',source_peak_reference='https://arxiv.org/html/2506.12708v1#S3.SS3.SSS1',actual_request_numeric_ceiling=None,actual_native_attention_Cube_mode_or_mixed_capacity_attested=False)
    write(O/'V41-QB-RoPE-work-and-bound-models-v75.json',dict(models=models,producer_tile_partner_checks=tile_cases,active_V41_head_norm_512_work=0,legacy_v74_head_norm_models_not_applied_to_active_case=True,operator_fusion_conditions=['Keep Q_B BF16 result before FP32 rotary and BF16_RINT result','Both final output partners ready in same tile or priced cross-tile handoff','Correct local row/head/position/C-S generation and all live query observers','Attention full512 score reduction/precision remains;pair-local RoPE does not make full attention pair-local','Price shared Cube/Vector/dequant/quantizer/layout/queue and captured-buffer ABI'],shared_resource_model=resource,new_device_work=False,actual_runtime_speedup=None,overall_complete=False))
    # Supersede exactly the three v74 clauses and one group which imported the
    # legacy head RMS into V4.1. Retain their full previous text in a correction.
    clauses=deepcopy(read(O/'candidate-relation-clauses-v74.json'));changes=[]
    replacements={
        'REL74-03':('ACTIVE_QR1280_NORM;LEGACY_HEAD512_OUTSIDE_CASE','Current V4.1 QR1280 normalization precedes QB;its output and quantizer remain distinct from rotary. Legacy CP head512 RMS is not in target/draft V4.1 query forward and contributes no current-case work.','Target/draft V4.1 operator entry and both ordinary/multistream preprocess branches;legacy program mathematics is separate.'),
        'REL74-04':('JOINT_QB_INDEXQ_WITHOUT_LEGACY_HEAD_RMS','Current QB/IndexQ shared quant/input tiling preserves distinct outputs,scales and the QB BF16→tail64 RoPE chain. Do not impose a post-QB head512 reduction from unused legacy query forward.','V4.1 target/draft source and exact direct-call stack bindings;captured graph role evidence retains its own grade.'),
        'REL74-05':('QUERY_ROPE_CONSUMER_FUSION_WITH_CURRENT_NUMERICS','QB/tail64 RoPE into attention retains source BF16 query rounding and full512 score-dot dependencies. Current V4.1 has no head512 RMS barrier;both kernels share finite compute/Vector resources.','Current V4.1 path;no query-transform deletion or claimed whole512 intermediate traffic removal from a tail-only fusion.'),
    }
    for row in clauses['clauses']:
        if row['id']in replacements:
            old=deepcopy(row);kind,claim,domain=replacements[row['id']]
            row.update(relation=kind,operator_claim=claim,applicable_domain=domain)
            row['evidence']+=['V41-query-operator-path-contract-v75.json','V41-QB-native-source-binding-v75.json','V41-QB-RoPE-work-and-bound-models-v75.json']
            row['supersedes_current_case_head_RMS_claim_v74']=True
            changes.append(dict(id=row['id'],previous=old,current=deepcopy(row)))
    assert len(changes)==3
    group_changes=[]
    for group in clauses['higher_order_groups']:
        if group['id']=='HG74-Q-ATTENTION':
            old=deepcopy(group);group['contract']='Current V4.1 QR1280 norm/quant→QB BF16→tail64 RoPE→attention BF16→inverse tail RoPE. No legacy post-QB head512 RMS in this entry. Preserve full512 attention score reductions and choose one materialization/pack graph;shared typed resource capacities still constrain fusion.';group_changes.append(dict(previous=old,current=deepcopy(group)))
    assert len(group_changes)==1
    write(O/'candidate-relation-clauses-v75.json',clauses)
    graph=deepcopy(read(O/'candidate-relation-coverage-v74.json'));graph.update(parent_relation_index='candidate-relation-coverage-v74.json',scope='43selected pair clauses/8groups/all35nodes as before;3clauses/1group corrected for actual V4.1 entry.552centralpairreconciliations and all whole-pair/full-model opportunity gates remain open.',overall_complete=False);write(O/'candidate-relation-coverage-v75.json',graph)
    write(O/'V41-head-RMS-scope-correction-v75.json',dict(previous_authority='QB-head-RMS-RoPE-applicability-v74.json',error='Legacy query-forward constraints were incorrectly applied to target/draft V4.1 entry despite separate forward/preprocess dispatch.',source_and_native_correction='Exact V4.1 entry/preprocess clauses plus native owner→captured CPU callchain binding;absence of a standalone ComputeInfo was never by itself proof.',relations=changes,groups=group_changes,legacy_mathematical_witnesses_retained_as_legacy_only=True,actual_V41_QB_head_RMS_added_work=0,loaded_ELF_attested=False,overall_complete=False))
    write(O/'V41-query-local-analysis-history-v75.json',dict(resolved_events=[dict(kind='analytical-scope-error',status='RESOLVED',description='v74 traced legacy CP._forward rather than actual V4.1 query entry. Corrected through target/draft forward factory/MRO and captured source-frame binding.'),dict(kind='SQL-syntax-error',status='RESOLVED',description='One exploratory query used limit5 without keyword whitespace;fixed before evidence extraction;no analytical artifact written by failure.'),dict(kind='SQL-plan-cancelled',status='RESOLVED',description='A correlated EXISTS scan was explicitly cancelled after identifying this local saved-DB reader as PID39814/session1790;SIGINT deferred by sqlite then SIGTERM yielded exit143. No DB writes. Replaced with one scan per grouped ID set and a20second SQLite progress bound per database;not restarted because of observation timeout.')],no_server_process_or_device_actions=True,overall_complete=False))
    print(dict(native_QB=4367,direct=1167,bound_direct=len(records),unbound_direct=len(unbound),graph_role_only=3200,direct_contexts=dict(contexts),frames=dict(frame_counts),active_V41_post_QB_headRMS=False,corrected_clauses=3,corrected_groups=1,tile_cases=len(tile_cases),remote_reads=0,overall_complete=False))


if __name__=='__main__':main()
