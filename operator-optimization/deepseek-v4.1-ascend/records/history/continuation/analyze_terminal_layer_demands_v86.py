"""Profile-led terminal-layer demand cut; no model import, remote read or implementation."""
from collections import defaultdict,Counter
from pathlib import Path
from copy import deepcopy
import ast,hashlib,zipfile
from finalize_origin_coverage_v9 import N,O,R,W,read,write
from mirror_saved_profile_core_v53 import CACHE

def receipt(path):
    data=path.read_bytes();return dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())

def main():
    assert not(O/'terminal-layer-native-frontier-v86.json').exists()
    assert read(O/'REVIEW-CHECKPOINT-v85.json')['status']=='PROGRESS'
    prior_names=['M15-normalized-content-consumer-and-precision-contract-v31.json',
      'heads-sampling-fusion-variants-v65.json','heads-sampling-fusion-costs-v65.json',
      'DSpark-tail-prior-evidence-reconciliation-v85.json','DSpark-tail-native-boundaries-v85.json',
      'M16-current-head-ownership-and-step-census-v32.json']
    write(O/'terminal-layer-prior-evidence-v86.json',dict(input_receipts=[receipt(O/n)for n in prior_names],
      already_established=['v11 direct DSpark raw-snapshot drafter-content answer',
        'v31 prepared head logits_indices precedes target forward; normalized prompt reader and state handoff; final-Post conditional candidate',
        'v31/v65 ordered all-reader row union and sparse publication; v65 equal-count8max(U_p) padding rule',
        'v85 refined final-Post private interface and shared memory costs'],
      genuinely_new_scope=['Exact last-layer39 typed35-ComputeInfo/task schema over80graphs, joined by existing execution identities',
        'Backward output-demand cut across final attention Q/output and FFN while preserving complete required KV/state writers',
        'Fixed-shape balanced/concentrated owner-quota envelopes and stage-specific conventional work/weight costs'],
      source_question_closed_before_this_turn=True,new_full_reader_or_native_fusion_closure=False,overall_complete=False))
    oldsrc=read(O/'DSpark-tail-source-index-v85.json');source=[]
    wanted={
      'vllm-ascend/vllm_ascend/models/deepseek_v41/model.py':[(262,328),(844,920),(1554,1565),(1592,1608)],
      'vllm-ascend/vllm_ascend/worker/model_runner_v1.py':[(1635,1694),(1771,1798),(2355,2364),(2596,2598),
        (2610,2667),(2785,2801),(2835,2848),(2881,2888),(2052,2091),(3130,3134)],
      'vllm/vllm/v1/worker/gpu_model_runner.py':[(5643,5667),(5685,5712)]}
    with zipfile.ZipFile(oldsrc['original_archive']['path'])as archive:
        for item in oldsrc['source_files']:
            if item['archive_member']not in wanted:continue
            path=Path(item['path']);data=path.read_bytes();assert data==archive.read(item['archive_member'])
            assert hashlib.sha256(data).hexdigest()==item['sha256']
            lines=data.decode().splitlines()
            source.append(dict(**receipt(path),archive_member=item['archive_member'],
              clauses=[dict(first=a,last=b,source=lines[a-1:b])for a,b in wanted[item['archive_member']]]))
    assert len(source)==3
    runner=next(s for s in source if s['archive_member'].endswith('worker/model_runner_v1.py'))
    tree=ast.parse(Path(runner['path']).read_text());uses={}
    for cls in tree.body:
        if not isinstance(cls,ast.ClassDef):continue
        for fn in cls.body:
            if isinstance(fn,ast.FunctionDef)and fn.name in ['execute_model','sample_tokens','_bookkeeping_sync','propose_draft_token_ids','_model_forward']:
                uses[fn.name]=sorted({n.lineno for n in ast.walk(fn)if isinstance(n,ast.Name)and isinstance(n.ctx,ast.Load)and n.id=='hidden_states'})
    assert uses['_bookkeeping_sync']==[3132]
    write(O/'terminal-layer-operator-source-v86.json',dict(source_files=source,original_archive=oldsrc['original_archive'],
      normalized_hidden_identifier_load_inventory=uses,
      inventory_scope='Only named values in selected pinned model/runner clauses; not installed-global observer closure or framework research.',
      head_selector='Prepared before target forward. Non-spec end-of-segment indices; spec suffix indices from scheduler counts. Optional lmhead_tp pad adds index0 duplicates. Same historical native tensor/values not attested.',
      normalized_content_readers='Head I plus optional prompt per-request intervals J; current DSpark aux branch selects3pre-layer checkpoints instead of normalized target hidden. State handoff is a lifetime escape, not by itself a numerical reader.',
      raw_and_normalized_all_readers_closed=False,new_target_source_files_obtained=0,new_remote_source_read=False,overall_complete=False))
    assignment_path=CACHE/'analysis/native-PMU-complete-v18/resource-functional-attribution-and-uncertainty-v20-v3.json'
    assignments=defaultdict(dict)
    for a in read(assignment_path)['execution_assignments']:assignments[a[0]][a[1]]=a
    frontiers=read(O/'DSpark-tail-native-boundaries-v85.json')['graph_boundaries'];bydb=defaultdict(list)
    for row in frontiers:bydb[row['database_index']].append(row)
    results=[];schemas=Counter();input_receipts=[receipt(assignment_path)]
    for db,rows in bydb.items():
        path=CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json';native=read(path);input_receipts.append(receipt(path))
        tasks={r['task_record_rowid']:r for r in native['TASK_records']};defs={r['globalTaskId']:r for r in native['COMPUTE_definitions']}
        for frontier in rows:
            selected=[a for a in assignments[db].values()if a[2]=='GRAPH'and a[8]==frontier['graph_ordinal']and a[9]==39]
            stages=[]
            for a in selected:
                task=tasks[a[1]];d=defs[task['globalTaskId']]
                stages.append(dict(TASK_rowid=a[1],globalTaskId=task['globalTaskId'],operator=d['opType_text'],
                  input_shapes=d['inputShapes_text'],output_shapes=d['outputShapes_text'],input_types=d['inputDataTypes_text'],
                  output_types=d['outputDataTypes_text'],historical_role=a[5],modules=a[3],has_PMU=a[11],startNs=task['startNs'],endNs=task['endNs']))
            stages.sort(key=lambda s:s['startNs']);ops=Counter(s['operator']for s in stages)
            assert len(stages)==35 and ops['HcPre']==ops['HcPost']==2
            assert ops['SparseFlashMla']==ops['ScatterNdUpdateSk']==ops['DispatchFFNCombine']==1
            assert stages[-1]['TASK_rowid']==frontier['last_FFN_HcPost_TASK']
            assert next(s for s in stages if s['operator']=='ForeachAddList')['TASK_rowid']==frontier['last_shared_routed_merge_TASK']
            schemas[tuple(sorted(ops.items()))]+=1
            results.append(dict(database_index=db,graph_ordinal=frontier['graph_ordinal'],root_bound=frontier['root_bound'],layer=39,
              local_M=3,global_N=24,stages=stages,graph_task_schema=dict(ops),
              actual_selected_row_values_or_owner_plan_attested=False,all_state_writers_or_external_observers_closed=False))
    assert len(results)==80 and len(schemas)==1
    roots=read(O/'M16-current-head-ownership-and-step-census-v32.json')['root_and_current_step_audit']['target_root_regimes']
    assert sum(r['roots']for r in roots)==88
    write(O/'terminal-layer-native-frontier-v86.json',dict(graphs=results,graphs_count=80,root_bound=64,phase_unbound=16,
      reused_ComputeInfo_executions=2800,already_reviewed_regimes_not_promoted=True,
      identical_operator_type_multiplicity_schema_count=1,operator_count_per_graph=35,schema=dict(next(iter(schemas))),
      per_case_shapes_and_dtypes_kept_without_cross_case_equality_claim=True,
      target_head_row_regimes=roots,input_receipts=input_receipts,
      direct_scope='24DIRECT target tails have typed N6400/6528/6672 and existing88head root joins overall; no new last-layer DIRECT edge join or actual index value/owner claim.',
      no_new_native_or_critical_path_evidence=True,overall_complete=False))
    stages=[
      ('last_layer_input_HC_and_pre_layer_aux','FULL_REQUIRED_ROWS','All final-layer KV inputs and all configured earlier/pre-layer auxiliary readers remain.'),
      ('attention_HcPre_and_input_RMS','FULL_KV_DEMAND_PLUS_SELECTED_Q','Q demand alone cannot prune common HC/RMS input needed by full required KV projection.'),
      ('SWA_KV_projection_norm_RoPE_cache_write','FULL_REQUIRED_STATE_ROWS','Keep required scheduled-token writes/slots/positions/visibility and native pad protocol; selected output rows are not the KV writer set.'),
      ('compressor_index_cache_branches_if_applicable','OWN_STATE_AND_QUERY_DEMANDS','No deletion from absence in this35node schema; alternate layers/branches retain pool/cache/index state contracts.'),
      ('Q_A_QR_norm_quant_Q_B_query_RoPE','SELECTED_QUERY_ROWS_CONDITIONAL','Exact SP/CP owner and original positions/sequence/causal metadata, same per-row quant/rounding. New empty/ragged ABI required.'),
      ('SparseFlashMla_query_output','SELECTED_QUERY_ROWS_CONDITIONAL','Keep all required historical/current keys/values and causal windows; compact query ordinal cannot replace original position/sequence identity.'),
      ('inverse_RoPE_A2A_WO_A_WO_B_RS','SELECTED_GLOBAL_OUTPUT_ROWS_CONDITIONAL','Retain all head-group partials for each selected row, original BF16/RS order and group-uniform ragged/capture mapping.'),
      ('attention_Post_and_FFN_HcPre_RMSNormCast','SELECTED_ROWS_CONDITIONAL','Both HC and FFN branch coefficients/residuals and BF16 barriers retain their individual row generation.'),
      ('router_shared_routed_experts_combine','SELECTED_ROWS_CONDITIONAL','Keep input-ID/vision/hash predicates and original expert maps/weight epochs. EPLB heat/load counters or other batch/state observers may require retained work; MC2 physical receiver padding is not linear in selected count.'),
      ('final_Post_collapse_Norm','UNION_OF_ALL_FINAL_READERS_CONDITIONAL','Need no additional raw/normalized observers and exact aliases/lifetime. Preserve Post/C/Norm rounds, prompt J and inverse ordered maps.'),
      ('LM_head_and_sampling','ORIGINAL_ORDERED_K_ROWS','Head selection already precedes GEMM. Do not claim N→U pruning also removes K-row head/probability work. Preserve duplicates/order and all prompt/processor/RNG contracts.'),
      ('previous_layer_output','FULL_LAST_LAYER_KV_INPUT_DEMAND','Cannot propagate output-only row deletion through the last-layer KV/state writer cut. Fusion may remove a materialization while retaining full necessary values/work.')]
    write(O/'terminal-layer-demand-cut-v86.json',dict(stages=[dict(stage=s,demand=d,conditions=c)for s,d,c in stages],
      definition='I is ordered head selector incl duplicates/pad; J all active additional final-normalized content rows, including prompt. U=distinct valid rows in union(I,J). Rawsnapshot and intermediate layer observers impose independent demands.',
      applicability='Conditional operator graph only. Exact active branch/all observers/numeric+mapping+capture+state side effects still unproved. Source ordering and separability do not establish native new-shape parity.',
      source_advance_not_new_reader_closure=True,previous_layer_pruning_or_future_cache_write_deletion_authorized=False,
      no_new_maincandidate=True,implementation=False,native_test=False,overall_complete=False))
    print(dict(profile=[80,64,16,2800],operators_per_graph=35,source_files=3,prior_selector_answer_version=31,new_scope='terminal output-demand/state-writer cut',all_readers_closed=False))

if __name__=='__main__':main()
