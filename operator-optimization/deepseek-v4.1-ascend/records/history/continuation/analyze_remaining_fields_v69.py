"""Saved-profile operator contracts: no framework audit or device work."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import hashlib,math,re
from finalize_origin_coverage_v9 import N,O,W,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_state_engram_v62 import fullkey,dimensions
from analyze_projection_ceiling_v61 import stat

DT={'INT32':4,'INT64':8,'UINT64':8,'BOOL':1,'FLOAT':4,'DT_BF16':2}
S=W/'frozen-source-snapshot'
def line(role):return int(re.search(r'\((\d+)\)',role).group(1))
def family(role):
    l=line(role)
    if 'ops/rope_dsv4.py' in role:return 'ROPE_BANK_LOOKUP'
    if 'ascend_forward_context.py' in role:return 'MC2_ACTIVE_MASK'
    if 'sequence_parallel.py' in role and l==149:return 'SP_NONE_MASK'
    if 'ops/triton/compute_slot_mapping.py' in role:return 'GROUP_SLOT'
    if 'dspark_swa_indices.py' in role:return 'SWA_PRODUCER'
    if 'spec_decode/dspark_proposer.py' in role:return 'DRAFT_FIRST_PASS' if l<350 else 'DRAFT_DUMMY_FIELDS'
    if 'spec_decode/llm_base_proposer.py' in role and 2231<=l<=2247:return 'NEXT_ID'
    if 'spec_decode/llm_base_proposer.py' in role and l==2414:return 'PADDED_INPUTS'
    if 'worker/model_runner_v1.py' in role and l in [3583,3584]:return 'WORKER_PAD_TAIL'
    if 'worker/model_runner_v1.py' in role and l in [4137,4154,4241]:return 'WORKER_DUMMY_FIELDS'
    return None

def main():
    assert not (O/'remaining-field-regime-models-v69.json').exists()
    cv={r['regime_id']:r for r in read(O/'operator-analysis-coverage-ledger-v68.json')['regimes']}
    rs=[r for r in read(O/'operator-shape-ceiling-ledger-v55.json')['observed_regimes'] if r['module_candidates']==['M21'] and family(r['role']) and cv[r['regime_id']]['state'].startswith('REGIME_')]
    assert (len(rs),sum(r['executions']for r in rs))==(161,6680)
    lookup={fullkey(r['module_candidates'],r['operator'],r['role'],r['origin'],r['context'],r['signature']):r for r in rs}
    selected=defaultdict(dict);opaque=defaultdict(dict)
    for a in read(CACHE/'analysis/native-PMU-complete-v18/resource-functional-attribution-and-uncertainty-v20-v3.json')['execution_assignments']:
        if a[3]==['M21'] and family(a[5]):selected[a[0]][a[1]]=a
        if a[3]==['M21'] and any(x in a[5] for x in ['(1176)','(1254)']):opaque[a[0]][a[1]]=a
    ids=defaultdict(list);times=defaultdict(list);missing=Counter();types=defaultdict(Counter)
    chains=defaultdict(list);opaque_rows=[]
    for di in range(32):
        e=read(CACHE/f'analysis/native-execution-identity-v14/db-{di:02d}.json');ts={t['task_record_rowid']:t for t in e['TASK_records']};ds={d['globalTaskId']:d for d in e['COMPUTE_definitions']}
        for tid,a in selected[di].items():
            t=ts[tid];d=ds[t['globalTaskId']];r=lookup[fullkey(a[3],d['opType_text'],a[5],a[2],a[6],d)];rid=r['regime_id']
            ids[rid].append([di,tid]);times[rid].append(t['endNs']-t['startNs']);missing[rid]+=not a[11];types[rid][t['task_type_name']]+=1
        for tid,a in opaque[di].items():
            t=ts[tid];d=ds[t['globalTaskId']]
            opaque_rows.append(dict(database_index=di,TASK_rowid=tid,operator=d['opType_text'],task_type=t['task_type_name'],taskType=t['taskType'],duration_ns=t['endNs']-t['startNs'],has_PMU=a[11],blockNum=d['blockNum'],output_shape=d['outputShapes_text'],output_dtype=d['outputDataTypes_text']))
        def op(t):return ds[t['globalTaskId']]['opType_text']
        def ln(t):return line(selected[di][t['task_record_rowid']][5])
        def fam(t):return family(selected[di][t['task_record_rowid']][5])
        def comp(t):return dict(TASK_rowid=t['task_record_rowid'],operator=op(t),line=ln(t),streamId=t['streamId'],startNs=t['startNs'],endNs=t['endNs'])
        def group(pool,n,counts,name,shapeop):
            pool=sorted(pool,key=lambda t:(t['startNs'],t['task_record_rowid']));assert len(pool)%n==0,(di,name,len(pool))
            for i in range(0,len(pool),n):
                gg=pool[i:i+n];assert Counter(op(t)for t in gg)==counts,(di,name,i,Counter(op(t)for t in gg))
                assert len({t['streamId']for t in gg})==1
                a=dimensions(ds[next(t for t in gg if op(t)==shapeop)['globalTaskId']]['inputShapes_text'])
                chains[name].append(dict(database_index=di,shape=a[0],stages=[comp(t)for t in gg],typed_source_order_only=True,actual_alias_value_identity_proved=False))
        chosen=[ts[k]for k in selected[di]]
        group([t for t in chosen if fam(t)=='NEXT_ID'],13,{'NotEqual':2,'Cast':4,'Less':1,'LogicalAnd':1,'ReduceSum':1,'Sub':1,'ClipByValueV2':1,'GatherElementsV2':1,'SelectV2':1},'next_ID_13TASK','ReduceSum')
        for l in [173,174]:group([t for t in chosen if fam(t)=='ROPE_BANK_LOOKUP'and ln(t)==l],3,{'BroadcastTo':1,'Cast':1,'GatherElementsV2':1},f'rope_{l}_3TASK','BroadcastTo')
        for l in [185,186]:group([t for t in chosen if fam(t)=='ROPE_BANK_LOOKUP'and ln(t)==l],2,{'IndexCheck':1,'Index':1},f'rope_{l}_2TASK','Index')
    assert sum(map(len,ids.values()))==6680 and sum(missing.values())==0
    assert len(chains['next_ID_13TASK'])==88
    assert [len(chains[f'rope_{l}_{n}TASK'])for l,n in [(173,3),(174,3),(185,2),(186,2)]]==[288,288,96,96]
    assert len(opaque_rows)==529
    models=[]
    contracts={
      'ROPE_BANK_LOOKUP':'Runtime/spec out banks FLOAT64wide; cached gather native INT64-expanded→INT32/error semantics vs uncached advanced-index negative-wrap. Joint rowlookup/CSE requires config/group/position/cast/domain/strides/lastreader-generation; consumer inline requires all reader/new ABI. No dense table scan.',
      'MC2_ACTIVE_MASK':'Source field mc2_mask[:A]=True,[A:P]=False. Direct BOOL stores avoid native scalar→INT32 broadcast→BOOL temporaries. All P must match; dirty interval maintains same-bank old threshold under writer/reset fences. Materialized mask cannot disappear unless every consumer accepts threshold or constant-mask ABI.',
      'SP_NONE_MASK':'Only is_padding=None creates false prefix. Fuse false-prefix+SP-pad true-tail into producer under existing SP semantics; supplied masks may contain true active rows and remain distinct. No identity claim for ambiguous80 BOOL stores.',
      'GROUP_SLOT':'Already13group fusion and ordinary/adaptive tiles; UINT64[13] are pointers. Actual dereferenced INT32 outputs [0:C). Native int32→float32→int32 block lookup rounding retained; dcp1, circular negative PAD, power2/window bounds preserved. Dirty suffix/full-bank invariant is conditional, C not inferred from address descriptor.',
      'SWA_PRODUCER':'Already one launch for all indices+raw lengths; each query repeats request-local noncausal window. Preserve raw V (not clipped count), -1 tail, request map/capacity. Per-request storage requires new SMLA/CP consumer ABI; direct same-layout request-tiling changes loads/arithmetic but still writes every N*C index.',
      'NEXT_ID':'Literal (x!=-1)&(x<V) includes x<-1. Count then select x[max(count-1,0)], NOT arbitrary last-valid-index. Preserve discard-to-1, all-discard backup, INT64 count/INT32 ID, all async readers. FP32 reduction exact for captured K1/6; no unbounded-K integer replacement claim.',
      'PADDED_INPUTS':'Already rejection+selector fused. d=cu[i]-cu[i-1],r=where(d>0,d+1-valid,0),sample=QSL[i+1]-1-r; no clamp. Can fuse with next-ID counts only same request/bank/generation, valid count all readers, cu/QSL ready; no token compaction implied.',
      'DRAFT_FIRST_PASS':'Existing per-group copy-expand writes context positions cast INT64→INT32 and slot INT32, query positions/IDs/slots INT32 and sample indices INT32; source descriptor8192 is capacity, no full copy. Position uses accepted context position; slot uses authoritative length-reject+DCP linear mapping. Shared field writes repeated across groups can be specialized only group-independent values and all-reader/fence proof.',
      'DRAFT_DUMMY_FIELDS':'Full query/context PAD and token-index0 initialization must survive real→dummy transition. Coalesce disjoint typed buffers; skip repeats only whole affected bank/epoch and no intervening writes. Dummy clearing not subsumed by active-prefix dirty proof alone.',
      'WORKER_PAD_TAIL':'Only observed Fill output regions are writes: INT32 slot[P-N] and block-table padded request rows. Same value different bank/window is not redundant. Producer tail elision needs containment, same epoch, no intermediate writer; table zero separately. Unknown aliased addresses not inferred.',
      'WORKER_DUMMY_FIELDS':'Slot full-bank -1, compressed dummy positions127, dummy logits indices0 have different fields/consumers. Whole writes measured by output shape; zero-output indices can be persistent read-only only no mutation/lifetime and downstream indexing ABI. Preserve capture paths and cache safety.'}
    for r in rs:
        rid=r['regime_id'];assert len(ids[rid])==r['executions']and missing[rid]==r['missing_PMU']
        d=r['signature'];op=r['operator'];f=family(r['role']);a=dimensions(d['inputShapes_text']);z=dimensions(d['outputShapes_text']);it=d['inputDataTypes_text'].split(';');ot=d['outputDataTypes_text'].split(';')
        ib=[None if s is None or t not in DT else math.prod(s)*DT[t]for s,t in zip(a or [],it)];ob=[None if s is None or t not in DT else math.prod(s)*DT[t]for s,t in zip(z or [],ot)];on=math.prod(z[0])if z and z[0] is not None else None
        io=None;work={}
        if f=='GROUP_SLOT':
            B=a[0][0]-1;nr=a[1][0];G=a[2][0];assert G==13 and a[3]==[13]
            work=dict(N=nr,B=B,G=G,descriptor_pointer_array_bytes=104,required_output_stores_symbolic='4*G*C',active_output_bytes=4*G*nr,group_repeated_position_load_reference_bytes=8*G*nr,dirty_suffix_new_output_bytes='4*G*(N+max(previous_N-N,0))',capacity_C_bound=False,block_table_window_reads_unbound=True)
        elif f=='SWA_PRODUCER':
            B=a[0][0]-1;nr,_,C=z[0];assert a[1]==[B] and C==256 and nr==5*B
            io=4*nr*(B+C+4)
            work=dict(N=nr,B=B,C=C,logical_load_reference_bytes=4*nr*(B+3),required_materialized_output_bytes=4*nr*(C+1),unique_input_descriptor_bytes=8*B+4,request_compact_output_bytes=4*B*(C+1),new_consumer_ABI_required_for_compact=True,raw_length_must_remain_unclipped=True)
        elif op=='copy_and_expand_dflash_and_dspark_inputs_kernel':
            B=a[0][0];T=a[1][0];work=dict(B=B,context_position_descriptor_T=T,context_slot_descriptor_capacity=a[2][0],output_8192_is_capacity=True,T_active_symbolic='cad.query_start_loc_cpu[B]',context_copy_value_bytes='20*T_active',query_rows='B*Q',query_publication_bytes='12*B*Q+4*B*D',query_source_value_reference_bytes='(28+4*HAS_REJECTED)*B*Q',hidden_copy_separate=True,source_hidden_copy_bytes='2*T_active*5120*2')
        elif op=='prepare_inputs_padded_kernel':
            B=a[0][0];assert it==['INT32','INT64','INT32'];io=28*B-4;work=dict(B=B,integer_row_formulas=3,required_output_bytes=8*B,cumulative_previous_reference_reads=B-1,valid_count_INT64=True)
        elif op=='GatherElementsV2':
            io=on*(DT[it[0]]+DT[it[1]]+DT[ot[0]]);work=dict(selected_elements=on,index_elements=math.prod(a[1]),table_descriptor_bytes=ib[0],whole_table_scan=False)
        elif op=='Index':
            assert f=='ROPE_BANK_LOOKUP';nr=z[0][0];io=8*nr+2*ob[0];work=dict(N=nr,selected_table_elements=on,table_descriptor_bytes=ib[0],known_value_bytes_exclude_small_index_metadata=True,negative_wrap_and_bounds_check=True)
        elif op=='IndexCheck':
            io=ib[-1];work=dict(index_elements=math.prod(a[-1]),no_tensor_value_output=True,keep_validation_or_prove_legal_domain=True)
        elif op in ['Fill','ZerosLike']:
            io=sum(x for x in ob if x is not None);work=dict(value_stores=on,input_value_reads=0,output_shape_is_write_extent=True)
        elif op=='TensorMove':
            io=ib[0]+ob[0];work=dict(copied_elements=on,source_copy_extent_not_capacity=True,required_hidden_reader_banks=True)
        elif op=='BroadcastTo':
            io=ib[0]+ob[0];work=dict(output_elements=on,exclude_shape_vector_bytes=True)
        elif op=='ReduceSum':
            io=ib[0]+ob[0];work=dict(reduction_adds=math.prod(a[0])-on,native_reduction_dtype=ot[0],capture_counts_exact_FP32=True)
        else:
            assert op in ['Cast','NotEqual','Less','LogicalAnd','Sub','ClipByValueV2','SelectV2','Add','Mul'],op
            io=sum(x for x in ib+ob if x is not None);work=dict(elements=on,known_values_exclude_unbound_empty_scalar_operands=True)
        models.append(dict(regime_id=rid,module_candidates=['M21'],operator=op,role=r['role'],origin=r['origin'],context=r['context'],signature=d,family=f,executions=r['executions'],native_TASK_keys=ids[rid],native_TASK_ns=stat(times[rid]),native_TASK_types=dict(types[rid]),missing_PMU=missing[rid],known_logical_value_bytes_excluding_control=io,work_model=work,consumer_contract=contracts[f],consumer_fusion_review_complete_in_this_turn=True,physical_numeric_ceiling_proved=False))
    source_names=['spec_decode/llm_base_proposer.py','spec_decode/dspark_proposer.py','spec_decode/dflash_proposer.py','ops/rope_dsv4.py','ops/triton/spec_decode/utils.py','ops/triton/spec_decode/dspark_swa_indices.py','ascend_forward_context.py','worker/model_runner_v1.py','models/common/ops/sequence_parallel.py']
    sources=[S/'vllm-ascend/vllm_ascend'/x for x in source_names]+[W/'sampler-source-supplement/frozen-source-snapshot/vllm-ascend/vllm_ascend/ops/triton/compute_slot_mapping.py']
    source_files=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in sources]
    old=read(O/'M21-slot-rejection-state-and-packing-contract-v38.json')['source_pins']
    for p in source_files:
        remote='/vllm-workspace/vllm-ascend/'+p['path'].split('/vllm-ascend/')[-1]
        if remote in old:assert p['sha256']==old[remote]
    write(O/'remaining-field-regime-models-v69.json',dict(generated_utc=datetime.now(timezone.utc).isoformat(),regimes=models,regime_count=len(models),captured_execution_count=6680,regimes_by_family=dict(Counter(r['family']for r in models)),executions_by_family={f:sum(r['executions']for r in models if r['family']==f)for f in sorted(contracts)},scope='161 actual operator regimes; typed selected work/materialized IO/source-specific conditional choices. Opaque planners/accessor/other fill families remain pending.',overall_complete=False))
    write(O/'remaining-field-native-chains-v69.json',dict(chains=chains,scope='Exact per-database/source-order typed TASK groups, no fabricated tensor address/value/criticalpath proof.',overall_complete=False))
    write(O/'remaining-field-operator-contract-v69.json',dict(source_files=source_files,source_clauses_reused_locally=True,new_remote_read=False,contracts=contracts,source_copy_expand_field_dtypes=dict(context_positions='INT32 from INT64',query_positions='INT32',input_ids='INT32',context_slots='INT32',query_slots='INT32',token_indices='INT32',seed='INT64'),slot_bank_capacity_C='Not present in UINT64[13] trace descriptor; retain symbolic C, no8192 assumption.',no_model_Torch_NPU_capture_health_process_implementation=True,whole_modules_closed=0,overall_complete=False))
    write(O/'opaque-plan-TASK-type-audit-v69.json',dict(native_tasks=opaque_rows,task_type_counts=dict(Counter(r['task_type']for r in opaque_rows)),operator_counts=dict(Counter(r['operator']for r in opaque_rows)),missing_PMU=sum(not r['has_PMU']for r in opaque_rows),scope='TASK-type evidence only; opaque planner internal work/purity/address dependence NOT established by blockNum0/output4096B, remains pending. Global849missingPMU preserved.',overall_complete=False))
    print(dict(regimes=len(models),executions=6680,byfamily={f:[sum(r['family']==f for r in models),sum(r['executions']for r in models if r['family']==f)]for f in sorted(contracts)},chains={k:len(v)for k,v in chains.items()},opaque_task_types=dict(Counter(r['task_type']for r in opaque_rows)),sources=len(sources)))

if __name__=='__main__':main()
