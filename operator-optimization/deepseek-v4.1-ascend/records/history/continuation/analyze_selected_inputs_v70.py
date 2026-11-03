"""Bind the last60 nonplan M21 regimes to selected rows and concrete operator contracts."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import hashlib,math,re
from finalize_origin_coverage_v9 import N,O,W,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_state_engram_v62 import fullkey,dimensions
from analyze_projection_ceiling_v61 import stat
from analyze_remaining_fields_v69 import DT
S=W/'frozen-source-snapshot'
def line(role):return int(re.search(r'\((\d+)\)',role).group(1))
def family(role):
    l=line(role)
    if 'spec_decode/llm_base_proposer.py'in role:
        if l==1061:return 'INITIAL_SELECTOR_TEMP'
        if l in [1212,1216,1220,1343]:return 'DRAFT_FIXED_BANK_TAILS'
        if l==1246:return 'NEXT_STEP_POSITIONS'
        if l==1534:return 'SAMPLE_HIDDEN_ROWS'
        if l==1584:return 'MARKOV_SEED_COLUMN'
    if 'worker/model_runner_v1.py'in role:
        if l in [1836,1837]:return 'COMPOSED_DRAFT_ID_GATHER'
        if l==2069:return 'AUX_PREFIX_CONCAT'
        if l==2086:return 'TARGET_INPUT_ID_GATHER'
        if l==2089:return 'AUX_GATHER_CONCAT'
    if 'v/v1/worker/gpu_model_runner.py'in role and l==1031:return 'TARGET_POSITION_GATHER'
    return None

def main():
    assert not(O/'selected-input-regime-models-v70.json').exists()
    cv={r['regime_id']:r for r in read(O/'operator-analysis-coverage-ledger-v69.json')['regimes']}
    rs=[r for r in read(O/'operator-shape-ceiling-ledger-v55.json')['observed_regimes']if r['module_candidates']==['M21']and family(r['role'])and cv[r['regime_id']]['state'].startswith('REGIME_')]
    assert(len(rs),sum(r['executions']for r in rs))==(60,2024)
    lk={fullkey(r['module_candidates'],r['operator'],r['role'],r['origin'],r['context'],r['signature']):r for r in rs};selected=defaultdict(dict)
    for a in read(CACHE/'analysis/native-PMU-complete-v18/resource-functional-attribution-and-uncertainty-v20-v3.json')['execution_assignments']:
        if a[3]==['M21']and family(a[5]):selected[a[0]][a[1]]=a
    ids=defaultdict(list);times=defaultdict(list);missing=Counter();types=defaultdict(Counter);chains=defaultdict(list)
    for di in range(32):
        e=read(CACHE/f'analysis/native-execution-identity-v14/db-{di:02d}.json');ts={t['task_record_rowid']:t for t in e['TASK_records']};ds={d['globalTaskId']:d for d in e['COMPUTE_definitions']}
        for tid,a in selected[di].items():
            t=ts[tid];d=ds[t['globalTaskId']];r=lk[fullkey(a[3],d['opType_text'],a[5],a[2],a[6],d)];rid=r['regime_id'];ids[rid].append([di,tid]);times[rid].append(t['endNs']-t['startNs']);missing[rid]+=not a[11];types[rid][t['task_type_name']]+=1
        def op(t):return ds[t['globalTaskId']]['opType_text']
        def fam(t):return family(selected[di][t['task_record_rowid']][5])
        def comp(t):return dict(TASK_rowid=t['task_record_rowid'],operator=op(t),line=line(selected[di][t['task_record_rowid']][5]),streamId=t['streamId'],startNs=t['startNs'],endNs=t['endNs'])
        def group(f,num,counts):
            pool=sorted([ts[k]for k in selected[di]if fam(ts[k])==f],key=lambda t:(t['startNs'],t['task_record_rowid']));assert len(pool)%num==0,(di,f,len(pool))
            for j in range(0,len(pool),num):
                gg=pool[j:j+num];assert Counter(op(t)for t in gg)==counts,(di,f,j,Counter(op(t)for t in gg))
                assert len({t['streamId']for t in gg})==1
                shapes=[dict(input=dimensions(ds[t['globalTaskId']]['inputShapes_text']),output=dimensions(ds[t['globalTaskId']]['outputShapes_text']))for t in gg]
                x=dict(database_index=di,stages=[comp(t)for t in gg],shapes=shapes,typed_source_order_only=True,actual_tensor_generation_identity_proved=False)
                chains[f].append(x)
        group('COMPOSED_DRAFT_ID_GATHER',5,{'IndexCheck':2,'Index':2,'Add':1})
        group('AUX_GATHER_CONCAT',7,{'IndexCheck':3,'Index':3,'ConcatD':1})
        group('DRAFT_FIXED_BANK_TAILS',4,{'Fill':4})
        for f in ['TARGET_INPUT_ID_GATHER','TARGET_POSITION_GATHER','NEXT_STEP_POSITIONS','SAMPLE_HIDDEN_ROWS']:group(f,2,{'IndexCheck':1,'Index':1})
        for f,opname in [('INITIAL_SELECTOR_TEMP','Sub'),('AUX_PREFIX_CONCAT','ConcatD'),('MARKOV_SEED_COLUMN','ViewCopy')]:group(f,1,{opname:1})
    assert {k:len(v)for k,v in chains.items()}=={'COMPOSED_DRAFT_ID_GATHER':72,'AUX_GATHER_CONCAT':72,'DRAFT_FIXED_BANK_TAILS':88,'TARGET_INPUT_ID_GATHER':72,'TARGET_POSITION_GATHER':72,'NEXT_STEP_POSITIONS':88,'SAMPLE_HIDDEN_ROWS':104,'INITIAL_SELECTOR_TEMP':16,'AUX_PREFIX_CONCAT':16,'MARKOV_SEED_COLUMN':104}
    contracts={
      'INITIAL_SELECTOR_TEMP':'FallbackQSL[1:]-1 computed when incoming selectorNone, but DSpark firstpass replaces incoming selector unconditionally before any read. Skip dead value only under matching DSpark override and no failure/debug observer; other proposers kept.',
      'DRAFT_FIXED_BANK_TAILS':'Four distinct persistentbanks:slot PAD-1,seq0,QSL0,tokenindices0. NativeFill is actual slice extent. First3 precede build_draft_attn_metadata;4th later, so cannot blindly coalesce all after last producer. Dirty full-bank suffix invariants/copy+fill output ABI with all readers/epochs/reset required.',
      'NEXT_STEP_POSITIONS':'used_update_positions only passed under should_update_next_steps=notparallel andCPguard. DSpark parallelTrue makes value dead; removing IndexCheck needs valid index domain/error contract. Otherwise gather needed. No main/dummy/prepost mask conflation.',
      'SAMPLE_HIDDEN_ROWS':'Real DSpark SAMPLE_FROM_ANCHOR can produce ordered arange query selectors; dummy may provide repeated zeros and lmhead alignment may pad zeros. Equal input/output shape alone does not permit view/identity. Direct head-row reader or privatecopy maintains snapshot/lifetime; precision and TP indexing retained.',
      'MARKOV_SEED_COLUMN':'ViewCopy writes only B INT64 values in column0 stride6;24-element descriptor is basecapacity. First Markov step can read seedbank directly only if dynamic_spec consumer gets correct full buffer/seed and fixedbank generation protected. Returnexcludescol0 does not prove nootherreaders.',
      'COMPOSED_DRAFT_ID_GATHER':'inputIDs[L] then temp[INT32(J+1)]. Compose nested indices, preserving native INT32add/wrap, negative normalization and full first-L validation even for unselected elements unless legaldomain proved. Can hold allS INT64 indices inlocal storage. ReturnedL metadata remains needed.',
      'TARGET_INPUT_ID_GATHER':'prepare_inputs_padded returns arange[:T] (not accepted-token compaction). Same-prefix TensorMove removes index work preserving copy. View only with no source overwritten before final reader; unpadded path arbitrary selections separate.',
      'TARGET_POSITION_GATHER':'INT64 positions source8192 is capacity; selectT rows. padded arange path same-prefix read; mrope2D uses separate branch. Slot linearposition, RoPE/MM position semantics not interchangeable; queue-roleonecall remains counted.',
      'AUX_PREFIX_CONCAT':'Three BF16[T,5120] prefixes concat to[T,15360]. F27 already has direct3-source singlemain-contraction option. New budget binds actual16concat TASK; do not split into3roundedBF16GEMMs. Lastreader/TPpartition/fullfeature norm stillrequired.',
      'AUX_GATHER_CONCAT':'Three selectedBF16[T,5120] copies thenconcat[T,15360]. Jointgather+concat reads index once, preserve order/repeats/negativeper-source normalization. F27existingindirectsingle15360Dcontraction canavoidallprivateplanes if samegeneration/allreaders/precision/layout satisfied. Identitypaddedpath distinct fromarbitraryunpadded.'}
    models=[]
    for r in rs:
        rid=r['regime_id'];assert len(ids[rid])==r['executions']and missing[rid]==r['missing_PMU']
        d=r['signature'];op=r['operator'];f=family(r['role']);a=dimensions(d['inputShapes_text']);z=dimensions(d['outputShapes_text']);it=d['inputDataTypes_text'].split(';');ot=d['outputDataTypes_text'].split(';');ib=[None if s is None or t not in DT else math.prod(s)*DT[t]for s,t in zip(a or [],it)];ob=[None if s is None or t not in DT else math.prod(s)*DT[t]for s,t in zip(z or [],ot)];on=math.prod(z[0])if z and z[0]is not None else None
        if op=='IndexCheck':io=ib[-1];work=dict(index_elements=math.prod(a[-1]),index_dtype=it[-1],no_tensor_value_output=True)
        elif op=='Index':
            T=z[0][0];width=math.prod(z[0][1:]);io=DT[it[-1]]*T+2*DT[ot[0]]*on;work=dict(selected_rows=T,row_elements=width,selected_source_elements=on,input_storage_capacity_elements=math.prod(a[0]),index_dtype=it[-1],negative_wrap_and_bounds_retained=True)
        elif op=='ConcatD':io=sum(ib)+sum(ob);work=dict(copied_elements=on,input_planes=len(a),input_row_widths=[s[-1]for s in a],dense_output_elements=on)
        elif op=='ViewCopy':
            assert f=='MARKOV_SEED_COLUMN'and it==['INT64']*8 and a[0]==[24];B=a[4][0];io=16*B;work=dict(selected_INT64_seed_elements=B,base_capacity_INT64_elements=24,base_capacity_is_not_copy_extent=True,output_column=0,row_stride_elements=6)
        elif op=='Fill':io=ob[0];work=dict(written_INT32_values=on,source_scalar_not_read_per_element=True)
        else:
            assert op in ['Add','Sub'];io=sum(x for x in ib+ob if x is not None);work=dict(pointwise_INT32_elements=on,unknown_empty_scalar_operand_not_guessed=True,native_INT32_wrap=True)
        models.append(dict(regime_id=rid,module_candidates=['M21'],operator=op,role=r['role'],origin=r['origin'],context=r['context'],signature=d,family=f,executions=r['executions'],native_TASK_keys=ids[rid],native_TASK_ns=stat(times[rid]),native_TASK_types=dict(types[rid]),missing_PMU=missing[rid],known_logical_value_bytes_excluding_control=io,work_model=work,consumer_contract=contracts[f],consumer_fusion_review_complete_in_this_turn=True,physical_numeric_ceiling_proved=False))
    paths=[S/'vllm-ascend/vllm_ascend'/x for x in ['spec_decode/llm_base_proposer.py','spec_decode/dspark_proposer.py','worker/model_runner_v1.py','models/deepseek_v41/dspark.py','ops/triton/spec_decode/utils.py']]+[S/'vllm/vllm'/x for x in ['v1/worker/gpu_model_runner.py','config/speculative.py','v1/spec_decode/llm_base_proposer.py']]
    pins={}
    for fn,k in [('M21-slot-rejection-state-and-packing-contract-v38.json','source_pins'),('M18-partition-bank-RoPE-and-slot-contract-v34.json','source_files')]:pins.update(read(O/fn)[k])
    sources=[]
    for q in paths:
        repo='vllm-ascend'if '/vllm-ascend/'in str(q)else'vllm';remote='/vllm-workspace/'+repo+'/'+str(q).split('/'+repo+'/',1)[1];sha=hashlib.sha256(q.read_bytes()).hexdigest();old=pins.get(remote)
        if old:assert sha==old
        sources.append(dict(path=str(q),remote=remote,sha256=sha,prior_historical_SHA_match=bool(old),loaded_code_object_attested=False))
    assert sum(s['prior_historical_SHA_match']for s in sources)==6
    assert sum(missing.values())==0
    write(O/'selected-input-regime-models-v70.json',dict(generated_utc=datetime.now(timezone.utc).isoformat(),regimes=models,regime_count=60,captured_execution_count=2024,regimes_by_family=dict(Counter(r['family']for r in models)),executions_by_family={f:sum(r['executions']for r in models if r['family']==f)for f in sorted(contracts)},scope='Last60nonplanM21regimes;exactselectedrowIOandconcreteconsumers/alternatives,notopaqueplanneralgorithm/wholemodelclosure.',overall_complete=False))
    write(O/'selected-input-native-chains-v70.json',dict(chains=chains,scope='Source-order groups checked perDB/stream/typedshape;notactualvalue/alias/epoch proof;criticalpathgainnotcomputed.',overall_complete=False))
    write(O/'selected-input-operator-contract-v70.json',dict(source_files=sources,contracts=contracts,existing_F27_indirect_mechanism='ReuseF27-v34-INDIRECTandv64consumer/precisionmodels;newnativeM21population/costbinding,notnewunseenfusion.',support_config_binding='TwoadditionallocalfrozenfilesbindDSparkparallelTrueandconstructorcopy;independenthistorical166fileSHA_not_reused. Preserve same-version condition.',same166revision_condition_for_two_support_files=True,new_remote_source_read=False,local_operator_clause_review=True,no_model_Torch_NPU_capture_health_process_kill_implementation=True,whole_modules_closed=0,overall_complete=False))
    print(dict(regimes=60,executions=2024,chains={k:len(v)for k,v in chains.items()},selected_missing_PMU=sum(missing.values()),sources=8,prior_historical_SHA_matches=6))

if __name__=='__main__':main()
