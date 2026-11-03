"""Use CONNECTION_IDS sets to bind MemSet helpers to their actual constant-pad API invocation."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import hashlib,math
from finalize_origin_coverage_v9 import N,O,W,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_state_engram_v62 import fullkey,dimensions
from analyze_projection_ceiling_v61 import stat

def main():
    assert not (O/'pad-owner-invocations-v72.json').exists()
    cv=read(O/'operator-analysis-coverage-ledger-v71.json');pending={r['regime_id']:r for r in cv['regimes']if r['state'].startswith('REGIME_')}
    all_regimes=read(O/'operator-shape-ceiling-ledger-v55.json')['observed_regimes']
    regs=[r for r in all_regimes if r['regime_id']in pending and r['operator']=='MemSet']
    assert len(regs)==7 and sum(r['executions']for r in regs)==1272
    lookup={fullkey(r['module_candidates'],r['operator'],r['role'],r['origin'],r['context'],r['signature']):r for r in regs}
    all_lookup={fullkey(r['module_candidates'],r['operator'],r['role'],r['origin'],r['context'],r['signature']):r for r in all_regimes}
    selected=defaultdict(dict);assigns=defaultdict(dict)
    for a in read(CACHE/'analysis/native-PMU-complete-v18/resource-functional-attribution-and-uncertainty-v20-v3.json')['execution_assignments']:assigns[a[0]][a[1]]=a
    owners=[];native_keys=defaultdict(list);durations=defaultdict(list);all_names=Counter();kinds=Counter();noncompute=Counter();duplicates=[]
    for di in range(32):
        e=read(CACHE/f'analysis/native-execution-identity-v14/db-{di:02d}.json');tasks={t['task_record_rowid']:t for t in e['TASK_records']};defs={d['globalTaskId']:d for d in e['COMPUTE_definitions']}
        set_to_con=defaultdict(set);con_to_set=defaultdict(set);api_by_set=defaultdict(list);tasks_by_con=defaultdict(list)
        for x in e['CONNECTION_IDS']:set_to_con[x['id']].add(x['connectionId']);con_to_set[x['connectionId']].add(x['id'])
        for api in e['linked_PYTORCH_APIs']:api_by_set[api['connectionId']].append(api)
        for task in e['TASK_records']:tasks_by_con[task['connectionId']].append(task)
        for tid,a in assigns[di].items():
            task=tasks[tid];d=defs[task['globalTaskId']];key=fullkey(a[3],d['opType_text'],a[5],a[2],a[6],d)
            if key not in lookup:continue
            r=lookup[key];rid=r['regime_id'];native_keys[rid].append([di,tid]);durations[rid].append(task['endNs']-task['startNs']);all_names[d['name_text']]+=1
            candidates={x['api_rowid']:x for setid in con_to_set[task['connectionId']]for x in api_by_set[setid]if x['name_text']=='aclnnConstantPadNd'}
            assert len(candidates)==1,(di,tid,list(candidates))
            api=next(iter(candidates.values()));connections=set_to_con[api['connectionId']]
            bound=sorted({t['task_record_rowid']:t for con in connections for t in tasks_by_con[con]if t['globalPid']==task['globalPid']}.values(),key=lambda x:(x['startNs'],x['task_record_rowid']))
            stages=[];pads=[];memsets=[]
            for t in bound:
                de=defs.get(t['globalTaskId']);assignment=assigns[di].get(t['task_record_rowid']);reg=None
                if de is not None and assignment is not None:
                    reg=all_lookup[fullkey(assignment[3],de['opType_text'],assignment[5],assignment[2],assignment[6],de)]
                row=dict(TASK_rowid=t['task_record_rowid'],connectionId=t['connectionId'],task_type=t['task_type_name'],streamId=t['streamId'],startNs=t['startNs'],endNs=t['endNs'],duration_ns=t['endNs']-t['startNs'],operator=de['opType_text']if de else None,name=de['name_text']if de else None,regime_id=reg['regime_id']if reg else None)
                stages.append(row)
                if de is None:noncompute[t['task_type_name']]+=1
                if de is not None and de['opType_text']=='PadV3':pads.append((t,de,reg))
                if de is not None and de['opType_text']=='MemSet':memsets.append(t)
            assert len(memsets)==1 and memsets[0]['task_record_rowid']==tid and len(pads)==1,(di,tid,Counter(x['operator']for x in stages))
            pt,pad,preg=pads[0];assert preg is not None and preg['module_candidates']==r['module_candidates']and preg['role']==r['role']and preg['context']==r['context']
            assert len({t['streamId']for t in bound})==1
            ash=dimensions(pad['inputShapes_text']);zsh=dimensions(pad['outputShapes_text']);dtype=pad['inputDataTypes_text'].split(';')[0]
            assert ash[0]is not None and zsh[0]is not None and ash[0][1:]==zsh[0][1:]and zsh[0][0]>ash[0][0]
            field='MC2_HIDDEN'if '(282)'in r['role']else 'MC2_ROUTER'if '(283)'in r['role']else 'MC2_MASK'if '(284)'in r['role']else 'SP_MASK'if '(120)'in r['role']else 'SP_HIDDEN'if dtype=='DT_BF16'else 'SP_IDS'
            kinds[field]+=1
            owners.append(dict(database_index=di,MemSet_TASK_rowid=tid,MemSet_regime_id=rid,field=field,module_candidates=r['module_candidates'],role=r['role'],context=r['context'],owner_API_table='PYTORCH_API',owner_API_rowid=api['api_rowid'],owner_API_name=api['name_text'],owner_connection_SET_id=api['connectionId'],owner_runtime_connection_ids=sorted(connections),owner_API_startNs=api['startNs'],owner_API_endNs=api['endNs'],PadV3_TASK_rowid=pt['task_record_rowid'],PadV3_regime_id=preg['regime_id'],PadV3_signature=pad,stages=stages,native_compute_operator_counts=dict(Counter(s['operator']for s in stages if s['operator'])),native_noncompute_type_counts=dict(Counter(s['task_type']for s in stages if s['operator']is None)),native_task_sum_ns=sum(s['duration_ns']for s in stages),native_enclosing_body_span_ns=bound[-1]['endNs']-bound[0]['startNs'],MemSet_operand_address_or_length_proved=False,owner_join_is_not_temporal_adjacency=True,source_pure_pad_replacement_scope='EliminateentireconstantpadAPIonlyunderfulloutputsemantic/error/alias/readinesscontract. NoindependentMemSetdeletionorbyteextentclaim.'))
    assert len(owners)==1272 and len({(x['database_index'],x['owner_API_rowid'])for x in owners})==1272
    assert set(all_names)=={'aclnnConstantPadNd_PadV3AiCore_MemSet'}
    assert kinds=={'MC2_HIDDEN':320,'MC2_ROUTER':320,'MC2_MASK':320,'SP_MASK':104,'SP_HIDDEN':104,'SP_IDS':104}
    models=[]
    for r in regs:
        rid=r['regime_id'];assert len(native_keys[rid])==r['executions'];models.append(dict(regime_id=rid,executions=r['executions'],module_candidates=r['module_candidates'],role=r['role'],native_TASK_keys=native_keys[rid],native_TASK_ns=stat(durations[rid]),exact_enclosing_constantpad_API_proved=True,independent_operand_address_size_still_unknown=True,separate_MemSet_fusion_or_physical_traffic_complete=False,conditional_enclosing_replacement_analyzed=True,overall_complete=False))
    compute=Counter()
    for x in owners:compute.update(x['native_compute_operator_counts'])
    write(O/'pad-owner-invocations-v72.json',dict(generated_utc=datetime.now(timezone.utc).isoformat(),invocations=owners,owner_API_calls=1272,native_compute_counts=dict(compute),native_noncompute_counts=dict(noncompute),fields=dict(kinds),kernel_names=dict(all_names),scope='ExactCONNECTION_IDS-set→PYTORCH_APIjoinwithTASKconnectionmembership. NonComputeTASKsretainedseparately.Notinferredbytime/sourceparent/adjacentPadshape.',overall_complete=False))
    write(O/'pad-helper-regime-refinement-v72.json',dict(regimes=models,regime_count=7,native_execution_count=1272,pending_regimes_promoted=0,pending_execution_promoted=0,scope='EnclosingPadreplacementproofadded;7operand-specificIO/byte/memorylevelgapspreserved. Nohelperidentity/sizeinvented.',overall_complete=False))
    write(O/'pad-owner-API-namespace-reconciliation-v72.json',dict(prior_record='router-MemSet-profile-insufficiency-v63.json',prior_field='native_records[].linked_CANN_API_names',prior_value='launch',correction='native_linked_API_rowidsarePYTORCH_APIidentitiesinthiscache;CANN_APIROWIDhasaseparatenamespace. JoinTASK.connectionId→CONNECTION_IDS.connectionId→setid→PYTORCH_API.connectionIdtoresolveaclnnConstantPadNd. Priorcross-tableROWIDnamelookupdoesnotproveCANNowner.',all1272_owners_independently_rejoined=True,byte_or_operand_gap_not_resolved_by_owner_name=True,global_functional_assignments_unchanged=3136,scope='OnlythisMemSetownerlookupcorrected;donotextrapolateotherglobalcorrelationproofswithoutaudit.',overall_complete=False))
    print(dict(owners=len(owners),fields=dict(kinds),compute=dict(compute),noncompute=dict(noncompute),operand_gaps_retained=[7,1272],pending_regimes_promoted=0))

if __name__=='__main__':main()
