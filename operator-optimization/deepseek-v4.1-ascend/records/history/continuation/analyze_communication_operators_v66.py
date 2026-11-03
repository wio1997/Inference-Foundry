"""Reconstruct all saved HCCL operators locally, with exact namespace joins."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,sqlite3
from finalize_origin_coverage_v9 import N,O,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_state_engram_v62 import fullkey
from analyze_projection_ceiling_v61 import stat

def main():
    assert not (O/'communication-native-operator-population-v66.json').exists()
    roles=read(O/'communication-role-contract-v16.json')
    cr={r['chain_key']:r for r in roles['communication_chains']}
    queue=roles['queue_source']
    universe=[r for r in read(O/'operator-shape-ceiling-ledger-v55.json')['observed_regimes']if r['module_candidates']==['M23']]
    assert len(universe)==17 and sum(r['executions']for r in universe)==11528
    lookup={fullkey(r['module_candidates'],r['operator'],r['role'],r['origin'],r['context'],r['signature']):r for r in universe}
    assigns=defaultdict(dict)
    for a in read(CACHE/'analysis/native-PMU-complete-v18/resource-functional-attribution-and-uncertainty-v20-v3.json')['execution_assignments']:
        if a[3]==['M23']:assigns[a[0]][a[1]]=a
    graphs=read(CACHE/'analysis/native-execution-identity-v14/graph-blocks-PMU-and-MC2-v14.json')['graph_executions']
    bygraph=defaultdict(dict)
    for g in graphs:bygraph[g['database_index']][g['graph_ordinal']]=g
    records=[];graph_rows=[];source_kind=Counter();internal_total=0;schedule_definitions=0;paths=[]
    regkeys=defaultdict(list);regops=defaultdict(list);regtimes=defaultdict(list)
    for di in range(32):
        e=read(CACHE/f'analysis/native-execution-identity-v14/db-{di:02d}.json')
        r=Path(e['database_path']);i=max(i for i,x in enumerate(r.parts)if x=='analysis');path=CACHE/Path(*r.parts[i:])
        # Original immutable DB hash is bound by cached copy receipts; recheck bytes here.
        sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==e['database_sha256'],path
        paths.append(dict(database_index=di,path=str(path),sha256=sha))
        db=sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON')
        strings=dict(db.execute('SELECT id,value FROM STRING_IDS'))
        dt={r['id']:r['name']for r in db.execute('SELECT * FROM ENUM_HCCL_DATA_TYPE')}
        tasks={t['task_record_rowid']:t for t in e['TASK_records']}
        defs={d['globalTaskId']:d for d in e['COMPUTE_definitions']}
        compute_by_conn=defaultdict(list);schedule_by_conn=defaultdict(list)
        schedules={r['globalTaskId']:dict(r)for r in db.execute('SELECT rowid AS rowid,* FROM COMMUNICATION_SCHEDULE_TASK_INFO')}
        schedule_definitions+=len(schedules)
        for t in e['TASK_records']:
            if t['task_record_rowid']in assigns[di]:compute_by_conn[t['connectionId']].append(t)
            if t['globalTaskId']in schedules:schedule_by_conn[t['connectionId']].append(t)
        hosts=defaultdict(set);apis=defaultdict(list)
        for c in e['CONNECTION_IDS']:hosts[c['connectionId']].add(c['id'])
        for a in e['linked_PYTORCH_APIs']:apis[a['connectionId']].append(a)
        ops=[dict(r)for r in db.execute('SELECT rowid AS COMM_OP_rowid,* FROM COMMUNICATION_OP ORDER BY startNs')]
        need={a['callchainId']for op in ops for h in hosts[op['connectionId']]for a in apis[h]if a['callchainId']is not None}
        if di==queue['database_index']:need.add(queue['source']['callchainId'])
        frames=defaultdict(list)
        if need:
            values=','.join(str(int(v))for v in sorted(need))
            for r in db.execute('SELECT c.id,c.stackDepth,s.value AS frame FROM PYTORCH_CALLCHAINS c JOIN STRING_IDS s ON s.id=c.stack WHERE c.id IN ('+values+') ORDER BY c.id,c.stackDepth'):
                frames[r['id']].append(dict(stackDepth=r['stackDepth'],frame=r['frame']))
        def chain_key(callchain):
            f=frames[callchain]
            return hashlib.sha256(json.dumps(f,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()if f else None
        internals=defaultdict(Counter);internal_n=Counter();alg=defaultdict(Counter)
        for t in db.execute('SELECT * FROM COMMUNICATION_TASK_INFO'):
            internals[t['opId']][strings.get(t['taskType'],str(t['taskType']))]+=1;internal_n[t['opId']]+=1
            alg[t['opId']][str(t['transportType'])+':'+str(t['linkType'])]+=1
        internal_total+=sum(internal_n.values())
        controls=e['graph_API_control_bindings'];dbrecs=[]
        for op in ops:
            same_interval=lambda t:(t['startNs'],t['endNs'])==(op['startNs'],op['endNs'])
            candidates=[t for t in compute_by_conn[op['connectionId']]if same_interval(t)]
            sched=[t for t in schedule_by_conn[op['connectionId']]if same_interval(t)]
            assert len(candidates)<=1,(di,op['COMM_OP_rowid'],len(candidates))
            node=candidates[0]if candidates else None
            typ=strings[op['opType']]
            row=dict(database_index=di,COMM_OP_rowid=op['COMM_OP_rowid'],opId=op['opId'],operator=typ,
                count=op['count'],dtype=dt[op['dataType']],rankSize=op['rankSize'],group_name=strings[op['groupName']],
                algorithm_label=strings[op['algType']],startNs=op['startNs'],endNs=op['endNs'],duration_ns=op['endNs']-op['startNs'],
                connectionId=op['connectionId'],TASK_rowid=node['task_record_rowid']if node else None,streamId=node['streamId']if node else None,
                node_namespace='COMPUTE' if candidates else 'NO_COMPUTEINFO_NODE',
                exact_same_connection_interval_schedule_TASK_rows=[t['task_record_rowid']for t in sched],internal_records=internal_n[op['opId']],
                internal_record_types=dict(internals[op['opId']]),transport_link_enum_record_counts=dict(alg[op['opId']]))
            graph=[(j,g)for j,g in enumerate(controls)if g['control_pair_valid']and g['control_pair_start_ns']<=op['startNs']and op['endNs']<=g['control_pair_end_ns']]
            chains={chain_key(a['callchainId'])for h in hosts[op['connectionId']]for a in apis[h]if a['callchainId']is not None}
            chains.discard(None)
            if chains:
                assert len(chains)==1 and not graph;key=next(iter(chains));c=cr[key]
                row.update(source_grade='DIRECT_NATIVE_FULLCHAIN',semantic_role=c['semantic_role'],chain_key=key,source_module=c['module'])
            elif di==queue['database_index']and op['COMM_OP_rowid']==queue['COMM_OP_rowid']:
                assert op['connectionId']==queue['native_connection']and typ=='hcom_alltoall_'
                a=dict(db.execute('SELECT rowid AS api_rowid,* FROM PYTORCH_API WHERE rowid=?',(queue['source']['api_rowid'],)).fetchone())
                assert a['callchainId']==queue['source']['callchainId']and chain_key(a['callchainId'])==queue['source']['chain_key']
                c=cr[queue['source']['chain_key']]
                row.update(source_grade='REUSED_EXPLICIT_QUEUE_FLOW_PROOF',semantic_role=c['semantic_role'],source_module=c['module'],chain_key=c['chain_key'])
            else:
                assert len(graph)==1 and not chains,(di,op['COMM_OP_rowid'])
                gi,control=graph[0];g=bygraph[di][gi]
                blocks=[b['block_ordinal']for b in g['blocks']if b['startNs']<=op['startNs']and op['endNs']<=b['endNs']]
                assert len(blocks)<=1
                if blocks:
                    role={'hcom_allGather_':'ATTENTION_INPUT_SP_ALLGATHER','hcom_alltoall_':'ATTENTION_RESTORE_TP_HEADS_ALLTOALL','hcom_reduceScatter_':'ATTENTION_OUTPUT_SP_REDUCESCATTER'}[typ]
                elif typ=='hcom_allReduce_':role='VOCAB_EMBEDDING_ALLREDUCE'
                elif row['dtype']=='FP32':role='MTP_FP32_PREMIX_ALLGATHER'
                else:role='GRAPH_OUTSIDE_BF16_GATHER_PENDING_CLASSIFICATION'
                row.update(source_grade='GRAPH_CONTROL_AND_ORDER_SCHEMA_NOT_TENSOR_EDGE',semantic_role=role,
                    graph_ordinal=gi,block_ordinal=blocks[0]if blocks else None,root_bound=bool(g['prepare_root_keys']))
            source_kind[row['source_grade']]+=1
            if candidates:
                t=candidates[0];a=assigns[di][t['task_record_rowid']];d=defs[t['globalTaskId']]
                rid=lookup[fullkey(a[3],d['opType_text'],a[5],a[2],a[6],d)]['regime_id'];assert d['opType_text']==typ
                row['compute_regime_id']=rid;regkeys[rid].append([di,t['task_record_rowid']]);regops[rid].append([di,op['COMM_OP_rowid']]);regtimes[rid].append(row['duration_ns'])
            dbrecs.append(row)
        for gi,g in bygraph[di].items():
            rs=[r for r in dbrecs if r.get('graph_ordinal')==gi];assert len(rs)==126
            for block in g['blocks']:
                rr=[r for r in rs if r['block_ordinal']==block['block_ordinal']]
                assert len(rr)==3 and Counter(r['operator']for r in rr)=={'hcom_allGather_':1,'hcom_alltoall_':1,'hcom_reduceScatter_':1}
            outside=[r for r in rs if r['block_ordinal']is None];assert len(outside)==6
            bf=[r for r in outside if r['semantic_role']=='GRAPH_OUTSIDE_BF16_GATHER_PENDING_CLASSIFICATION'];assert len(bf)==4
            small=min(r['count']for r in bf)
            assert Counter(r['count']for r in bf)=={small:3,small*4:1}
            for r in bf:r['semantic_role']='MTP_FULL_HC_STATE_ALLGATHER'if r['count']==small*4 else'DSPARK_AUX_RESIDUAL_ALLGATHER'
            graph_rows.append(dict(database_index=di,graph_ordinal=gi,root_bound=bool(g['prepare_root_keys']),
                calls=126,role_counts=dict(Counter(r['semantic_role']for r in rs)),COMM_OP_keys=[[di,r['COMM_OP_rowid']]for r in rs],
                grade='Exactcontrolcontainer+40block3operationorder/count/dtypeinvariants,notnewtensorpointerorPythonstackproof.'))
        records+=dbrecs;db.close()
    assert len(records)==14720 and internal_total==557688
    assert Counter(r['node_namespace']for r in records)=={'COMPUTE':11528,'NO_COMPUTEINFO_NODE':3192}
    assert schedule_definitions==3192
    assert source_kind=={'DIRECT_NATIVE_FULLCHAIN':4639,'REUSED_EXPLICIT_QUEUE_FLOW_PROOF':1,'GRAPH_CONTROL_AND_ORDER_SCHEMA_NOT_TENSOR_EDGE':10080}
    assert len(graph_rows)==80 and sum(g['root_bound']for g in graph_rows)==64
    rows=[]
    for r in universe:
        rid=r['regime_id'];assert len(regkeys[rid])==r['executions']and r['missing_PMU']==0
        rows.append(dict(regime_id=rid,executions=r['executions'],module_candidates=r['module_candidates'],operator=r['operator'],
            role=r['role'],origin=r['origin'],context=r['context'],signature=r['signature'],native_TASK_keys=regkeys[rid],
            exact_COMM_OP_keys=regops[rid],native_interval_ns=stat(regtimes[rid]),compute_shapes_not_available=True,
            scope='HCCLnamedComputeInfoisnotadditionalarithmeticwork;actualdtype/count/PfromexactCOMMjoin.3192schedule-onlycallsremainseparatebroaderpopulation.'))
    write(O/'communication-native-operator-population-v66.json',dict(generated_utc=datetime.now(timezone.utc).isoformat(),
        operations=records,operation_count=14720,compute_named_nodes=11528,operations_without_ComputeInfo_node=3192,
        communication_schedule_definitions=schedule_definitions,noncompute_ops_to_schedule_one_to_one_join_proved=False,
        internal_records_count=internal_total,source_grades=dict(source_kind),raw_database_files=paths,
        semantic_role_counts=dict(Counter(r['semantic_role']for r in records)),
        limits='11528exactDB-localconnection+interval+operatorComputejoins;3192otherCOMMopsretainedwithoutinventedone-to-oneSchedulejoin(theircountcoincideswith3192Scheduledefinitions). Graphconnectionreusedacrossexecutionsrequiresintervalqualification. Internal557688recordsarenotadditionalmessages/wirebytes;COMMcount/APIaritymustbeboundseparately. Noendpointspeedup/physicalMFU.',overall_complete=False))
    write(O/'communication-compute-regime-bindings-v66.json',dict(regimes=rows,regime_count=17,captured_execution_count=11528,
        source_scope='All14720COMMoperatorsmodeledseparatelyincluding3192notinComputeInfo.',overall_complete=False))
    write(O/'communication-graph-role-bindings-v66.json',dict(graphs=graph_rows,graph_count=80,root_bound_graphs=64,phase_unbound_graphs=16,overall_complete=False))
    print(dict(COMM_operations=len(records),ComputeInfo=11528,non_ComputeInfo_ops=3192,schedule_defs_separate=3192,internal_records=internal_total,
        regimes=17,graph_counts=[80,64,16],source_grades=dict(source_kind),role_counts=dict(Counter(r['semantic_role']for r in records))))

if __name__=='__main__':main()
