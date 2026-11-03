"""Audit exact profiler correlation for unresolved native operands;no temporal matching."""
from collections import Counter,defaultdict
from pathlib import Path
import hashlib,json,sqlite3
from finalize_origin_coverage_v9 import N,O,W,read,write
from mirror_saved_profile_core_v53 import CACHE


def pin(p):
    p=Path(p);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
    assert not(O/'REVIEW-CHECKPOINT-v125.json').exists()
    refs=read(O/'raw-operator-evidence-extraction-v98.json')['result']['databases']
    pending=read(O/'remaining-operand-evidence-status-v98.json')['execution_bindings']
    bydb=defaultdict(list)
    for r in pending:bydb[r['database_index']].append(r)
    source=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend/attention/dsa_v41.py'
    inputs=[O/'raw-operator-evidence-extraction-v98.json',O/'remaining-operand-evidence-status-v98.json',
        O/'operand-evidence-method-gates-v107.json',O/'native-pad-consequences-v120.json',
        N/'native-pad-probe-results-v120b/probe-result.json',O/'sparse-index-postcheck-result-v124b.json',
        O/'sparse-index-postcheck-repair-v124b.json',N/'verify_sparse_index_semantics_v124b.py',source]
    write(O/'operand-correlation-inputs-v125.json',dict(inputs=[pin(p)for p in inputs],
        questions=['Can native pending task connection IDs be joined to specific recorded CPU operator calls?',
            'Does a same-shape BOOL24 zeros_like stack identify the unresolved graph task or a different eager task?',
            'Which original operands require capture-creation versus eager-launch evidence,without guessing addresses from names?'],
        no_remote_Torch_NPU_model_or_service_work=True,overall_complete=False))
    rows=[];dbs=[];bool_cpu=[];stacks={};totals=Counter();source_lines=source.read_text().splitlines()
    for ref in refs:
        i=ref['database_index'];relative=ref['path'].split('/analysis/offline-db-v12/',1)[1]
        p=CACHE/'analysis/offline-db-v12'/relative
        identity=pin(p);assert identity['sha256']==ref['sha256'] and identity['bytes']==ref['bytes']
        before=p.stat();con=sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True);con.row_factory=sqlite3.Row
        con.execute('PRAGMA temp_store=MEMORY')
        strings=dict(con.execute('SELECT id,value FROM STRING_IDS'))
        pending_cids=[r['connectionId']for old in bydb[i]for r in con.execute('SELECT connectionId FROM TASK WHERE rowid=?',(old['task_record_rowid'],))]
        values=sorted(set(pending_cids));placeholders=','.join('?'for _ in values)
        cann_by_cid=defaultdict(list);cpu_by_cid=defaultdict(list)
        for x in con.execute('SELECT rowid AS cann_api_rowid,* FROM CANN_API WHERE connectionId IN ('+placeholders+')',values):
            cann_by_cid[x['connectionId']].append(dict(x,name_text=strings.get(x['name'])))
        linked_cpu=list(con.execute('SELECT p.rowid AS pytorch_api_rowid,p.*,c.connectionId AS runtime_connection_id FROM PYTORCH_API p JOIN CONNECTION_IDS c ON p.connectionId=c.id WHERE c.connectionId IN ('+placeholders+')',values))
        for x in linked_cpu:cpu_by_cid[x['runtime_connection_id']].append(x)
        zeros=list(con.execute("SELECT p.rowid AS pytorch_api_rowid,p.* FROM PYTORCH_API p JOIN STRING_IDS s ON p.name=s.id WHERE s.value='aten::zeros_like'"))
        bool_roots=[r for r in zeros if str(strings.get(r['inputDtypes'])).split(';',1)[0].strip().lower()=='bool'and str(strings.get(r['inputShapes'])).split(';',1)[0].strip()=='24']
        inplace_ids=[k for k,v in strings.items()if v=='aclnnInplaceZero']
        backend=[]
        for name_id in inplace_ids:
            backend.extend(con.execute('SELECT rowid AS pytorch_api_rowid,* FROM PYTORCH_API WHERE name=? AND connectionId IS NOT NULL',(name_id,)))
        selected_children={r['pytorch_api_rowid']:[x for x in backend if x['globalTid']==r['globalTid']and int(x['startNs'])>=int(r['startNs'])and int(x['endNs'])<=int(r['endNs'])]for r in bool_roots}
        chain_ids=sorted({r['callchainId']for r in linked_cpu+bool_roots+[x for rr in selected_children.values()for x in rr]if r['callchainId']is not None})
        chain_rows=defaultdict(list)
        if chain_ids:
            for x in con.execute('SELECT * FROM PYTORCH_CALLCHAINS WHERE id IN ('+','.join('?'for _ in chain_ids)+') ORDER BY id,stackDepth',chain_ids):
                chain_rows[x['id']].append(dict(stackDepth=x['stackDepth'],frame=strings.get(x['stack'])))
        definitions={r['globalTaskId']:r for r in con.execute('SELECT rowid AS compute_record_rowid,* FROM COMPUTE_TASK_INFO')}
        def cpu_info(row):
            r=dict(row)
            for field in ['name','inputDtypes','inputShapes']:
                r[field+'_text']=strings.get(r.get(field))
            cid=r.get('callchainId')
            if cid is not None:
                key=str(i)+':'+str(cid)
                if key not in stacks:
                    stacks[key]=chain_rows[cid]
                r['stack_ref']=key
            return r
        for old in bydb[i]:
            t=dict(con.execute('SELECT rowid AS task_record_rowid,* FROM TASK WHERE rowid=?',(old['task_record_rowid'],)).fetchone())
            definition=definitions[t['globalTaskId']]
            assert strings[definition['opType']]==old['operator']
            cann=cann_by_cid[t['connectionId']]
            cpu=[cpu_info(x)for x in cpu_by_cid[t['connectionId']]]
            graph=t['modelId']!=0xffffffff
            state=('CAPTURED_TASK_WITHOUT_RECORDED_CPU_PRODUCER' if graph and not cpu else
                'RECORDED_OPERATOR_CORRELATION_METADATA_ONLY' if cpu else 'NO_RECORDED_CPU_OPERATOR_LINK')
            totals[old['operator']+':'+state]+=1
            rows.append(dict(database_index=i,database_sha256=identity['sha256'],operator=old['operator'],TASK=t,
                compute_definition=dict(globalTaskId=t['globalTaskId'],name_text=strings.get(definition['name']),
                    input_shapes=strings.get(definition['inputShapes']),input_dtypes=strings.get(definition['inputDataTypes'])),
                CANN_APIs=cann,exact_connection_CPU_APIs=cpu,correlation_state=state,
                capture_model_task=graph,no_timestamp_or_nearest_neighbor_match_used=True,
                destination_address=None,native_write_length=None,BOOL_semantic_field_or_all_readers=None))
        # Full CPU BOOL24 zeros_like census;follow nested backend API group IDs exactly.
        for r in bool_roots:
            root=cpu_info(r);start,end=int(r['startNs']),int(r['endNs'])
            children=[cpu_info(x)for x in selected_children[r['pytorch_api_rowid']]]
            links=[]
            for child in children:
                for link in con.execute('SELECT id,connectionId FROM CONNECTION_IDS WHERE id=?',(child['connectionId'],)):
                    taskrows=[dict(x)for x in con.execute('SELECT rowid AS task_record_rowid,* FROM TASK WHERE connectionId=?',(link['connectionId'],))]
                    for task in taskrows:
                        d=definitions.get(task['globalTaskId'])
                        links.append(dict(child_pytorch_api_rowid=child['pytorch_api_rowid'],connection_group_id=link['id'],runtime_connection_id=link['connectionId'],
                            TASK=task,native_operator=None if d is None else strings.get(d['opType']),
                            native_input_shape=None if d is None else strings.get(d['inputShapes']),
                            is_one_of_original_pending_tasks=any(x['task_record_rowid']==task['task_record_rowid']for x in bydb[i])))
            bool_cpu.append(dict(database_index=i,root=root,backend_children=children,exact_native_links=links))
        schemas={name:[x['name']for x in con.execute('PRAGMA table_info("'+name+'")')]for name in ['TASK','CANN_API','CONNECTION_IDS','PYTORCH_API','PYTORCH_CALLCHAINS']}
        con.close();after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        dbs.append(dict(database_index=i,**identity,schemas=schemas,SQLite_mode='ro&immutable=1',stat_unchanged=True))
    assert len(rows)==1352 and Counter(r['operator']for r in rows)=={'MemSet':1272,'ZerosLike':80}
    write(O/'operand-correlation-native-records-v125.json',dict(records=rows,summary=dict(totals),databases=dbs,overall_complete=False))
    write(O/'operand-correlation-CPU-stacks-v125.json',dict(stacks=stacks,BOOL24_zeros_like_calls=bool_cpu,
        exact_backend_group_join='PYTORCH_API.connectionId -> CONNECTION_IDS.id -> CONNECTION_IDS.connectionId -> TASK.connectionId. CANN_API.connectionId is the runtime namespace. Do not compare group ID directly to TASK ID.',
        no_dataflow_storage_or_reader_identity_claim_from_time_containment=True,overall_complete=False))
    graph_count=sum(r['capture_model_task']for r in rows);cpu_linked=sum(bool(r['exact_connection_CPU_APIs'])for r in rows)
    eager_bool_links=[x for c in bool_cpu for x in c['exact_native_links']]
    probe=read(N/'native-pad-probe-results-v120b/probe-result.json')
    kernels=[t for t in probe['tasks']if t.get('type')==1 and t.get('get_params_returncode')==0]
    write(O/'operand-correlation-consequences-v125.json',dict(
        original_pending_executions=1352,original_pending_regimes=9,captured_pending_tasks=graph_count,
        eager_pending_tasks=1352-graph_count,pending_with_exact_CPU_API_link=cpu_linked,
        pending_without_exact_CPU_API_link=1352-cpu_linked,CPU_BOOL24_zeros_like_calls=len(bool_cpu),
        CPU_BOOL24_exact_linked_tasks=len(eager_bool_links),CPU_BOOL24_links_to_original_pending=sum(x['is_one_of_original_pending_tasks']for x in eager_bool_links),
        important_limit='An exact recorded CPU backend/launch link can bind its source call and logical geometry,but supplies no native MemSet destination/write extent. A captured template connection ID without a recorded CPU producer cannot be repaired by nearest timestamp,shape or a different same-name eager op.',
        BOOL_false_match='Audit every BOOL24 CPU zeros_like stack and exact backend link separately. A C2 complete-mask eager initialization is not an unresolved prefix graph value merely because shape,dtype and operator match.',
        getter_probe_fact=dict(successful_kernel_metadata_queries=len(kernels),isHostArgs_counts=dict(Counter(str(t['isHostArgs'])for t in kernels)),
            raw_args_not_dereferenced=True,conclusion='Actual tiny-Pad getter records mark args as device-side(isHostArgs=0). Do not ctypes.string_at these opaque addresses as host packet memory;getter size is not an operand extent.'),
        collection_split=[dict(domain='CAPTURED_TEMPLATE',required='Original capture-creation/modelRI/task/parameter-update epoch plus supported bounded argument dump and Python storage/field identity. Replay-only CPU scopes cannot reconstruct omitted capture operators.'),
            dict(domain='EAGER_NATIVE_HELPER',required='Same-call launch/argument evidence or a justified equivalent owned capture;graph-only collector does not automatically cover eager tasks. Exact recorded API/stack alone still lacks native pointer/length.'),
            dict(domain='BOOL_FIELD_AND_ALL_READERS',required='Exact graph value/storage generation and all actual reader/mutation/alias contracts. A source stack for a different eager BOOL24 task is not a substitute.')],
        additional_probe_or_model_capture_authorized=False,new_device_work=False,new_native_promotions=0,
        physical_traffic_time_or_fusion_gain_proved=False,overall_complete=False))
    source_part=dict(**pin(source),first_line=1334,last_line=1350,source=source_lines[1333:1350])
    write(O/'operand-correlation-source-and-case-witness-v125.json',dict(source_part=source_part,
        source_fact='C2 complete=(input_positions%2==1)&valid;under skip_ring_update,complete=torch.zeros_like(complete),then copy into persistent complete-mask bank. This is a source role for the matching CPU stack only,not a promotion of the separate unresolved graph tasks.',
        cases=bool_cpu[:1],no_native_duplicate_field_binding_by_shape=True,overall_complete=False))
    print(json.dumps(dict(status='ANALYZED',databases=len(dbs),pending_tasks=len(rows),summary=dict(totals),
        captured=graph_count,eager=1352-graph_count,CPU_BOOL24_calls=len(bool_cpu),BOOL_exact_links=len(eager_bool_links),
        BOOL_links_to_pending=sum(x['is_one_of_original_pending_tasks']for x in eager_bool_links),new_device_work=False,overall_complete=False)))


if __name__=='__main__':main()
