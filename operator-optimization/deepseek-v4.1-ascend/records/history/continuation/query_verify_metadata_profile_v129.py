"""Read cached32DB operator ancestry only; no remote/model/runtime work."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,re,sqlite3
from finalize_origin_coverage_v9 import N,O,read,write


def frame_roles(s):
    out=[]
    m=re.search(r'/spec_decode/llm_base_proposer.py\((\d+)\)',s)
    if m:
        line=int(m[1]);roles={1481:'FIRST_PASS_INPUT_BUILD',1494:'DRAFT_DECODER',1584:'SEED_PUBLICATION',
            1587:'MARKOV_EMBED',1588:'MARKOV_BIAS',1589:'MARKOV_ADD',1599:'GREEDY_ARGMAX',1602:'ID_PUBLICATION'}
        if line in roles:out.append(roles[line])
    if ': prepare_next_token_ids_padded'in s:out.append('NEXT_ID_COUNT')
    m=re.search(r'/spec_decode/utils.py\((\d+)\)',s)
    if m and 278<=int(m[1])<=635:out.append('DYNAMIC_SCHEDULER_SOURCE_SURFACE')
    m=re.search(r'/worker/model_runner_v1.py\((\d+)\)',s)
    if m and 2205<=int(m[1])<=2226:out.append('KEEP_LENGTH_RETRIEVAL_SOURCE_SURFACE')
    if '/deepseek_v41/dspark.py(439)'in s or '/deepseek_v41/dspark.py(441)'in s:out.append('CONFIDENCE_WRAPPER_SOURCE_SURFACE')
    if '/model_executor/models/qwen3_dspark.py('in s:
        line=int(re.search(r'\((\d+)\)',s)[1])
        if 185<=line<=189:out.append('CONFIDENCE_HEAD_SOURCE_SURFACE')
    return out


def main():
    assert not (O/'REVIEW-CHECKPOINT-v129.json').exists()
    refs=read(O/'host-launch-original-API-coverage-v126.json')['databases'];dbs=[];selected=[];totals=Counter();task_totals=Counter()
    for ref in refs:
        p=Path(ref['path']);blob=p.read_bytes();assert len(blob)==ref['bytes']and hashlib.sha256(blob).hexdigest()==ref['sha256'];before=p.stat()
        con=sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True);con.row_factory=sqlite3.Row
        strings={r['id']:r['value']for r in con.execute('SELECT *FROM STRING_IDS')}
        fr={i:frame_roles(s)for i,s in strings.items()if frame_roles(s)};chains=defaultdict(set);hits=defaultdict(list)
        for row in con.execute('SELECT *FROM PYTORCH_CALLCHAINS'):
            if row['stack']in fr:
                chains[row['id']].update(fr[row['stack']]);hits[row['id']].append(dict(stackDepth=row['stackDepth'],frame=strings[row['stack']]))
        apis=[];cids=defaultdict(list)
        for row in con.execute('SELECT rowid AS api_rowid,*FROM PYTORCH_API WHERE callchainId IS NOT NULL'):
            if row['callchainId']not in chains:continue
            a=dict(row);a['startNs']=int(a['startNs']);a['endNs']=int(a['endNs'])
            a['roles']=sorted(chains[row['callchainId']]);a['name_text']=strings.get(a['name']);a['matched_frames']=sorted(hits[row['callchainId']],key=lambda x:x['stackDepth'])
            apis.append(a)
            if a['connectionId']is not None:cids[a['connectionId']].append(a)
        runtime=defaultdict(list)
        for row in con.execute('SELECT *FROM CONNECTION_IDS'):
            for a in cids.get(row['id'],[]):runtime[row['connectionId']].append(a['api_rowid'])
        info={r['globalTaskId']:dict(r)for r in con.execute('SELECT *FROM COMPUTE_TASK_INFO')};byapi=defaultdict(set);tasks={}
        for row in con.execute('SELECT rowid AS task_record_rowid,*FROM TASK'):
            if row['connectionId']not in runtime:continue
            t=dict(row);d=info.get(row['globalTaskId']);t['opType_text']=strings.get(d['opType'])if d else None
            t['inputShapes_text']=strings.get(d['inputShapes'])if d else None
            tasks[t['task_record_rowid']]=t
            for aid in runtime[row['connectionId']]:byapi[aid].add(t['task_record_rowid'])
        role_apis=Counter();role_keys=defaultdict(set)
        for a in apis:
            a['native_TASK_rowids']=sorted(byapi[a['api_rowid']])
            for role in a['roles']:role_apis[role]+=1;role_keys[role].update(a['native_TASK_rowids'])
            # Keep the exact per-API raw keys/times/frames. No nearest-event field binding.
            selected.append(dict(database_index=ref['database_index'],**a))
        role_task_counts={r:len(keys)for r,keys in role_keys.items()}
        dbs.append(dict(path=str(p),sha256=ref['sha256'],bytes=ref['bytes'],database_index=ref['database_index'],
            API_role_counts=dict(role_apis),TASK_role_counts=role_task_counts,
            TASK_types_by_role={r:dict(Counter(tasks[k]['opType_text']or'NO_COMPUTE'for k in keys))for r,keys in role_keys.items()},
            stat_unchanged=True))
        totals.update(role_apis);task_totals.update(role_task_counts)
        con.close();after=p.stat();assert(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    result=dict(databases=dbs,selected_APIs=selected,API_role_counts=dict(totals),TASK_role_counts=dict(task_totals),
        join='PYTORCH_API.connectionId -> CONNECTION_IDS.id -> CONNECTION_IDS.connectionId -> TASK.connectionId;callchain roles preserve recorded stackDepth/source lines.',
        scope='All32cachedDB recorded Torch callchains. Absence does not prove flags disabled or whole observer closure. Host API order is not device completion or cross-iteration identity.',
        dynamic_feature_or_metadata_values_attested=False,no_remote_model_Torch_CANN_or_NPU_work=True,overall_complete=False)
    write(O/'verify-metadata-profile-scope-v129.json',result)
    print(json.dumps(dict(status='PROFILE_SCOPE_RECONCILED',databases=len(dbs),selected_APIs=len(selected),API_role_counts=dict(totals),TASK_role_counts=dict(task_totals),new_device_work=False)))


if __name__=='__main__':main()
