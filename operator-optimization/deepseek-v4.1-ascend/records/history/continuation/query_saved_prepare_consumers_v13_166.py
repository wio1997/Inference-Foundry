from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib, json, socket

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
b=(P/'analysis/prepare-field-origins-v12.json').read_bytes()
assert hashlib.sha256(b).hexdigest()=='c23b75f3fbd3ae3d3d7152f6882524e899bade53001ea2f1befa09ce943f091f'
x=json.loads(b);records=x['selected_API_records'];chains=x['full_canonical_callchains']
def project(key):
    return [f['frame'] for f in chains[key] if '/vllm_ascend/' in f['frame'] or '/vllm/vllm/' in f['frame']]
byroot=defaultdict(list)
for r in records:byroot[r['root_key']].append(r)
batch=[];utility_groups=defaultdict(list);sync_groups=defaultdict(list)
for root in x['prepare_roots']:
    rs=byroot[root['root_key']]
    suffix=[r for r in rs if r['name']=='acl_memcpy_host_to_device' and 'dsa_v41_cp.py(140)' in project(r['chain_key'])[0]]
    copies=[r for r in rs if r['name']=='aten::copy_' and 'dsa_v41.py(903)' in project(r['chain_key'])[0]]
    classified=[]
    for r in copies:
        fs=project(r['chain_key'])
        scope='CP_GLOBAL' if any('dsa_v41_cp.py(114)' in f for f in fs) else 'CP_LOCAL' if any('dsa_v41_cp.py(166)' in f for f in fs) else 'OTHER_NATIVE'
        classified.append(dict(scope=scope,api_rowid=r['api_rowid'],startNs=r['startNs'],endNs=r['endNs'],chain_key=r['chain_key'],input_shapes=r['input_shapes'],input_dtypes=r['input_dtypes']))
    local=[r for r in classified if r['scope']=='CP_LOCAL'];other=[r for r in classified if r['scope']=='OTHER_NATIVE']
    root_initializations=sorted(local+other,key=lambda r:r['startNs'])
    batch.append(dict(root_key=root['root_key'],suffix_H2D=len(suffix),batch_copy_scope_counts=dict(Counter(r['scope'] for r in classified)),initialization_copies=classified,
        suffix_after_first_local_batch_copy=sum(r['startNs']>local[0]['endNs'] for r in suffix) if len(local)==1 else None,
        suffix_after_first_root_batch_copy=sum(r['startNs']>root_initializations[0]['endNs'] for r in suffix) if root_initializations else None,
        suffix_API_rowids=[r['api_rowid'] for r in suffix]))
    for r in rs:
        fs=project(r['chain_key'])
        if r['name']=='acl_memcpy_host_to_device' and 'vllm/v1/utils.py(143)' in fs[0]:
            parents=[q for q in rs if q['name']=='aten::copy_' and q['chain_key']==r['chain_key'] and q['startNs']<=r['startNs'] and r['endNs']<=q['endNs']]
            minimum=min((q['endNs']-q['startNs'] for q in parents),default=None)
            inner=[q for q in parents if q['endNs']-q['startNs']==minimum]
            utility_groups[r['chain_key']].append(dict(root_key=r['root_key'],database_index=r['database_index'],H2D_api_rowid=r['api_rowid'],parent_candidate_count=len(parents),innermost_parent_count=len(inner),
                parents=[dict(api_rowid=q['api_rowid'],input_shapes=q['input_shapes'],input_dtypes=q['input_dtypes']) for q in inner]))
        elif r['name']=='Event::synchronize':sync_groups[r['chain_key']].append(r)
utility=[]
for key,rs in sorted(utility_groups.items()):
    utility.append(dict(chain_key=key,project_frames=project(key),records=len(rs),roots=len({r['root_key'] for r in rs}),
        unique_innermost_parent_records=sum(r['innermost_parent_count']==1 for r in rs),
        shape_dtype_counts=[dict(input_shapes=k[0],input_dtypes=k[1],records=v) for k,v in Counter((p['input_shapes'],p['input_dtypes']) for r in rs for p in r['parents']).items()],H2D_parent_correspondences=rs))
sync=[dict(chain_key=k,project_frames=project(k),records=len(rs),roots=len({r['root_key'] for r in rs}),row_keys=[dict(root_key=r['root_key'],database_index=r['database_index'],api_rowid=r['api_rowid']) for r in rs]) for k,rs in sorted(sync_groups.items())]
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),roots=len(batch),nonempty_suffix_roots=sum(r['suffix_H2D']>0 for r in batch),
    batch_copy_scope_histograms={scope:dict(Counter(r['batch_copy_scope_counts'].get(scope,0) for r in batch)) for scope in ['CP_GLOBAL','CP_LOCAL','OTHER_NATIVE']},
    suffix_after_first_local_batch_copy_hist=dict(Counter(str(r['suffix_after_first_local_batch_copy']) for r in batch)),
    suffix_after_first_root_batch_copy_hist=dict(Counter(str(r['suffix_after_first_root_batch_copy']) for r in batch)),
    later_suffix_H2D_after_root_batch_init=sum(r['suffix_after_first_root_batch_copy'] or 0 for r in batch),
    utility_H2D_records=sum(u['records'] for u in utility),utility_full_caller_chains=len(utility),utility_records_with_unique_innermost_parent=sum(u['unique_innermost_parent_records'] for u in utility),
    sync_records=sum(s['records'] for s in sync),sync_full_caller_chains=len(sync),
    no_new_source_read=True,no_new_DB_or_device_work=True,whole_modules_closed=0,overall_complete=False,
    limits=['Cachehit interpretation stillrequires source shared-dictionary/consumer contract;temporal APIorder alone is not dead-value proof.',
            'H2D parent relation uses same exact fullchain,root andhostinterval containment;not nearesttime,tensor/storage identity or physicalDMAbytes.',
            'Every observedfullcaller is retained;first project helper is not a unique metadatafield.'])
OUT=P/'analysis/saved-prepare-consumers-v13.json';assert not OUT.exists();payload=json.dumps(dict(summary=summary,batch_initialization_and_suffix=batch,utility_copy_callers=utility,synchronize_callers=sync,full_canonical_callchains=chains,source_databases=x['databases']),ensure_ascii=False,separators=(',',':'))+'\n';OUT.write_text(payload)
public=dict(summary=summary,full_result=str(OUT),result_sha256=hashlib.sha256(payload.encode()).hexdigest(),utility_callers=[{k:v for k,v in u.items() if k not in ['H2D_parent_correspondences','shape_dtype_counts']} for u in utility],sync_callers=[{k:v for k,v in s.items() if k!='row_keys'} for s in sync])
if len(json.dumps(public).encode())>30000:public.pop('utility_callers');public['display_truncated']=True
print(json.dumps(public,ensure_ascii=False))
