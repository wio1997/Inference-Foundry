from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,re,socket

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis')
E=P/'native-execution-identity-v14';C=P/'native-communications-v15';M=P/'native-PMU-complete-v18'
OUT=M/'M15-final-HC-norm-row-selection-and-state-scope-v31.json';assert not OUT.exists()
def pin(path,sha):
    b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==sha,str(path);return json.loads(b)
fs=pin(E/'native-compute-fullchains-v14-v2.json','6c7bf4bd3a7dcf557327aeadc4a37ec4b9b23cb1ba675353bbd6e5d56a4e1ca0')
ix=pin(E/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4')
gr=pin(E/'graph-blocks-PMU-and-MC2-v14.json','a035c4de08191656141ad0727e8305a6444cc77e6184d4951f86136021af0991')
cf=pin(C/'communication-fullchains-v15.json','2781a2a848883e6dee25ce4af368017dbc09627f4c2adc57ada9b37305360800')
cr=pin(C/'communication-and-queue-review-v15.json','c196d5083904a75a794c81fadfaf9d528d11461fdd5d81a6e941cad1e9693d6c')
prep=pin(P/'prepare-field-origins-v12.json','c23b75f3fbd3ae3d3d7152f6882524e899bade53001ea2f1befa09ce943f091f')
roots={r['root_key']:r for r in prep['prepare_roots']};bydb=defaultdict(list);markers=defaultdict(list)
for r in roots.values():bydb[r['database_index']].append(r)
for b in prep['bindings']:markers[b['database_index']].append(b['marker'])
def links(di,a):
    mm=[m for m in markers[di]if m['globalTid']==a['globalTid']and m['startNs']<=a['startNs']and a['endNs']<=m['endNs']]
    return sorted({r['root_key']for m in mm for r in bydb[di]if r['root']['globalTid']==m['globalTid']and m['startNs']<=r['root']['startNs']and r['root']['endNs']<=m['endNs']})
def relevant(frames):
    return any(': hc_collapse'in f or re.search(r'/deepseek_v41/model.py\((?:159[3-9]|160[0-9])\)',f)or'/model_runner_v1.py(2634)'in f for f in frames)
def shape(c):return dict(op=c['opType_text'],input=c.get('inputShapes_text'),input_types=c.get('inputDataTypes_text'),output=c.get('outputShapes_text'),output_types=c.get('outputDataTypes_text'))
def census(cc):return[dict(signature=json.loads(k),TASKs=v)for k,v in Counter(json.dumps(shape(c),sort_keys=True)for c in cc).items()]
selected={};apirows=[];chains=set();bychain=defaultdict(list)
for db in fs['databases']:
    di=db['database_index'];selected[di]={}
    for a in db['API_records']:
        if relevant(a['project_frames']):
            row=dict(database_index=di,API=a,prepare_root_candidates=links(di,a));selected[di][a['api_rowid']]=row;apirows.append(row);chains.add(a['chain_key'])
native=[]
for ref in ix['databases']:
    di=ref['database_index'];d=pin(Path(ref['path']),ref['sha256']);cc={c['globalTaskId']:c for c in d['COMPUTE_definitions']}
    for t in d['TASK_records']:
        ids=sorted(set(t.get('native_linked_API_rowids',[]))&selected[di].keys())
        if not ids:continue
        rr=sorted({k for key in ids for k in selected[di][key]['prepare_root_candidates']})
        row=dict(database_index=di,TASK=t,definition=cc.get(t['globalTaskId']),source_APIs=[selected[di][key]['API']for key in ids],prepare_root_candidates=rr);native.append(row)
        for key in {a['chain_key']for a in row['source_APIs']}:bychain[key].append(row)
opmap={(v['database_index'],v['COMM_OP_rowid']):v for v in cr['communication_operations']};native_comms=[]
for db in cf['databases']:
    for r in db['records']:
        hits=[dict(source_group_kind=g['kind'],API=a,prepare_root_candidates=links(db['database_index'],a))for g in r['source_groups']for a in g['candidates']if relevant(a['project_frames'])]
        if hits:native_comms.append(dict(database_index=db['database_index'],operation=opmap[(db['database_index'],r['COMM_OP_rowid'])],source_candidates=hits,prepare_root_candidates=sorted({k for x in hits for k in x['prepare_root_candidates']})))
suffixes=[]
for g in gr['graph_executions']:
    left=next(b for b in g['blocks']if b['block_ordinal']==39)
    cc=sorted([v for v in g['outside_marker_tasks']if left['endNs']<=v['task']['startNs']],key=lambda v:(v['task']['startNs'],v['task']['task_record_rowid']))
    co=[r for r in cr['communication_operations']if r['database_index']==g['database_index']and left['endNs']<=r['startNs']and any(x['graph_ordinal']==g['graph_ordinal']for x in r['graph_candidates'])]
    suffixes.append(dict(database_index=g['database_index'],graph_ordinal=g['graph_ordinal'],prepare_root_keys=g['prepare_root_keys'],phase_records=g['phase_records'],last_block_endNs=left['endNs'],compute=cc,COMM_candidates=co,role_grade='EXACT_OUTER_SUFFIX;SOURCE_CONSISTENT_ROLES_NOT_NATIVE_STACK_OR_ALIAS_PROOF'))
headers=[]
for key in sorted(chains):
    rs=bychain[key];headers.append(dict(chain_key=key,APIs=sum(a['API']['chain_key']==key for a in apirows),TASKs=len(rs),typed_shapes=census([r['definition']for r in rs if r['definition']]),root_ambiguity_hist=dict(Counter(len(r['prepare_root_candidates'])for r in rs)),project_frames=[f['frame']for f in fs['canonical_fullchains'][key]if'/vllm'in f['frame']]))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),native_APIs=len(apirows),native_TASKs=len(native),native_chains=len(headers),native_typed_shapes=census([r['definition']for r in native if r['definition']]),native_root_ambiguity_hist=dict(Counter(len(r['prepare_root_candidates'])for r in native)),source_bound_native_collectives=len(native_comms),native_collective_regimes=[dict(family=x[0],dtype=x[1],count=x[2],rankSize=x[3],operations=v)for x,v in Counter((r['operation']['family'],r['operation']['dtype'],r['operation']['count'],r['operation']['rankSize'])for r in native_comms).items()],graph_executions=len(suffixes),graph_suffix_compute_hist=dict(Counter(len(s['compute'])for s in suffixes)),graph_suffix_types=dict(Counter(r['definition']['opType_text']for s in suffixes for r in s['compute'])),graph_suffix_shapes=census([r['definition']for s in suffixes for r in s['compute']]),graph_suffix_COMM_count_hist=dict(Counter(len(s['COMM_candidates'])for s in suffixes)),whole_module_closed=False,overall_complete=False)
result=dict(summary=summary,native_APIs=apirows,native_TASKs=native,native_canonical_fullchains={k:fs['canonical_fullchains'][k]for k in chains},native_collectives=native_comms,graph_suffixes=suffixes,limits=['Exactsource/nativeflowandmarkercontainmentonly;notnearesttimepairing.','Selectorsincludeallhc_collapsecallerssoactualtarget/draft/auxrolesmustbeseparated,notsummedblindly.','COMM/ComputeInfo/APIscopesaredifferentlevels;noadditivelatencyclaim.','Graphsuffixeshavepartialphasebinding;allunboundrecordsremain.','Noactualselectedrowindices,rawsnapshotobservers,alias/storage/eventgenerationsornumericalparityinferredfromcounts.','Nonewsource/DB/model/NPU/service/healthwork.'])
blob=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(blob)
print(json.dumps(dict(summary=summary,native_chain_headers=headers,full_result=str(OUT),result_sha256=hashlib.sha256(blob).hexdigest()),ensure_ascii=False))
