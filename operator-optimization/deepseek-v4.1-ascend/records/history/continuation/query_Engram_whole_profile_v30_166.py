from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis');E=P/'native-execution-identity-v14';C=P/'native-communications-v15';M=P/'native-PMU-complete-v18'
out=M/'Engram-allrank-native-graph-and-collective-scope-v30.json';assert not out.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha,str(p);return json.loads(b)
fs=pin(E/'native-compute-fullchains-v14-v2.json','6c7bf4bd3a7dcf557327aeadc4a37ec4b9b23cb1ba675353bbd6e5d56a4e1ca0')
ix=pin(E/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4')
gr=pin(E/'graph-blocks-PMU-and-MC2-v14.json','a035c4de08191656141ad0727e8305a6444cc77e6184d4951f86136021af0991')
cf=pin(C/'communication-fullchains-v15.json','2781a2a848883e6dee25ce4af368017dbc09627f4c2adc57ada9b37305360800')
cr=pin(C/'communication-and-queue-review-v15.json','c196d5083904a75a794c81fadfaf9d528d11461fdd5d81a6e941cad1e9693d6c')
def relevant(frames):return any('/engram/'in f or'/engram.py'in f or'prepare_engram'in f for f in frames)
selected={};apirows=[];chains=set();bychain=defaultdict(list)
for db in fs['databases']:
 di=db['database_index'];selected[di]={}
 for a in db['API_records']:
  if not relevant(a['project_frames']):continue
  row=dict(database_index=di,API=a);selected[di][a['api_rowid']]=row;apirows.append(row);chains.add(a['chain_key'])
native=[]
for ref in ix['databases']:
 di=ref['database_index'];d=pin(Path(ref['path']),ref['sha256']);cc={c['globalTaskId']:c for c in d['COMPUTE_definitions']}
 for t in d['TASK_records']:
  ids=sorted(set(t.get('native_linked_API_rowids',[]))&selected[di].keys())
  if ids:
   row=dict(database_index=di,TASK=t,definition=cc.get(t['globalTaskId']),source_APIs=[selected[di][key]['API']for key in ids]);native.append(row)
   for key in {a['chain_key']for a in row['source_APIs']}:bychain[key].append(row)
graph_segments=[]
for g in gr['graph_executions']:
 for previous,following in[(0,1),(13,14)]:
  left=next(b for b in g['blocks']if b['block_ordinal']==previous);right=next(b for b in g['blocks']if b['block_ordinal']==following)
  candidates=[v for v in g['outside_marker_tasks']if left['endNs']<=v['task']['startNs']and v['task']['endNs']<=right['startNs']]
  candidates.sort(key=lambda v:(v['task']['startNs'],v['task']['task_record_rowid']))
  graph_segments.append(dict(database_index=g['database_index'],graph_ordinal=g['graph_ordinal'],previous_block=previous,next_block=following,prepare_root_keys=g['prepare_root_keys'],phase_records=g['phase_records'],
   evidence_grade='EXACT_OUTER_SEGMENT;ENGRAM_ROLE_SOURCE_CONSISTENT_NOT_POINTER_PROOF',compute=candidates))
opmap={(v['database_index'],v['COMM_OP_rowid']):v for v in cr['communication_operations']};comms=[]
for db in cf['databases']:
 for r in db['records']:
  hits=[dict(source_group_kind=s['kind'],API=a)for s in r['source_groups']for a in s['candidates']if relevant(a['project_frames'])]
  if hits:comms.append(dict(database_index=db['database_index'],operation=opmap[(db['database_index'],r['COMM_OP_rowid'])],source_candidates=hits))
headers=[]
for key in sorted(chains):
 rs=bychain[key];headers.append(dict(chain_key=key,APIs=sum(a['API']['chain_key']==key for a in apirows),TASKs=len(rs),op_types=dict(Counter(r['definition']['opType_text']if r['definition']else'NO_COMPUTE_DEFINITION'for r in rs)),
  project_frames=[f['frame']for f in fs['canonical_fullchains'][key]if'/vllm'in f['frame']]))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),native_selected_APIs=len(apirows),native_selected_TASKs=len(native),native_fullchains=len(headers),
 native_op_types=dict(Counter(r['definition']['opType_text']if r['definition']else'NO_COMPUTE_DEFINITION'for r in native)),
 graph_executions=len(gr['graph_executions']),graph_Engram_segments=len(graph_segments),segment_compute_count_hist=dict(Counter(len(s['compute'])for s in graph_segments)),
 graph_Engram_op_types=dict(Counter(r['definition']['opType_text']for s in graph_segments for r in s['compute'])),
 source_bound_collectives=len(comms),collective_regimes=[dict(family=k[0],dtype=k[1],rankSize=k[2],count=k[3],operations=v)for k,v in Counter((r['operation']['family'],r['operation']['dtype'],r['operation']['rankSize'],r['operation']['count'])for r in comms).items()],
 no_new_source_DB_model_device_service_work=True,overall_complete=False,
 limits=['Graphoutersegmentsareexactmarkerpartitionswithsource-consistentroles;notnewnativePythonstacksortensor-pointeraliasproof.',
 'EngramCPUhash/cache/hits/readiness/bytegenerationcannotbezero-filledfromComputeInfoabsence.',
 'No nearest-time attributionofnon-computeMEMCPY;priorv11copyevidenceremainsscoped,separatefromthiscurrentcompute/COMMquery.',
 'Savedgraph/native/COMMpopulationsoverlaphierarchicallyandmustnotbeaddedasrequestlatency.'])
result=dict(summary=summary,native_APIs=apirows,native_TASKs=native,native_canonical_fullchains={k:fs['canonical_fullchains'][k]for k in chains},graph_segments=graph_segments,collectives=comms)
blob=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();out.write_bytes(blob)
print(json.dumps(dict(summary=summary,native_chain_headers=headers,first_segment=graph_segments[:1],full_result=str(out),result_sha256=hashlib.sha256(blob).hexdigest()),ensure_ascii=False))
