from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis')
def pin(path,sha):
 b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==sha,(path,'SHA');return json.loads(b)
C=P/'native-communications-v15';E=P/'native-execution-identity-v14'
chains=pin(C/'communication-fullchains-v15.json','2781a2a848883e6dee25ce4af368017dbc09627f4c2adc57ada9b37305360800')
review=pin(C/'communication-and-queue-review-v15.json','c196d5083904a75a794c81fadfaf9d528d11461fdd5d81a6e941cad1e9693d6c')
proj=pin(E/'graph-projection-domains-v15.json','6de529df96fa255becf4febdc3de48d5bde1417d59605ff56729db9c2cc5af56')
idx=pin(E/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4')
OUT=C/'communication-roles-and-projection-streams-v16.json';assert not OUT.exists()
opmap={(r['database_index'],r['COMM_OP_rowid']):r for r in review['communication_operations']}
bychain=defaultdict(list)
for db in chains['databases']:
 for op in db['records']:
  keys={a['chain_key']for g in op['source_groups']for a in g['candidates']if a['chain_key']}
  assert len(keys)<=1
  for k in keys:bychain[k].append(opmap[(db['database_index'],op['COMM_OP_rowid'])])
chainrows=[]
for key,frames in chains['canonical_fullchains'].items():
 ops=bychain[key];reg=Counter((o['family'],o['dtype'],o['rankSize'],o['count'],o['group_name'])for o in ops)
 chainrows.append(dict(chain_key=key,operations=len(ops),project_frames=[f['frame']for f in frames if '/vllm_ascend/'in f['frame']or '/vllm/vllm/'in f['frame']],regimes=[dict(family=k[0],dtype=k[1],rankSize=k[2],count=k[3],group_name=k[4],operations=v)for k,v in sorted(reg.items())],operation_keys=[(o['database_index'],o['COMM_OP_rowid'])for o in ops]))
graphs=review['graph_communication_partitions'];graphhist=Counter(json.dumps(g['per_block_operation_count_hist'],sort_keys=True)for g in graphs)
seqgroups=defaultdict(list)
for g in graphs:seqgroups[g['recorded_sequence_signature_sha256']].append(g)
graphseq=[dict(signature=k,graphs=len(gs),family_counts=gs[0]['family_counts'],per_block_operation_count_hist=gs[0]['per_block_operation_count_hist'],outside_block_sequence=[s for s in gs[0]['sequence']if not s[4]],first_block_sequence=[s for s in gs[0]['sequence']if s[4]==[0]])for k,gs in seqgroups.items()]
bydb=defaultdict(list)
for r in proj['records']:bydb[r['database_index']].append(r)
streamrows=[]
for ref in idx['databases']:
 d=pin(Path(ref['path']),ref['sha256']);di=d['database_index'];tm={t['task_record_rowid']:t for t in d['TASK_records']};bygraph=defaultdict(list)
 for r in bydb[di]:
  v=dict(database_index=di,graph_ordinal=r['graph_ordinal'],block_ordinal=r['block_ordinal'],candidates={})
  for role,cs in r['candidates'].items():
   v['candidates'][role]=[dict(c,streamId=tm[c['task_record_rowid']]['streamId'],contextId=tm[c['task_record_rowid']]['contextId'],startNs=tm[c['task_record_rowid']]['startNs'],endNs=tm[c['task_record_rowid']]['endNs'])for c in cs]
  bygraph[r['graph_ordinal']].append(v)
 for gi,rows in bygraph.items():
  baseline=Counter(c['streamId']for r in rows if len(r['candidates']['KV'])==1 for c in r['candidates']['KV'])
  qstreams=Counter(c['streamId']for r in rows for c in r['candidates']['Q_A'])
  ambiguous=[r for r in rows if len(r['candidates']['KV'])!=1]
  for a in ambiguous:a['candidates_on_unique_unambiguous_KV_stream']=[c for c in a['candidates']['KV']if len(baseline)==1 and c['streamId']in baseline]
  streamrows.append(dict(database_index=di,graph_ordinal=gi,unambiguous_KV_stream_counts=dict(baseline),Q_stream_counts=dict(qstreams),ambiguous_blocks=ambiguous))
amb=[a for r in streamrows for a in r['ambiguous_blocks']]
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),native_COMM_source_operations=sum(r['operations']for r in chainrows),canonical_communication_chains=len(chainrows),graph_count=len(graphs),per_graph_marker_block_COMM_hist=dict(graphhist),graph_sequence_signatures=len(graphseq),ambiguous_projection_blocks=len(amb),ambiguous_block_ordinals=dict(Counter(a['block_ordinal']for a in amb)),graphs_with_one_unambiguous_KV_stream=sum(len(r['unambiguous_KV_stream_counts'])==1 for r in streamrows),ambiguous_with_one_candidate_on_unique_KV_stream=sum(len(a['candidates_on_unique_unambiguous_KV_stream'])==1 for a in amb),no_source_read_or_device_work=True,overall_complete=False,limits=['Communication source role requires full caller review; direct chains do not assign Python source to graph-only operations.','Same stream association is an observed invariant, not tensor pointer or all-consumer proof. No typed candidate was discarded.','No operation-duration sum is interpreted as request gain.'])
payload=(json.dumps(dict(summary=summary,communication_chains=chainrows,graph_sequence_groups=graphseq,projection_streams=streamrows),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
short=lambda s:s.replace('/vllm-workspace/vllm-ascend/vllm_ascend/','a/').replace('/vllm-workspace/vllm/vllm/','v/')
public=dict(summary=summary,communication_chains=[dict(chain_key=r['chain_key'],operations=r['operations'],project_frames=[short(f)for f in r['project_frames']],regimes=r['regimes'])for r in chainrows],graph_sequence_groups=graphseq,projection_samples=streamrows[:1],full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest())
print(json.dumps(public,ensure_ascii=False))
