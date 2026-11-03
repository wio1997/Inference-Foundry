from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,socket,re
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis');E=P/'native-execution-identity-v14';C=P/'native-communications-v15';M=P/'native-PMU-complete-v18';OUT=M/'M19-allrank-DSpark-forward-embedding-HC-decoder-and-communication-scope-v35.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha,str(p);return json.loads(b)
fs=pin(E/'native-compute-fullchains-v14-v2.json','6c7bf4bd3a7dcf557327aeadc4a37ec4b9b23cb1ba675353bbd6e5d56a4e1ca0');ix=pin(E/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4');cf=pin(C/'communication-fullchains-v15.json','2781a2a848883e6dee25ce4af368017dbc09627f4c2adc57ada9b37305360800');cr=pin(C/'communication-and-queue-review-v15.json','c196d5083904a75a794c81fadfaf9d528d11461fdd5d81a6e941cad1e9693d6c')
def sites(frames):return[f for f in frames if '/deepseek_v41/dspark.py('in f and ': forward'in f]
def phase(frames):
 text='\n'.join(frames)
 if any(q in text for q in['_dummy_run',': dummy_run',': profile_run']):return'DUMMY_ANCESTRY'
 if any(q in text for q in['/llm_base_proposer.py','/dspark_proposer.py','propose_draft_token_ids']):return'PROPOSAL_ANCESTRY'
 return'OTHER_NATIVE_ANCESTRY'
def shape(c):return dict(op=c['opType_text'],input=c.get('inputShapes_text'),output=c.get('outputShapes_text'),input_types=c.get('inputDataTypes_text'),output_types=c.get('outputDataTypes_text'))
def census(cs):return[dict(signature=json.loads(k),TASKs=v)for k,v in Counter(json.dumps(shape(c),sort_keys=True)for c in cs).items()]
selected={};apis=[];bychain=defaultdict(list);keys=set()
for db in fs['databases']:
 di=db['database_index'];selected[di]={}
 for a in db['API_records']:
  ss=sites(a['project_frames'])
  if not ss:continue
  z=dict(database_index=di,API=a,DSpark_forward_sites=ss,phase=phase(a['project_frames']));selected[di][a['api_rowid']]=z;apis.append(z);keys.add(a['chain_key'])
tasks=[]
for ref in ix['databases']:
 di=ref['database_index'];d=pin(Path(ref['path']),ref['sha256']);cc={c['globalTaskId']:c for c in d['COMPUTE_definitions']}
 for t in d['TASK_records']:
  hits=sorted(set(t.get('native_linked_API_rowids',[]))&selected[di].keys())
  if not hits:continue
  rr=[selected[di][key]for key in hits];z=dict(database_index=di,TASK=t,definition=cc.get(t['globalTaskId']),phases=sorted({r['phase']for r in rr}),DSpark_forward_sites=sorted({s for r in rr for s in r['DSpark_forward_sites']}),source_APIs=[r['API']for r in rr]);tasks.append(z)
  for key in{r['API']['chain_key']for r in rr}:bychain[key].append(z)
opmap={(r['database_index'],r['COMM_OP_rowid']):r for r in cr['communication_operations']};comms=[]
for db in cf['databases']:
 for r in db['records']:
  hits=[dict(kind=g['kind'],API=a,DSpark_forward_sites=sites(a['project_frames']),phase=phase(a['project_frames']))for g in r['source_groups']for a in g['candidates']if sites(a['project_frames'])]
  if hits:comms.append(dict(database_index=db['database_index'],operation=opmap[(db['database_index'],r['COMM_OP_rowid'])],source_candidates=hits,phases=sorted({h['phase']for h in hits}),DSpark_forward_sites=sorted({s for h in hits for s in h['DSpark_forward_sites']})))
headers=[]
for key in sorted(keys):
 aa=[r for r in apis if r['API']['chain_key']==key];tt=bychain[key];frames=[f['frame']for f in fs['canonical_fullchains'][key]if'/vllm'in f['frame']]
 headers.append(dict(chain_key=key,APIs=len(aa),TASKs=len(tt),phases=sorted({r['phase']for r in aa}),DSpark_forward_sites=sites(frames),typed_shapes=census([r['definition']for r in tt if r['definition']]),project_frames=frames))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),native_APIs=len(apis),native_TASKs=len(tasks),native_chains=len(headers),phase_op_types=[dict(phase=k[0],op=k[1],TASKs=v)for k,v in Counter(('|'.join(t['phases']),t['definition']['opType_text']if t['definition']else'NO_COMPUTE_DEFINITION')for t in tasks).items()],DSpark_forward_site_counts=[dict(site=k,APIs=v)for k,v in Counter(s for r in apis for s in r['DSpark_forward_sites']).items()],source_bound_COMM=len(comms),collective_regimes=[dict(phase=k[0],family=k[1],dtype=k[2],count=k[3],rankSize=k[4],operations=v)for k,v in Counter(('|'.join(r['phases']),r['operation']['family'],r['operation']['dtype'],r['operation']['count'],r['operation']['rankSize'])for r in comms).items()],whole_module_closed=False,overall_complete=False)
result=dict(summary=summary,native_APIs=apis,native_TASKs=tasks,chain_headers=headers,collectives=comms,limits=['AllnativeDSparkforwardancestryincludingreuseddecoder/attention/MoEoperators;notM19functionalresourcebucketornewexclusivecandidateassignment.','Contextprecompute/combinehidden/headpostforwardarenotseparatelyrecountedhereunlessliteralforwardancestryispositive.','Noindex/count/shapeisrequest/pointer/weight/slotgenerationidentity.','CompleteCOMMrecordsandComputeInfohcomTASKsaredifferentpopulations.','Dummycostisnotstartup/removable;graph/target80pathsremainseparatefromnativeeagerdraft.','SavedprofileCPUonly;nosource/model/torch/NPU/service/healthwork.'])
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
compact=[dict(chain_key=h['chain_key'],APIs=h['APIs'],TASKs=h['TASKs'],phases=h['phases'],DSpark_forward_sites=h['DSpark_forward_sites'],ops=dict(Counter({c['signature']['op']:sum(z['TASKs']for z in h['typed_shapes']if z['signature']['op']==c['signature']['op'])for c in h['typed_shapes']})),leaf_project_frames=h['project_frames'][:4])for h in headers]
print(json.dumps(dict(summary=summary,compact_chain_headers=compact,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
