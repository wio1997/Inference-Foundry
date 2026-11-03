from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,socket

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis')
E=P/'native-execution-identity-v14';M=P/'native-PMU-complete-v18'
OUT=M/'target-sampler-rejection-and-draft-caller-surfaces-v28.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha,str(p);return json.loads(b)
fs=pin(E/'native-compute-fullchains-v14-v2.json','6c7bf4bd3a7dcf557327aeadc4a37ec4b9b23cb1ba675353bbd6e5d56a4e1ca0')
ix=pin(E/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4')
selected={};chainkeys=set();apirows=[]
for db in fs['databases']:
 di=db['database_index'];selected[di]={}
 for a in db['API_records']:
  frames=a.get('project_frames',[]);text='\n'.join(frames)
  if not any('/sample/'in f or'/layers/sampler.py'in f or'rejection_sampler'in f or'greedy_sample'in f or'random_sample'in f for f in frames):continue
  kind='DRAFT_PROPOSE_CALLER'if'propose_draft_token_ids'in text else('DUMMY_OR_PROFILE_CALLER'if any(x in text for x in ['_dummy_sampler_run','_dummy_run','profile_run'])else'TARGET_SAMPLER_CALLER')
  row=dict(database_index=di,API=a,context=kind)
  selected[di][a['api_rowid']]=row;apirows.append(row);chainkeys.add(a['chain_key'])
tasks=[];unlinked_field=Counter()
for ref in ix['databases']:
 d=pin(Path(ref['path']),ref['sha256']);di=ref['database_index'];cc={x['globalTaskId']:x for x in d['COMPUTE_definitions']}
 for t in d['TASK_records']:
  unlinked_field['PRESENT'if'native_linked_API_rowids'in t else'ABSENT_UNLINKED']+=1
  hits=sorted(set(t.get('native_linked_API_rowids',[]))&selected[di].keys())
  if hits:tasks.append(dict(database_index=di,TASK=t,definition=cc.get(t['globalTaskId']),source_APIs=[selected[di][h]for h in hits]))
bychain=defaultdict(list);roles=Counter();shapes=Counter();ambiguous=[]
for t in tasks:
 contexts=sorted({a['context']for a in t['source_APIs']})
 if len(contexts)!=1:ambiguous.append(dict(database_index=t['database_index'],TASK_rowid=t['TASK']['task_record_rowid'],contexts=contexts))
 for context in contexts:
  c=t['definition'];roles[(context,c['opType_text']if c else'NO_COMPUTE_DEFINITION')]+=1
  if c:shapes[(context,c['opType_text'],c['inputShapes_text'],c['inputDataTypes_text'],c['outputShapes_text'],c['outputDataTypes_text'])]+=1
 for key in {a['API']['chain_key']for a in t['source_APIs']}:bychain[key].append((t['database_index'],t['TASK']['task_record_rowid']))
chains=[]
for key in sorted(chainkeys):
 frames=fs['canonical_fullchains'][key]
 chains.append(dict(chain_key=key,source_API_records=sum(a['API']['chain_key']==key for a in apirows),
   TASK_execution_keys=bychain[key],TASKs=len(bychain[key]),
   frames=frames,project_frames=[f['frame']for f in frames if'/vllm'in f['frame']]))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),selected_native_APIs=len(apirows),selected_TASKs=len(tasks),selected_fullchains=len(chains),
    context_API_counts=dict(Counter(a['context']for a in apirows)),
    context_TASK_op_counts=[dict(context=k[0],opType=k[1],TASKs=v)for k,v in roles.items()],
    context_shape_counts=[dict(context=k[0],opType=k[1],inputShapes=k[2],inputDtypes=k[3],outputShapes=k[4],outputDtypes=k[5],TASKs=v)for k,v in shapes.items()],
    ambiguous_contexts=ambiguous,TASK_native_link_field_states=dict(unlinked_field),
    no_new_source_DB_model_device_or_service_work=True,overall_complete=False,
    limits=['Exact savednative-API source links only;unlinked graph/otherTASK work isnot invented or zero-filled.',
      'Dummy/profile iscallerpath context,not proofstartup/removablecost;draft-propose callers remainseparate.',
      'Observedgreedy/probability kernels do not alone resolve every sampler metadata flag or requested logprobs consumer.',
      'NativeAPI count isnot samplerinvocation count;full-vocab producer andCPU metadata costs areseparate.'])
result=dict(summary=summary,native_APIs=apirows,native_TASKs=tasks,canonical_fullchains=chains)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(summary=summary,chain_headers=[dict(chain_key=r['chain_key'],APIs=r['source_API_records'],TASKs=r['TASKs'],project_frames=r['project_frames'])for r in chains],full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
