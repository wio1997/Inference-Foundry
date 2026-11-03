from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis');E=P/'native-execution-identity-v14';M=P/'native-PMU-complete-v18';OUT=M/'M17-all-native-sampler-and-global-distribution-primitive-scope-v33.json';assert not OUT.exists()
def pin(p,s):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==s,str(p);return json.loads(b)
fs=pin(E/'native-compute-fullchains-v14-v2.json','6c7bf4bd3a7dcf557327aeadc4a37ec4b9b23cb1ba675353bbd6e5d56a4e1ca0');ix=pin(E/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4');prep=pin(P/'prepare-field-origins-v12.json','c23b75f3fbd3ae3d3d7152f6882524e899bade53001ea2f1befa09ce943f091f')
def sampler(frames):return any('/sample/'in f or'/layers/sampler.py'in f or'rejection_sampler'in f or'greedy_sample'in f or'random_sample'in f for f in frames)
def candidate(op):
 s=op.lower();return s in{'exp','log','log1p'}or any(x in s for x in['softmax','multinomial','random','uniform','gumbel','cumsum','sort','topk','rejection','entropy'])
all_apis={};selected={};chains=set();apirows=[]
for db in fs['databases']:
 di=db['database_index'];all_apis[di]={a['api_rowid']:a for a in db['API_records']};selected[di]={}
 for a in db['API_records']:
  if sampler(a['project_frames']):selected[di][a['api_rowid']]=a;apirows.append(dict(database_index=di,API=a));chains.add(a['chain_key'])
native=[];global_candidates=[];bychain=defaultdict(list)
for ref in ix['databases']:
 di=ref['database_index'];d=pin(Path(ref['path']),ref['sha256']);cc={c['globalTaskId']:c for c in d['COMPUTE_definitions']}
 for t in d['TASK_records']:
  c=cc.get(t['globalTaskId']);linked=t.get('native_linked_API_rowids',[]);hits=sorted(set(linked)&selected[di].keys())
  if hits:
   z=dict(database_index=di,TASK=t,definition=c,source_APIs=[selected[di][key]for key in hits]);native.append(z)
   for key in{a['chain_key']for a in z['source_APIs']}:bychain[key].append(z)
  if c and candidate(c['opType_text']):
   aa=[all_apis[di][key]for key in linked if key in all_apis[di]]
   kinds=sorted({'SAMPLER'if sampler(a['project_frames'])else'OTHER_NATIVE_ANCESTRY'for a in aa})or['NO_NATIVE_API_SOURCE_IN_FULLCHAIN_EXPORT']
   global_candidates.append(dict(database_index=di,TASK_rowid=t['task_record_rowid'],globalTaskId=t['globalTaskId'],op=c['opType_text'],name=c.get('name_text'),input=c.get('inputShapes_text'),output=c.get('outputShapes_text'),source_kinds=kinds,source_chain_keys=sorted({a['chain_key']for a in aa}),native_link_field_present='native_linked_API_rowids'in t))
fields={'all_greedy','all_random','max_num_logprobs','enable_entropy_verify','enable_block_verify','synthetic_mode','num_prompt_logprobs','prompt_logprobs','logprobs_mode','temperature'};recorded=[]
def visit(v,path,key):
 if isinstance(v,dict):
  for name,z in v.items():
   pp=path+'.'+name
   if name in fields:
    val=dict(type=type(z).__name__,size=len(z),empty=not z)if isinstance(z,(dict,list))else z
    recorded.append(dict(root_key=key,path=pp,value=val))
   if isinstance(z,(dict,list)):visit(z,pp,key)
 elif isinstance(v,list):
  for z in v:
   if isinstance(z,(dict,list)):visit(z,path+'[]',key)
for r in prep['prepare_roots']:visit(r.get('metadata'),'metadata',r['root_key'])
headers=[dict(chain_key=key,APIs=sum(r['API']['chain_key']==key for r in apirows),TASKs=len(bychain[key]),op_types=dict(Counter(r['definition']['opType_text']if r['definition']else'NO_COMPUTE_DEFINITION'for r in bychain[key])),project_frames=[f['frame']for f in fs['canonical_fullchains'][key]if'/vllm'in f['frame']])for key in sorted(chains)]
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),sampler_native_APIs=len(apirows),sampler_native_TASKs=len(native),sampler_native_chains=len(headers),sampler_TASK_types=dict(Counter(r['definition']['opType_text']if r['definition']else'NO_COMPUTE_DEFINITION'for r in native)),global_distribution_candidate_TASKs=len(global_candidates),global_distribution_candidate_types=[dict(op=k[0],source_kind=k[1],TASKs=v)for k,v in Counter((r['op'],'|'.join(r['source_kinds']))for r in global_candidates).items()],recorded_sampler_guard_fields=[dict(path=k[0],value=json.loads(k[1]),records=v)for k,v in Counter((r['path'],json.dumps(r['value'],sort_keys=True))for r in recorded).items()],missing_guard_fields='UNKNOWN_NOT_FALSE_OR_EMPTY',whole_module_closed=False,overall_complete=False)
result=dict(summary=summary,native_sampler_API_records=apirows,native_sampler_TASK_records=native,canonical_sampler_fullchains={k:fs['canonical_fullchains'][k]for k in chains},global_distribution_candidate_rows=global_candidates,recorded_guard_fields=recorded,limits=['Globaltypecandidatesarestructuralselection,notsemantics:OTHERorunlinkedsoftmax/TopKnotassignedsamplingbytime/name.','0nativeprobabilityprimitiveorthesame840/1144populationdoesnotproveeveryruntimeguardfalseorall_greedyTrue.','Graph/unlinked/taskswithoutfullcallerkeptseparately,nofabricatedPythonstacks.','NativeAPIcountsnotPythoninvocations;notcausalCPU/device/requesttiming.','Nonewsource/model/torch/NPU/service/healthwork.'])
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(summary=summary,sampler_chain_headers=headers,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
