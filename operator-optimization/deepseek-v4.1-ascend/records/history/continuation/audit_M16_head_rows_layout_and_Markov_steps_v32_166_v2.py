from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,re,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis');E=P/'native-execution-identity-v14';M=P/'native-PMU-complete-v18';OUT=M/'M16-root-index-head-layout-closure-and-current-Markov-step-census-v32-v2.json';assert not OUT.exists()
def pin(p,s):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==s,str(p);return json.loads(b)
d=pin(M/'M16-allrank-vocab-head-reconstruction-and-consumer-origins-v32.json','25cb705966615df6035e468696cd9f8584169aefdd5872846bb7aee63515d490');prior=pin(M/'M15-root-row-norm-snapshot-and-communication-audit-v31.json','c7085f401e95cf0a60daeb52687ffbeaae2e2ffa07b4a9d7d4ee8082f1827f00')
fs=pin(E/'native-compute-fullchains-v14-v2.json','6c7bf4bd3a7dcf557327aeadc4a37ec4b9b23cb1ba675353bbd6e5d56a4e1ca0');ix=pin(E/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4')
def shape(c):return dict(op=c['opType_text'],input=c.get('inputShapes_text'),output=c.get('outputShapes_text'),input_types=c.get('inputDataTypes_text'),output_types=c.get('outputDataTypes_text'))
def rows(c):return int(c['inputShapes_text'].strip('"').split(',')[0])
roles=[]
for r in d['native_TASKs']:
 c=r['definition'];op=c['opType_text'];ctx='|'.join(r['contexts']);role='other'
 if op=='Transpose':role='vocab_reconstruction_layout'
 elif op=='hcom_allGather_':role='vocab_hcom_ComputeInfo'
 elif op=='RmsNorm':role='draft_head_norm'
 elif op=='MatMulV2':role='replicated_Markov_GEMM'if';129280,256'in c['inputShapes_text']else'partitioned_base_GEMM'
 roles.append(dict(database_index=r['database_index'],TASK_rowid=r['TASK']['task_record_rowid'],role=role,context=ctx,prepare_root_candidates=r['prepare_root_candidates'],signature=shape(c)))
root_rows=[]
for root in prior['prepare_root_row_links']:
 key=root['root_key'];nn=[r for r in d['native_TASKs']if r['contexts']==['TARGET_PRIMARY']and r['prepare_root_candidates']==[key]];co=[r for r in d['collectives']if r['contexts']==['TARGET_PRIMARY']and r['prepare_root_candidates']==[key]]
 gemm=[r for r in nn if r['definition']['opType_text']=='MatMulV2'];trans=[r for r in nn if r['definition']['opType_text']=='Transpose']
 if root['selected_logits_rows']is None:assert not gemm and not trans and not co;continue
 assert len(gemm)==len(trans)==len(co)==1
 K=root['selected_logits_rows'];gc=gemm[0]['definition'];tc=trans[0]['definition'];op=co[0]['operation']
 assert gc['inputShapes_text'].strip('"')==f'{K},5120;16160,5120'
 assert tc['inputShapes_text'].strip('"')==f'8,{K},16160;3'and tc['outputShapes_text'].strip('"')==f'{K},8,16160'
 assert op['count']==K*16160 and op['rankSize']==8 and op['dtype']=='BFP16'
 root_rows.append(dict(root_key=key,normalized_rows=root['global_normalized_rows'],selected_logits_rows=K,physical_local_vocab=16160,physical_gathered_vocab=8*16160,GEMM_TASK_rowid=gemm[0]['TASK']['task_record_rowid'],Transpose_TASK_rowid=trans[0]['TASK']['task_record_rowid'],COMM_OP_rowid=op['COMM_OP_rowid'],COMM_count=op['count'],same_input_value_or_alias_proved=False))
apis={};chain_keys=set()
for db in fs['databases']:
 di=db['database_index'];apis[di]={}
 for a in db['API_records']:
  if any(re.search(r'/llm_base_proposer.py\((?:1589|159\d|160\d|161\d)\)',f)for f in a['project_frames']):apis[di][a['api_rowid']]=a;chain_keys.add(a['chain_key'])
post=[];bychain=defaultdict(list)
for ref in ix['databases']:
 di=ref['database_index'];x=pin(Path(ref['path']),ref['sha256']);cc={c['globalTaskId']:c for c in x['COMPUTE_definitions']}
 for t in x['TASK_records']:
  ids=sorted(set(t.get('native_linked_API_rowids',[]))&apis[di].keys())
  if not ids:continue
  source=[apis[di][key]for key in ids];text='\n'.join(f for a in source for f in a['project_frames']);ctx='DUMMY_ANCESTRY'if'_dummy_run'in text else'DRAFT_PROPOSAL_ANCESTRY';z=dict(database_index=di,TASK=t,definition=cc.get(t['globalTaskId']),context=ctx,source_APIs=source);post.append(z)
  for key in{a['chain_key']for a in source}:bychain[key].append(z)
headers=[]
for key in sorted(chain_keys):
 rr=bychain[key];headers.append(dict(chain_key=key,TASKs=len(rr),contexts=dict(Counter(r['context']for r in rr)),op_types=dict(Counter(r['definition']['opType_text']if r['definition']else'NO_DEFINITION'for r in rr)),project_frames=[f['frame']for f in fs['canonical_fullchains'][key]if'/vllm'in f['frame']]))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),head_native_TASKs=len(roles),head_roles_by_context=[dict(context=k[0],role=k[1],TASKs=v)for k,v in Counter((r['context'],r['role'])for r in roles).items()],exact_target_Index_head_transpose_COMM_roots=len(root_rows),target_root_regimes=[dict(normalized_rows=k[0],head_rows=k[1],roots=v)for k,v in Counter((r['normalized_rows'],r['selected_logits_rows'])for r in root_rows).items()],physical_local_vocab=16160,TP=8,physical_reconstructed_vocab=129280,Markov_post_native_APIs=sum(len(v)for v in apis.values()),Markov_post_TASKs=len(post),Markov_post_chains=len(headers),Markov_post_context_types=[dict(context=k[0],op=k[1],TASKs=v)for k,v in Counter((r['context'],r['definition']['opType_text']if r['definition']else'NO_DEFINITION')for r in post).items()],whole_module_closed=False,overall_complete=False)
result=dict(summary=summary,source=dict(head_scope_sha256='25cb705966615df6035e468696cd9f8584169aefdd5872846bb7aee63515d490',M15_root_sha256='c7085f401e95cf0a60daeb52687ffbeaae2e2ffa07b4a9d7d4ee8082f1827f00'),head_role_rows=roles,target_Index_head_layout_COMM_links=root_rows,Markov_post_native_TASKs=post,Markov_post_chain_headers=headers,limits=['Onlyexactroot/nativeflowandtypedshapes;nohistoricaltensorvalue/aliasgenerationinference.','Markovpostselectorisa source-siteband,notcompleteproposerorallactualflagproof.','Currentstepcountsarenotnearest-timeinvocationpairingorsamevalue/cachehitproof.','ViewCopy/Slicefullbasestorage-shapesdonotprovealllogicalbyteswerephysicallycopied.','NoSumoflogicalbytes/cycles/intervalsequalsrequestgain;v20functionaluncertaintyretained.'])
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(summary=summary,Markov_post_chain_headers=headers,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
