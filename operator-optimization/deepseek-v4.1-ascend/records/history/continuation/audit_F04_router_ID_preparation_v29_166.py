from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import subprocess,hashlib,json,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis');E=P/'native-execution-identity-v14';M=P/'native-PMU-complete-v18'
out=M/'F04-router-ID-preparation-exact-graph-and-native-role-audit-v29.json';assert not out.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
f04=pin(M/'F04-current-norm-cast-dual-output-and-FFN-fanout-v29.json','28656617d1c2900c2b1fdacb390dabed59267575c28d525aa53a0ece7dc8ba5f')
fs=pin(E/'native-compute-fullchains-v14-v2.json','6c7bf4bd3a7dcf557327aeadc4a37ec4b9b23cb1ba675353bbd6e5d56a4e1ca0')
ix=pin(E/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4')
apimap={};chainkeys=set()
for db in fs['databases']:
 di=db['database_index'];apimap[di]={}
 for a in db['API_records']:
  frames=a['project_frames']
  sites=[site for site in [163,176]if any('fused_topk_router.py('+str(site)+')'in f for f in frames)]
  if sites:apimap[di][a['api_rowid']]=dict(API=a,sites=sites);chainkeys.add(a['chain_key'])
native=[]
for ref in ix['databases']:
 di=ref['database_index'];d=pin(Path(ref['path']),ref['sha256']);cc={c['globalTaskId']:c for c in d['COMPUTE_definitions']}
 for t in d['TASK_records']:
  ids=sorted(set(t.get('native_linked_API_rowids',[]))&apimap[di].keys())
  if ids:native.append(dict(database_index=di,TASK=t,definition=cc.get(t['globalTaskId']),sources=[apimap[di][key]for key in ids]))
groups={};spans=[];ordered=[]
for b in f04['graph_FFN_blocks']:
 ts=b['after_norm_compute'];picked=[]
 for typ in ['Cast','Equal','Fill','SelectV2']:
  vs=[v for v in ts if v['definition']['opType_text']==typ];assert len(vs)==1,(b['block_ordinal'],typ,len(vs));picked+=vs
 assert len({v['TASK']['streamId']for v in picked})==1
 assert all(a['TASK']['endNs']<=z['TASK']['startNs']for a,z in zip(picked,picked[1:]))
 start=picked[0]['TASK']['startNs'];end=picked[-1]['TASK']['endNs']
 assert len([v for v in ts if start<=v['TASK']['startNs']and v['TASK']['endNs']<=end])==4
 assert picked[0]['definition']['inputDataTypes_text']=='INT32'and picked[0]['definition']['outputDataTypes_text']=='INT64'
 assert picked[-1]['definition']['outputDataTypes_text']=='INT64'
 key=(b['database_index'],b['graph_ordinal']);groups.setdefault(key,[]).append(b['block_ordinal']);spans.append(end-start)
 ordered.append(dict(database_index=b['database_index'],graph_ordinal=b['graph_ordinal'],block_ordinal=b['block_ordinal'],TASK_rowids=[v['TASK']['task_record_rowid']for v in picked],roles=['ID_cast','minus_one_test','zero_scalar','ID_select'],shape_candidates=[v['definition']for v in picked]))
assert len(groups)==80 and all(sorted(v)==list(range(40))for v in groups.values())
native_shapes=Counter()
for r in native:
 c=r['definition'];sites=tuple(sorted({s for a in r['sources']for s in a['sites']}));native_shapes[(sites,c['opType_text'],c['inputShapes_text'],c['inputDataTypes_text'],c['outputShapes_text'])]+=1
code=r'''from pathlib import Path
import ast,json,hashlib
p=Path('/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py');b=p.read_bytes();assert hashlib.sha256(b).hexdigest()=='101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91';ls=b.decode().splitlines();tree=ast.parse(b.decode());matches=[n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='forward'and n.lineno<=1592<=n.end_lineno];assert len(matches)==1;fn=matches[0];points=set();hits=[]
for n in ast.walk(fn):
 if isinstance(n,ast.Name)and n.id=='input_ids':
  hits.append(dict(line=n.lineno,context=type(n.ctx).__name__));points.update(range(max(fn.lineno-1,n.lineno-3),min(fn.end_lineno,n.lineno+4)))
points.update(range(1586,1600));assert len(points)<120
print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),function_start=fn.lineno,function_end=fn.end_lineno,input_ID_identifier_uses=hits,source=[str(k+1)+': '+ls[k]for k in sorted(points)],scope='Onlyinput_ids identifiercontexts andrecordedlayercall inoneexistingtargetforward;notwholemethodaudit.')))
'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stderr
span=sorted(spans)
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),graph_executions=80,graphs_with_all40_layer_ID_quads=80,exact_ID_quads=len(ordered),quad_TASKs=4*len(ordered),
 quad_observed_span_ns=dict(min=span[0],median=(span[1599]+span[1600])/2,max=span[-1]),
 native_ID_prepare_TASKs=len(native),native_source_fullchains=len(chainkeys),native_shapes=[dict(sites=list(k[0]),opType=k[1],inputShapes=k[2],inputDtypes=k[3],outputShapes=k[4],TASKs=v)for k,v in native_shapes.items()],
 no_model_import_device_service_work=True,overall_complete=False,
 limits=['Pergraphidenticaltypes/lengthsdonotprovethesameinputvalues/pointers/rowmap. Sharingrequirescompletegeneration/commalignment/read-onlyconsumercontract.',
         'Sanitizationisonlyminusone->zero,notallnegativeIDs:visiontokensentinelspreserved.',
         'Gatepreparationcontainscommunication pad/split/conditionalTPchunk;nohoistbeforetheseunlesskey/protocolproved.',
         'SpanandTASKcountsarenotremovablecriticalpathlatency;do notsumacrosslayers/ranks asgain.'])
result=dict(summary=summary,graph_ID_quad_records=ordered,native_ID_prepare_TASKs=native,
 canonical_fullchains={k:fs['canonical_fullchains'][k]for k in chainkeys},target_model_input_ID_source=json.loads(r.stdout))
blob=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();out.write_bytes(blob)
print(json.dumps(dict(summary=summary,target_model_input_ID_source=result['target_model_input_ID_source'],full_result=str(out),result_sha256=hashlib.sha256(blob).hexdigest()),ensure_ascii=False))
