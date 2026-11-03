from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,socket

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis')
E=P/'native-execution-identity-v14';M=P/'native-PMU-complete-v18'
OUT=M/'F04-current-norm-cast-dual-output-and-FFN-fanout-v29.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha,str(p);return json.loads(b)
fs=pin(E/'native-compute-fullchains-v14-v2.json','6c7bf4bd3a7dcf557327aeadc4a37ec4b9b23cb1ba675353bbd6e5d56a4e1ca0')
ix=pin(E/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4')
gr=pin(E/'graph-blocks-PMU-and-MC2-v14.json','a035c4de08191656141ad0727e8305a6444cc77e6184d4951f86136021af0991')
apis={d['database_index']:{a['api_rowid']:a for a in d['API_records']}for d in fs['databases']}
bydb=defaultdict(list)
for g in gr['graph_executions']:bydb[g['database_index']].append(g)
norms=[];native_chains=set();blocks=[];gaps=[]
for ref in ix['databases']:
 di=ref['database_index'];d=pin(Path(ref['path']),ref['sha256']);defs={c['globalTaskId']:c for c in d['COMPUTE_definitions']};tm={t['task_record_rowid']:t for t in d['TASK_records']}
 def wrap(t):return dict(TASK=t,definition=defs.get(t['globalTaskId']))
 for t in d['TASK_records']:
  c=defs.get(t['globalTaskId'])
  if not c or c['opType_text']!='RmsNormCast':continue
  aa=[apis[di][key]for key in t.get('native_linked_API_rowids',[])if key in apis[di]]
  for a in aa:native_chains.add(a['chain_key'])
  norms.append(dict(database_index=di,**wrap(t),native_APIs=aa,source_link_state='EXACT_NATIVE_API'if aa else'NO_NATIVE_API_LINK'))
 for g in bydb[di]:
  for block in g['blocks']:
   tasks=sorted([tm[key]for key in block['task_record_rowids']],key=lambda t:(t['startNs'],t['task_record_rowid']))
   cc=[wrap(t)for t in tasks if t['globalTaskId']in defs]
   nc=[v for v in cc if v['definition']['opType_text']=='RmsNormCast']
   if len(nc)!=1:
    gaps.append(dict(database_index=di,graph_ordinal=g['graph_ordinal'],block_ordinal=block['block_ordinal'],norm_candidates=len(nc)))
    continue
   n=nc[0];after=[v for v in cc if n['TASK']['endNs']<=v['TASK']['startNs']]
   blocks.append(dict(database_index=di,graph_ordinal=g['graph_ordinal'],block_ordinal=block['block_ordinal'],prepare_root_keys=g['prepare_root_keys'],phase_records=g['phase_records'],
      norm_cast=n,after_norm_compute=after,
      role_evidence='TypedstructuralFFNsuccessors insideexactHcmarkerblock;notpointer-dataflowproof.'))
shapes=Counter();states=Counter()
for r in norms:
 c=r['definition'];shapes[(r['source_link_state'],c['inputShapes_text'],c['inputDataTypes_text'],c['outputShapes_text'],c['outputDataTypes_text'])]+=1;states[r['source_link_state']]+=1
chainrows=[]
for key in sorted(native_chains):
 rs=[r for r in norms if any(a['chain_key']==key for a in r['native_APIs'])]
 chainrows.append(dict(chain_key=key,TASKs=len(rs),database_indices=sorted({r['database_index']for r in rs}),
    project_frames=[f['frame']for f in fs['canonical_fullchains'][key]if'/vllm'in f['frame']],
    shape_regimes=[dict(inputShapes=k[0],outputDtypes=k[1],TASKs=v)for k,v in Counter((r['definition']['inputShapes_text'],r['definition']['outputDataTypes_text'])for r in rs).items()]))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),RmsNormCast_TASKs=len(norms),source_link_states=dict(states),native_fullchains=len(chainrows),
 shape_regimes=[dict(source_link_state=k[0],inputShapes=k[1],inputDtypes=k[2],outputShapes=k[3],outputDtypes=k[4],TASKs=v)for k,v in shapes.items()],
 graph_executions=len(gr['graph_executions']),graph_blocks_reviewed=len(blocks),missing_or_ambiguous_norm_blocks=gaps,
 after_norm_optypes=dict(Counter(v['definition']['opType_text']for b in blocks for v in b['after_norm_compute'])),
 no_new_source_DB_model_device_service_work=True,overall_complete=False,
 limits=['ExactTASKexecutionkeys andfullnativecallers retained;definitionsarestaticandreused.',
         'Graphnormsuccessors aretyped/marker/ordercandidates,not inferredtensorpointers orPythonstacks.',
         'BF16/FLOAT dualoutputdoesnotprove howrounding/normalizationiscalculated;sourcecontractneededonlyforthatquestion.',
         'No current existingfusion isreportedasa newlyimplementedoptimization;no duration-sum gain.'])
result=dict(summary=summary,norm_TASKs=norms,native_chains=chainrows,graph_FFN_blocks=blocks)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
sample=blocks[0]if blocks else None
print(json.dumps(dict(summary=summary,native_chains=chainrows,first_FFN_block=None if not sample else dict(database_index=sample['database_index'],graph_ordinal=sample['graph_ordinal'],block_ordinal=sample['block_ordinal'],norm=sample['norm_cast'],
 successors=[dict(TASK_rowid=v['TASK']['task_record_rowid'],stream=v['TASK']['streamId'],startNs=v['TASK']['startNs'],endNs=v['TASK']['endNs'],opType=v['definition']['opType_text'],inputShapes=v['definition']['inputShapes_text'],inputDtypes=v['definition']['inputDataTypes_text'],outputShapes=v['definition']['outputShapes_text'],outputDtypes=v['definition']['outputDataTypes_text'])for v in sample['after_norm_compute']]),full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
