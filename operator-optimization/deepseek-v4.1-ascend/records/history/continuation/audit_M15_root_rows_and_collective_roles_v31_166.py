from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis');M=P/'native-PMU-complete-v18'
OUT=M/'M15-root-row-norm-snapshot-and-communication-audit-v31.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha,str(p);return json.loads(b)
d=pin(M/'M15-final-HC-norm-row-selection-and-state-scope-v31.json','705e588a6c76c15b292a65c72000103751e1eae1dac673e0ef0ef7a27446fed3')
prep=pin(P/'prepare-field-origins-v12.json','c23b75f3fbd3ae3d3d7152f6882524e899bade53001ea2f1befa09ce943f091f')
roots={r['root_key']:r for r in prep['prepare_roots']}
def frames(r):return '\n'.join(f for a in r['source_APIs']for f in a['project_frames'])
def shape(c):return dict(op=c['opType_text'],input=c.get('inputShapes_text'),output=c.get('outputShapes_text'),input_types=c.get('inputDataTypes_text'),output_types=c.get('outputDataTypes_text'))
def dim(t):return int(t.strip('"').split(';')[0].split(',')[0])
native=[]
for r in d['native_TASKs']:
 ff=frames(r);c=r['definition'];role='other'
 if '/model_runner_v1.py(2634)'in ff:role='target_logits_Index'if c['opType_text']=='Index'else'target_logits_IndexCheck'
 elif '/model.py(1602)'in ff:role='target_final_collapse'
 elif '/model.py(1605)'in ff:role='target_final_norm'
 elif '/model.py(1600)'in ff:role='target_fullHC_snapshot'
 elif '/model.py(1598)'in ff:role='target_premix_gather_ComputeInfo'
 elif '/dspark.py(273)'in ff:role='draft_HC_collapse'
 native.append(dict(database_index=r['database_index'],TASK_rowid=r['TASK']['task_record_rowid'],role=role,prepare_root_candidates=r['prepare_root_candidates'],signature=shape(c)))
comm_headers=defaultdict(lambda:dict(operations=set(),source_kinds=Counter(),frames=[]));comms=[]
for r in d['native_collectives']:
 text='\n'.join(f for x in r['source_candidates']for f in x['API']['project_frames'])
 role='target_fullHC_gather'if'/model.py(1597)'in text else'target_premix_gather'if'/model.py(1598)'in text else'other'
 cc=dict(database_index=r['database_index'],COMM_OP_rowid=r['operation']['COMM_OP_rowid'],role=role,prepare_root_candidates=r['prepare_root_candidates'],family=r['operation']['family'],count=r['operation']['count'],dtype=r['operation']['dtype'],rankSize=r['operation']['rankSize']);comms.append(cc)
 for x in r['source_candidates']:
  a=x['API'];key=(role,a.get('chain_key'));z=comm_headers[key];z['operations'].add((r['database_index'],r['operation']['COMM_OP_rowid']));z['source_kinds'][x['source_group_kind']]+=1;z['frames']=a['project_frames']
segments=[]
for s in d['graph_suffixes']:
 cc=s['compute'];types=[r['definition']['opType_text']for r in cc];assert types==['hcom_allGather_','hcom_allGather_','TensorMove','Cast','Mul','ReduceSum','Cast','RmsNorm'],types
 assert cc[2]['definition']['inputShapes_text'].strip('"')=='24,20480'
 assert cc[-1]['definition']['inputShapes_text'].strip('"')=='24,5120;5120'
 segments.append(dict(database_index=s['database_index'],graph_ordinal=s['graph_ordinal'],prepare_root_keys=s['prepare_root_keys'],phase_records=s['phase_records'],global_rows=24,snapshot_elements=24*20480,norm_rows=24,TASK_rowids=[r['task']['task_record_rowid']for r in cc],COMM_regimes=[dict(dtype=r['dtype'],count=r['count'],rankSize=r['rankSize'],COMM_OP_rowid=r['COMM_OP_rowid'])for r in s['COMM_candidates']]))
rr=[]
for key,root in roots.items():
 nn=[r for r in native if r['prepare_root_candidates']==[key]];ss=[s for s in segments if s['prepare_root_keys']==[key]];co=[r for r in comms if r['prepare_root_candidates']==[key]]
 ix=[r for r in nn if r['role']=='target_logits_Index'];norm=[r for r in nn if r['role']=='target_final_norm'];snap=[r for r in nn if r['role']=='target_fullHC_snapshot']
 assert len(ix)<=1 and len(norm)<=1 and len(snap)<=1 and len(ss)<=1
 N=dim(norm[0]['signature']['output'])if norm else ss[0]['norm_rows']if ss else None
 K=dim(ix[0]['signature']['output'])if ix else None
 if ix:assert N==dim(ix[0]['signature']['input'])
 rr.append(dict(root_key=key,database_index=root['database_index'],schedule=root.get('schedule'),global_normalized_rows=N,selected_logits_rows=K,selected_native_roles=dict(Counter(r['role']for r in nn)),graph_suffixes=len(ss),native_COMM=co,graph_COMM=ss[0]['COMM_regimes']if ss else[],native_index_TASK_rowid=ix[0]['TASK_rowid']if ix else None))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),native_role_counts=dict(Counter(r['role']for r in native)),native_root_ambiguity_by_role={role:dict(Counter(len(r['prepare_root_candidates'])for r in native if r['role']==role))for role in sorted({r['role']for r in native})},native_COMM_roles=dict(Counter(r['role']for r in comms)),native_COMM_root_ambiguity=dict(Counter(len(r['prepare_root_candidates'])for r in comms)),graph_suffixes=len(segments),graph_suffixes_unbound=sum(len(s['prepare_root_keys'])!=1 for s in segments),root_row_regimes=[dict(normalized_rows=k[0],selected_logits_rows=k[1],graph_suffixes=k[2],roots=v)for k,v in Counter((r['global_normalized_rows'],r['selected_logits_rows'],r['graph_suffixes'])for r in rr).items()],roots_with_Index=sum(r['selected_logits_rows']is not None for r in rr),selected_Index_TASKs=sum(r['role']=='target_logits_Index'for r in native),source_consistent_target_final_norms=24+len(segments),source_consistent_target_collapse_TASKs=96+4*len(segments),source_consistent_fullHC_snapshots=24+len(segments),graph_suffix_COMM_regimes=[dict(dtype=k[0],count=k[1],rankSize=k[2],operations=v)for k,v in Counter((r['dtype'],r['count'],r['rankSize'])for s in segments for r in s['COMM_regimes']).items()],whole_module_closed=False,overall_complete=False)
headers=[dict(role=key[0],chain_key=key[1],operations=len(v['operations']),source_kinds=dict(v['source_kinds']),frames=v['frames'])for key,v in comm_headers.items()]
result=dict(summary=summary,source=dict(path=str(M/'M15-final-HC-norm-row-selection-and-state-scope-v31.json'),sha256='705e588a6c76c15b292a65c72000103751e1eae1dac673e0ef0ef7a27446fed3'),native_role_rows=native,prepare_root_row_links=rr,COMM_rows=comms,COMM_source_headers=headers,graph_suffix_role_rows=segments,limits=['Exactroot/typed-shapeconsistencyisnotactualtensoralias/valuegenerationproof.','416drafthelperTASKsarenottargettailandnotassignedtargetrootsbytime.','Sourcegroupnamespaceisretained;Torch/CANNAPIrowidsmustnotbecrossjoinedwithouttableidentity.','24fullHCnativeCOMMrecordsarenotfabricatedas24additionalComputeInfoTASKs:ancestryqueryonlyselected24premixhcomTASKs;completeCOMMfactretainedseparately.','Noactualselectedrowvaluesorall-content-readerclosureisprovedbyK<N.','M15v20functionalPMUbucketisnotautomaticallyreplacedbythisselector.'])
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(summary=summary,COMM_source_headers=headers,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
