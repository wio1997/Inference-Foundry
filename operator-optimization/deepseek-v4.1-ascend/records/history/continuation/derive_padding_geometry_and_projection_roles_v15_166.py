from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import json,hashlib,socket,re,math
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis')
def pin(path,sha):
 b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
idx=pin(P/'native-execution-identity-v14/index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4')
c=pin(P/'native-execution-identity-v14/native-compute-fullchains-v14-v2.json','6c7bf4bd3a7dcf557327aeadc4a37ec4b9b23cb1ba675353bbd6e5d56a4e1ca0')
g=pin(P/'native-execution-identity-v14/graph-blocks-PMU-and-MC2-v14.json','a035c4de08191656141ad0727e8305a6444cc77e6184d4951f86136021af0991')
comm=pin(P/'native-communications-v15/communication-and-queue-review-v15.json','c196d5083904a75a794c81fadfaf9d528d11461fdd5d81a6e941cad1e9693d6c')
inv=pin(P/'profile-capabilities-index-v12-v2.json','79ba556ef3855b9066392e438371a0a9d3b38ea7511269ce7ff2a6814529a095')
OUT=P/'native-communications-v15/padding-geometry-and-projection-roles-v15.json';assert not OUT.exists()
role_frames={'Q_A':'dsa_v41.py(467): multistream_preprocess','KV':'dsa_v41.py(475): multistream_preprocess','Q_NORM':'dsa_v41.py(477): multistream_preprocess','KV_NORM':'dsa_v41.py(487): multistream_preprocess','Q_B':'dsa_v41.py(491): multistream_preprocess','WO_A':'dsa_cp.py(1784): _forward_o_proj','WO_B':'dsa_cp.py(1795): _forward_o_proj','INPUT_NORM':'model.py(896): forward'}
keyroles={}
for key,frames in c['canonical_fullchains'].items():
 texts=[f['frame']for f in frames];matches=[r for r,needle in role_frames.items()if any(needle in f for f in texts)]
 if matches:keyroles[key]=matches
rows=[]
for ref in idx['databases']:
 d=pin(Path(ref['path']),ref['sha256']);di=d['database_index'];amap={a['api_rowid']:a for a in c['databases'][di]['API_records']};defs={v['globalTaskId']:v for v in d['COMPUTE_definitions']}
 for t in d['TASK_records']:
  apis=t.get('native_linked_API_rowids',[]);keys={amap[a]['chain_key']for a in apis};roles=sorted({r for key in keys for r in keyroles.get(key,[])})
  if not roles:continue
  definition=defs[t['globalTaskId']];rows.append(dict(database_index=di,task_record_rowid=t['task_record_rowid'],roles=roles,source_chain_keys=sorted(keys),op_type=definition['opType_text'],input_shapes=definition['inputShapes_text'],output_shapes=definition['outputShapes_text'],input_dtypes=definition['inputDataTypes_text'],output_dtypes=definition['outputDataTypes_text'],blockNum=definition['blockNum'],mixBlockNum=definition['mixBlockNum']))
role_summaries=[]
for role in role_frames:
 rs=[r for r in rows if role in r['roles']];shapes=Counter((r['op_type'],r['input_shapes'],r['output_shapes'],r['input_dtypes'],r['output_dtypes'])for r in rs)
 role_summaries.append(dict(role=role,executions=len(rs),shape_signature_counts=[dict(op_type=k[0],input_shapes=k[1],output_shapes=k[2],input_dtypes=k[3],output_dtypes=k[4],executions=v)for k,v in shapes.items()]))
gcomm={(r['database_index'],r['graph_ordinal']):r for r in comm['graph_communication_partitions']}
geometry=[]
for graph in g['graph_executions']:
 if len(graph['phase_records'])!=1:continue
 r=graph['phase_records'][0];signature=gcomm[(graph['database_index'],graph['graph_ordinal'])]
 geometry.append(dict(database_index=graph['database_index'],graph_ordinal=graph['graph_ordinal'],root_key=graph['prepare_root_keys'][0],ep_rank=r['ep_rank'],scheduled_requests=len(r['scheduled_tokens']),scheduled_tokens=r['total_scheduled_tokens'],captured_target_MoE_static_M=3,TP_group_size=8,derived_target_row_extent=24,operation_sequence_signature_sha256=signature['recorded_sequence_signature_sha256'],potential_TP_divisible_row_extent=math.ceil(r['total_scheduled_tokens']/8)*8))
all_meta=[]
for f in inv['metadata']:
 for m in f['entries']:
  if m['event']!='execute_enter':continue
  scheduled=m['scheduled_tokens'];computed={};cache=m['cached_request_state'];ids=cache.get('req_ids');vals=cache.get('num_computed_tokens')
  if isinstance(ids,list)and isinstance(vals,list)and len(ids)==len(vals):computed.update(zip(ids,vals))
  for req in m['new_request_state']:computed[req['req_id']]=req['num_computed_tokens']
  uniform6=bool(scheduled)and all(v==6 for v in scheduled.values());context_ge=bool(scheduled)and all(isinstance(computed.get(k),int)and computed[k]>=32768 for k in scheduled)
  all_meta.append(dict(label=m['label'],pid=m['pid'],dp=m['data_parallel_rank'],scheduled_requests=len(scheduled),scheduled_tokens=m['total_scheduled_tokens'],uniform6=uniform6,all_CPU_context_ge32768=context_ge,no_new_requests=not m['new_request_state']))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_role_executions=len(rows),source_role_summaries=[dict(role=r['role'],executions=r['executions'],shape_signatures=len(r['shape_signature_counts']))for r in role_summaries],exact_decode_graphs=len(geometry),decode_geometry_hist=[dict(actual_requests=k[0],actual_scheduled_tokens=k[1],captured_physical_row_extent=k[2],TP_divisible_arithmetic_extent=k[3],graphs=v)for k,v in Counter((r['scheduled_requests'],r['scheduled_tokens'],r['derived_target_row_extent'],r['potential_TP_divisible_row_extent'])for r in geometry).items()],distinct_recorded_communication_sequences=len({r['operation_sequence_signature_sha256']for r in geometry}),all_armed_metadata_calls=len(all_meta),all_armed_uniform6_context_ge32768_call_shapes=[dict(scheduled_requests=k[0],scheduled_tokens=k[1],calls=v)for k,v in Counter((m['scheduled_requests'],m['scheduled_tokens'])for m in all_meta if m['uniform6']and m['all_CPU_context_ge32768']and m['no_new_requests']).items()],no_source_DB_or_device_work=True,overall_complete=False,
 limits=['Rowextent24 follows capturedM3×TP8; arithmetic16for12tokensisnotanimplemented/supportedbucketorlatencygain. Currentuniformquery6 constraintneedspreserving/reorganization,notblindrounding.', 'FullARMEDmetadata includesseedingandformalclientactivity; onlyexplicitCPUcontext/scheduleconditioncountsreported,notdevicevalidmaskorphysicalpaddingoutsidecapturedwindows.', 'Source-frameboundrolesarenative eager records; graphdefinitionsandalltensorconsumersremainseparate. Rolecounts canoverlapwhenacallcontainsmorethanoneknownframe.'])
payload=(json.dumps(dict(summary=summary,role_frame_keys=role_frames,source_role_records=rows,source_role_shape_signatures=role_summaries,decode_geometry_records=geometry,all_armed_schedule_records=all_meta),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
public=dict(summary=summary,projection_shapes=[r for r in role_summaries if r['role']in ['Q_A','KV','WO_A','WO_B']],full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest())
if len(json.dumps(public).encode())>30000:public['projection_shapes']=public['projection_shapes'][:2];public['display_truncated']=True
print(json.dumps(public,ensure_ascii=False))
