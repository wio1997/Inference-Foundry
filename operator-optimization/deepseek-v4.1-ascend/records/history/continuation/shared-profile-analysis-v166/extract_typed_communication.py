"""Correct enum decoding and select actual communication call-chain neighborhoods."""
import collections,concurrent.futures,json,sqlite3
from pathlib import Path
H=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/builtin-v164-official-db')
def one(db):
 c=sqlite3.connect('file:'+str(db)+'?mode=ro&immutable=1',uri=True);c.row_factory=sqlite3.Row
 strings=dict(c.execute('select id,value from STRING_IDS'));s=lambda i:strings.get(i,str(i))
 types=dict(c.execute('select id,name from ENUM_HCCL_DATA_TYPE'))
 rows=[];example_keys=set();anchors=[]
 for r in c.execute('select * from COMMUNICATION_OP'):
  rows.append({'op':s(r['opName']),'local_group_alias':s(r['groupName']),'op_type':s(r['opType']),
   'op_id':r['opId'],'count':r['count'],'data_type_id':r['dataType'],'data_type':types.get(r['dataType'],'UNKNOWN_ENUM'),
   'rank_size':r['rankSize'],'device':r['deviceId'],'start_ns':r['startNs'],'end_ns':r['endNs'],
   'connection_id':r['connectionId'],'alg_type_raw':r['algType'],'alg_type_string_if_present':strings.get(r['algType'])})
 # One query per profile,never one per potential pair or kernel instance.
 flows=collections.defaultdict(list)
 for r in c.execute('select p.rowid,p.name,p.callchainId,x.connectionId native_connection from PYTORCH_API p join CONNECTION_IDS x on p.connectionId=x.id where p.type=50001'):
  flows[r['native_connection']].append(dict(r))
 for r in rows:
  if r['op_type'] not in ['hcom_allGather_','hcom_reduceScatter_']:continue
  key=(r['op_type'],r['count'],r['data_type'])
  if key in example_keys:continue
  links=flows.get(r['connection_id'],[])
  if len(links)!=1:continue
  example_keys.add(key);p=links[0]
  stack=[s(z[0]) for z in c.execute('select stack from PYTORCH_CALLCHAINS where id=? order by stackDepth',(p['callchainId'],))]
  anchors.append({'collective':r,'CPU_rowid':p['rowid'],'CPU_op':s(p['name']),'callchain_id':p['callchainId'],'stack':stack})
 task_schemas={name:[r[1] for r in c.execute('pragma table_info('+name+')')] for name in ['COMMUNICATION_OP','COMMUNICATION_TASK_INFO']}
 c.close()
 return {'profile':db.parent.parent.name,'HCCL_data_type_enum':types,'collectives':rows,'unique_exact_CPU_anchors':anchors,
         'schemas':task_schemas,'group_alias_scope':'Local exporter alias;not a globally attested communicator incarnation.'}
if __name__=='__main__':
 dbs=sorted((H/'profiles').glob('*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler*.db'));assert len(dbs)==32
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,dbs))
 out={'status':'ALL32_COMMUNICATION_TYPED_ENUM_AND_CPU_NEIGHBORHOODS','profiles':rows,
      'correction':'v165 MAJOR-OP-FOCUS used STRING_IDS for COMMUNICATION_OP.dataType;that projection is invalid. Retain frozen result;this derived view uses ENUM_HCCL_DATA_TYPE. v165 task rankings/input-shape types are unaffected.',
      'limits':['API count is not physical traffic.','Exporter group aliases are not cross-process communicator identity.','Exact CPU connection/callchain does not prove all readers or captured replay operand identities.'],
      'new_independence_proofs':0,'overall_complete':False}
 (H/'TYPED-COMMUNICATION-v166.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'status':out['status'],'profiles':len(rows),'collectives':sum(len(r['collectives']) for r in rows),'CPU_anchors':sum(len(r['unique_exact_CPU_anchors']) for r in rows)}))
