#!/usr/bin/env python3
"""Inside stage: fixed per-rank communication JSON entries and database operations."""
import hashlib,json,pathlib,sqlite3,collections
from decimal import Decimal
base=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
res=[]
for r in SPEC:
 ep=r['ep'];folder=base/r['rank']/'ASCEND_PROFILER_OUTPUT';dbp=folder/('ascend_pytorch_profiler_%d.db'%ep);jp=folder/'communication.json'
 assert not pathlib.Path(str(dbp)+'-wal').exists() and sha(dbp)==r['db_sha']
 st=jp.stat(); assert 0<st.st_size<32*1024*1024
 jb=jp.read_bytes();jhash=hashlib.sha256(jb).hexdigest();doc=json.loads(jb,parse_float=Decimal)
 c=sqlite3.connect(dbp.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON');c.execute('PRAGMA automatic_index=OFF')
 records=[]
 def entries(obj,op,depth=0):
  if depth>12:return []
  out=[]
  if isinstance(obj,dict):
   for k,v in obj.items():
    if isinstance(k,str) and k.partition('@')[0]==op:out.append((k,v))
    if isinstance(v,(dict,list)):out.extend(entries(v,op,depth+1))
  elif isinstance(obj,list):
   for v in obj[:2000]:out.extend(entries(v,op,depth+1))
  return out
 for item in r['items']:
  op=c.execute('SELECT * FROM COMMUNICATION_OP WHERE opId=? AND connectionId=?',(item['opid'],item['connection'])).fetchall();assert len(op)==1
  op=dict(op[0]);assert (op['startNs'],op['endNs'])==(item['start'],item['end'])
  found=entries(doc,item['name']);out={'digest':item['digest'],'opid':item['opid'],'op_duration_ns':op['endNs']-op['startNs'],'json_entry_count':len(found)}
  if len(found)==1:
   key,node=found[0];ti=node.get('Communication Time Info',{});band=node.get('Communication Bandwidth Info',{})
   out['json_group_suffix_present']='@' in key
   out['json_time']={k:str(v) for k,v in ti.items() if k in ('Start Timestamp(us)','Elapse Time(ms)','Transit Time(ms)','Wait Time(ms)','Synchronization Time(ms)','Idle Time(ms)')}
   out['json_links']={link:{k:str(v) for k,v in data.items() if k in ('Transit Size(MB)','Transit Time(ms)','Bandwidth(GB/s)')} for link,data in band.items() if link in ('HCCS','SIO','SDMA','RDMA','PCIE','UB') and isinstance(data,dict)}
   if 'Start Timestamp(us)' in ti:out['json_start_minus_op_us']=str(Decimal(str(ti['Start Timestamp(us)']))-Decimal(op['startNs'])/1000)
   if 'Elapse Time(ms)' in ti:out['json_elapse_minus_op_ms']=str(Decimal(str(ti['Elapse Time(ms)']))-Decimal(op['endNs']-op['startNs'])/1000000)
  children=[dict(x) for x in c.execute('SELECT * FROM COMMUNICATION_TASK_INFO WHERE opId=? LIMIT 2001',(item['opid'],))];assert len(children)<=2000
  ids=[x['globalTaskId'] for x in children];out['child_rows']=len(children);out['child_unique_global_ids']=len(set(ids));names=collections.Counter();nonzero=collections.Counter();sizes=collections.Counter();name_cache={}
  for ch in children:
   key=ch['taskType']
   if key not in name_cache:
    n=c.execute('SELECT value FROM STRING_IDS WHERE id=?',(key,)).fetchone();name_cache[key]=n[0] if n else str(key)
   name=name_cache[key];names[name]+=1
   if ch['size']:nonzero[name]+=1;sizes[name]+=ch['size']
  out['child_type_counts']=dict(names);out['child_nonzero_size_counts']=dict(nonzero);out['child_raw_size_sums']=dict(sizes)
  records.append(out)
 c.close(); assert sha(dbp)==r['db_sha'] and not pathlib.Path(str(dbp)+'-wal').exists();assert (st.st_size,st.st_mtime_ns,st.st_ino)==(jp.stat().st_size,jp.stat().st_mtime_ns,jp.stat().st_ino) and hashlib.sha256(jp.read_bytes()).hexdigest()==jhash
 res.append({'ep':ep,'db_sha':r['db_sha'],'communication_json_bytes':st.st_size,'communication_json_sha256_first_observed':jhash,'items':records})
print(json.dumps({'scope':'16 exact rank-local ops; JSON prefix plus start/elapse comparisons; child table local opId','ranks':res,'limits':['JSON prefix and timestamp agreement is a candidate association; group suffix contract and exporter version unbound','durations are export records, not network-only time or AICPU decomposition','raw child sizes and link fields are not necessarily additive physical bytes','no cross-rank clock arithmetic and no device work']},indent=2))
