import pathlib,sqlite3,hashlib,json,collections,re
from decimal import Decimal
base=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
 return h.hexdigest()
out=[]
for sp in SPEC:
 p=base/'diagnostic-traces'/sp['rank']/'ASCEND_PROFILER_OUTPUT'/('ascend_pytorch_profiler_%s.db'%sp['ep'])
 try:
  assert not pathlib.Path(str(p)+'-wal').exists(); before=sha(p); db=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON');db.execute('PRAGMA automatic_index=OFF')
  def q(sql,args=()):return [dict(x) for x in db.execute(sql,args)]
  meta=q('SELECT value FROM META_DATA WHERE name=?',('parallel_group_info',)); info=json.loads(meta[0]['value']) if len(meta)==1 else {}
  rankmap=q('SELECT rankId,deviceId FROM RANK_DEVICE_MAP WHERE deviceId=?',(sp['ep'],))
  dtype_enum=q('SELECT * FROM ENUM_HCCL_DATA_TYPE WHERE id=?',(11,))
  selected=[]
  for item in sp['tasks']:
   t=q('SELECT * FROM TASK WHERE deviceId=? AND globalPid=? AND streamId=? AND taskId=? AND startNs=?',(sp['ep'],sp['pid'],int(item['stream']),int(item['task']),int(Decimal(item['start_us'])*1000)))
   if len(t)!=1:
    selected.append({'digest':item['digest'],'task_match_count':len(t)});continue
   conn=t[0]['connectionId']; links=q('SELECT * FROM CONNECTION_IDS WHERE id IN (SELECT id FROM CONNECTION_IDS WHERE connectionId=?) ORDER BY id,connectionId',(conn,)); hostids=sorted({x['id'] for x in links if x['connectionId']==conn});conns=sorted({x['connectionId'] for x in links});host=[];comm=[]
   for hid in hostids:
    host+=q('SELECT p.startNs,p.endNs,p.globalTid,p.connectionId,s.value name FROM PYTORCH_API p LEFT JOIN STRING_IDS s ON p.name=s.id WHERE p.connectionId=?',(hid,))
   for cc in conns:
    comm+=q('SELECT p.*,s.value nameVal,sg.value groupVal FROM COMMUNICATION_OP p LEFT JOIN STRING_IDS s ON p.opName=s.id LEFT JOIN STRING_IDS sg ON p.groupName=sg.id WHERE p.connectionId=?',(cc,))
   for op in comm:
    op['group_metadata']=info.get(op.get('groupVal'))
    op['same_opid_count']=q('SELECT COUNT(*) n FROM COMMUNICATION_OP WHERE opId=?',(op['opId'],))[0]['n']
   selected.append({'digest':item['digest'],'task_match_count':len(t),'task':t[0],'links':links,'hosts':host,'comm_ops':comm})
  db.close();after=sha(p); assert before==after and not pathlib.Path(str(p)+'-wal').exists()
  out.append({'ep':sp['ep'],'db_sha256':before,'rankmap':rankmap,'dtype11_enum':dtype_enum,'items':selected})
 except (sqlite3.DatabaseError,OSError,AssertionError) as e:out.append({'ep':sp['ep'],'error':str(e)})
print(json.dumps({'scope':'eight ranks, two exact tasks each; no cross-rank clock subtraction','ranks':out},indent=2))
