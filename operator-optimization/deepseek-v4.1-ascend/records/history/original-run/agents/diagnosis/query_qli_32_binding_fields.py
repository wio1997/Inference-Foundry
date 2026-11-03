import pathlib,sqlite3,json,hashlib,collections
R=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
src=R/'agents/diagnosis/f08-attention-intervals.json';assert sha(src)=='355cd1a831403c1129ab76e128c0803da39b05712c3304c5c285aa82e6744e6c'
x=json.loads(src.read_text()); out=[]
expected={3:'084b8718888e33271435ec79d7573c95ca1615f3e7484c807ea24590fad234c4',8:'e07ee2428be27c6129afd6dd0be8b700830048f30f13411b701e97c5d88e2486'}
for ep in (3,8):
 ps=list((R/'diagnostic-20261029/runtime/diagnostic-traces').glob(f'*ep{ep}_rank*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_{ep}.db')); assert len(ps)==1 and sha(ps[0])==expected[ep]
 c=sqlite3.connect(ps[0].as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('pragma query_only=on')
 for p in [v for v in x['indexer_attention_pairs'] if v['ep']==ep]:
  a=c.execute('select * from TASK where startNs=?',(int(p['indexer_flow']),)).fetchall();assert len(a)==1
  t=dict(a[0]);inf=c.execute('select * from COMPUTE_TASK_INFO where globalTaskId=?',(t['globalTaskId'],)).fetchall();assert len(inf)==1
  i=dict(inf[0]); strs={}
  for k in ('name','opType','attrInfo'):
   z=c.execute('select value from STRING_IDS where id=?',(i[k],)).fetchone();strs[k]=z[0] if z else None
  out.append({'ep':ep,'layer':p['layer'],'scope_id':p['scope_id'],'modelId':t['modelId'],'streamId':t['streamId'],'taskId':t['taskId'],'globalTaskId':t['globalTaskId'],'global_id_task_count':c.execute('select count(*) from TASK where globalTaskId=?',(t['globalTaskId'],)).fetchone()[0],'name':strs['name'],'opType':strs['opType'],'attrInfo':strs['attrInfo'],'blockNum':i['blockNum'],'mixBlockNum':i['mixBlockNum'],'gridDim':i['gridDim'],'blockDim':i['blockDim']})
 c.close()
print(json.dumps({'input_sha256':sha(src),'db_sha256':expected,'results':out,'limitations':['Only selected tasks and DB-exported fields, not runtime binary SHA, source, tiling key or per-row lengths. attr_N index semantics is unverified. globalTaskId repeat alone not graph semantics.','No NPU work.']},ensure_ascii=False,indent=2))
