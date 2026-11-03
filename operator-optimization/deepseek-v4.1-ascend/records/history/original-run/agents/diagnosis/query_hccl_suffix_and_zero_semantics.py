#!/usr/bin/env python3
"""Read fixed HCCL JSON keys, rank-local timestamps, profiler config and source."""
import argparse,hashlib,json,os,re,sqlite3
from decimal import Decimal,InvalidOperation
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
T=Path(a.task_root);R=Path(a.trace_root);p135=T/'agents/diagnosis/16rank-draft-alltoall-distribution.json';p138=T/'agents/diagnosis/24-draft-alltoall-identity.json';p139=T/'agents/diagnosis/24-draft-hccl-json-details.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
for p,h in ((p135,'633011028e6d5953f03f1493c197085bdd1a810c095cc646e4f5b0e115f61ec6'),(p138,'bc572249e448c5470d3a9c59a08fa6f872c7e9674b78b37e33e926bad1ddf9ef'),(p139,'2d56c4ebb8d83dc43d3cb6be57257c064e71756a3913d8a41ae8fccba895b61c')):
 if sha(p)!=h:raise RuntimeError('input hash mismatch '+p.name)
a=json.loads(p135.read_text());b=json.loads(p138.read_text());j=json.loads(p139.read_text());expected_json={x['rank']:x['sha256'] for x in j['files']};selected={}
for x in b['samples']:
 choices=[]
 for z in a['brackets']:
  if z['rank']!=x['rank'] or z['gap_ordinal']!=x['gap_ordinal'] or z['hc_input_shape']!=x['hc_input_shape']:continue
  for q in z['alltoall']:
   if len(q['exact_communication_op_matches'])==1 and q['exact_communication_op_matches'][0]['opId']==x['opId_local']:choices.append(q)
 if len(choices)!=1:raise RuntimeError('selected operation not unique')
 selected.setdefault(x['rank'],[]).append((x,choices[0]))
if sum(map(len,selected.values()))!=24:raise RuntimeError('expected 24 operations')
results=[]
for rank,items in sorted(selected.items()):
 root=R/rank/'ASCEND_PROFILER_OUTPUT';p=root/'communication.json'
 if sha(p)!=expected_json[rank]:raise RuntimeError('stage139 JSON hash mismatch')
 dbs=list(root.glob('ascend_pytorch_profiler_*.db'))
 if len(dbs)!=1 or sha(dbs[0])!=a['file_sha256_by_rank'][rank]['db']:raise RuntimeError('DB mismatch')
 con=sqlite3.connect(dbs[0].resolve().as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row;con.execute('PRAGMA query_only=ON')
 doc=json.loads(p.read_text());found={x['opId_local']:[] for x,_ in items}
 def walk(obj,depth=0):
  if depth>12:return
  if isinstance(obj,dict):
   for key,value in obj.items():
    if not isinstance(key,str):continue
    prefix,sep,suffix=key.partition('@');ph=hashlib.sha256(prefix.encode()).hexdigest()
    for x,h in items:
     if ph!=x['op_name_sha256']:continue
     arr=found[x['opId_local']]
     if len(arr)>=21:continue
     q=con.execute('SELECT s.value FROM COMMUNICATION_OP c LEFT JOIN STRING_IDS s ON s.id=c.groupName WHERE c.opId=?',(x['opId_local'],)).fetchone();group=q[0] if q else None
     info=value.get('Communication Time Info',{}) if isinstance(value,dict) else {}
     time=info.get('Start Timestamp(us)') if isinstance(info,dict) else None
     try:delta_ns=int(Decimal(str(time))*1000)-h['alltoall_start_ns'] if time is not None else None
     except (InvalidOperation,ValueError,TypeError):delta_ns=None
     numeric=bool(re.fullmatch(r'\d+',suffix)) if sep else False
     ep=int(rank.split('_ep')[1].split('_')[0]);local_rank=int(rank.split('_rank')[1].split('_')[0])
     arr.append({'has_suffix':bool(sep),'suffix_sha256':hashlib.sha256(suffix.encode()).hexdigest() if sep else None,
       'suffix_char_count':len(suffix) if sep else 0,'suffix_is_decimal':numeric,
       'suffix_equals_group':suffix==group if sep and group is not None else None,
       'suffix_equals_ep':suffix==str(ep) if sep else None,'suffix_equals_local_rank':suffix==str(local_rank) if sep else None,
       'suffix_equals_stream':suffix==str(h['stream']) if sep else None,'suffix_equals_local_opid':suffix==str(x['opId_local']) if sep else None,
       'group_contains_suffix':suffix in group if sep and group and suffix else None,
       'suffix_contains_group':group in suffix if sep and group else None,
       'start_timestamp_field_present':time is not None,'json_start_minus_task_start_ns':delta_ns,
       'elapsed_ms':info.get('Elapse Time(ms)') if isinstance(info,dict) and type(info.get('Elapse Time(ms)')) in (int,float) else None})
    if isinstance(value,(dict,list)):walk(value,depth+1)
  elif isinstance(obj,list):
   for v in obj[:2000]:walk(v,depth+1)
 walk(doc);con.close()
 for x,_ in items:results.append({'rank':rank,'opId_local':x['opId_local'],'matches':found[x['opId_local']][:20],'truncated':len(found[x['opId_local']])>20})
info_path=R/'dp0_pp0_tp7_dcp0_ep7_rank7_4869_20261028020238581_ascend_pt/profiler_info_7.json'
if sha(info_path)!='74cde5c6e1ece98eee3b06733d40eee2e5ee300b9d8a4f0c07443b39346a738d':raise RuntimeError('EP7 profiler_info mismatch')
config=[];config_key=re.compile(r'(?i)(profiler.level|aic.metrics|ai.core.metrics|activities|l2.cache|sys.interconnection|op.attr|export.type)')
safe_enum={'Level0','Level1','Level2','level0','level1','level2','PipeUtilization','L0','L1','L2','CPU','NPU','ProfilerActivity.CPU','ProfilerActivity.NPU','Text','Db','DbAndText'}
def scan_config(obj,path=''):
 if isinstance(obj,dict):
  for k,v in obj.items():
   if not isinstance(k,str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}',k):continue
   q=path+'.'+k if path else k
   if config_key.search(k):
    value=v if type(v) in (bool,int) or (isinstance(v,str) and v in safe_enum) else None
    if isinstance(v,list):value=[x for x in v if isinstance(x,str) and x in safe_enum]
    config.append({'path':q,'type':type(v).__name__,'safe_value':value})
   scan_config(v,q)
 elif isinstance(obj,list):
  for i,v in enumerate(obj[:100]):scan_config(v,path+'['+str(i)+']')
scan_config(json.loads(info_path.read_text()))
roots=[Path('/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis'),Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis')]
pat=re.compile(r'Idle Time\(ms\)|Communication Time Info|TRANSIT_TIME_MS|WAIT_TIME_MS',re.I);sources=[];scanned=0;skipped=[]
for root in roots:
 if not root.is_dir():skipped.append(str(root));continue
 for base,dirs,files in os.walk(root,followlinks=False):
  dirs[:]=sorted(d for d in dirs if d not in ('__pycache__','.git'))
  for filename in sorted(files):
   if not filename.endswith(('.py','.sql')):continue
   p=Path(base)/filename
   try:
    raw=p.read_bytes()
    if len(raw)>2_000_000:skipped.append(str(p));continue
    text=raw.decode('utf8');scanned+=1
   except (OSError,UnicodeError):skipped.append(str(p));continue
   lines=text.splitlines();hit=[i for i,s in enumerate(lines) if pat.search(s)]
   if hit:
    if len(sources)>=50:raise RuntimeError('too many source matches')
    ix=sorted({j for i in hit for j in range(max(0,i-2),min(len(lines),i+3))})
    sources.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'lines':[{'n':i+1,'text':lines[i].strip()[:220]} for i in ix[:100]],'truncated':len(ix)>100})
print(json.dumps({'scope':'24 fixed per-rank JSON labels; rank-local timestamp comparison; EP7 config and fixed-term source search',
 'inputs':{p.name:sha(p) for p in (p135,p138,p139)},'json_suffix_results':results,'ep7_profiler_config':config,
 'source_scanned':scanned,'source_skipped':skipped,'source_matches':sources,
 'limits':['Only @ decomposition and simple literal relationships tested; unequal group hash does not imply unequal group.',
 'Rank-local JSON and TASK numeric timestamp agreement is an association, not a verified clock or replay contract.',
 'Profiler config belongs to a session and may omit effective modes; safe whitelist may suppress values.',
 'Source scan only fixed terms in two .py/.sql roots; compiled exporter, docs and alias spellings are not covered.',
 'No prompt/token text or absolute timestamps are output; no cross-rank clock comparison or NPU activity.']},ensure_ascii=False,indent=2))
