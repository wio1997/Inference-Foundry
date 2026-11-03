#!/usr/bin/env python3
"""Two predetermined EP7 draft brackets: shape and exact profiler links only."""
import argparse,bisect,csv,hashlib,json,re,sqlite3
from collections import Counter,defaultdict
from decimal import Decimal
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--db',required=True);ap.add_argument('--csv',required=True);a=ap.parse_args()
T=Path(a.task_root);db=Path(a.db).resolve(strict=True);cp=Path(a.csv).resolve(strict=True)
rank='dp0_pp0_tp7_dcp0_ep7_rank7_4869_20261028020238581_ascend_pt'
mapping=T/'captured-layer-attribution/module-call-metrics.json';strict=T/'eager-draft-mainstream-attribution-v2/draft_call_metrics.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
expected={db:'8b8e72a61e2521f0d91050bc53ab852ddf26dd4e40e2d5ec5d3973652fd807c6',cp:'9aa04e4cbbe7edc2160befcec0b133fdaf3b3e2571a1d2f1d7c92c6998b9bdc1',mapping:'d98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a',strict:'0e6f0cb4d810fdd9f7ec99f7b9a7dbe2123a7db7e5a927115e721b8fd8982c48'}
for p,h in expected.items():
 if sha(p)!=h:raise RuntimeError('source hash mismatch: '+p.name)
rows=[];markers=defaultdict(list)
with cp.open(newline='') as f:
 for r in csv.DictReader(f):
  r['_s']=int(Decimal(r['Start Time(us)'].strip())*1000);r['_e']=r['_s']+int(Decimal(r['Duration(us)'].strip())*1000);rows.append(r)
  if r['Type'] in ('HcPre','HcPost'):markers[(r['Model ID'],r['Stream ID'])].append(r)
for v in markers.values():v.sort(key=lambda x:x['_s'])
mods=[x for x in json.loads(mapping.read_text()) if x['rank']==rank and x['run_in_model'] in (1,2,3)]
def endpoint(run,layer,key):
 x=[m for m in mods if m['run_in_model']==run and m['layer']==layer]
 if len(x)!=1:raise RuntimeError('target-layer anchor not unique')
 return x[0][key]
gaps=[(endpoint(i,39,'window_end_ns'),endpoint(i+1,0,'window_start_ns')) for i in (1,2)]
missing=None
for r in rows:  # same deterministic first-match rule as inspect_draft_missing_attention_v3.py
 if r['Type']!='DispatchFFNCombine' or r['Model ID']!='4294967295' or not any(a<=r['_s'] and r['_e']<=b for a,b in gaps):continue
 shape=r['Input Shapes'].strip('"').split(';')
 try:valid=int(shape[1].split(',')[0])==8 and int(shape[3].split(',')[-1])==3
 except (IndexError,ValueError):valid=False
 if not valid:continue
 m=markers[(r['Model ID'],r['Stream ID'])];i=bisect.bisect_right([x['_s'] for x in m],r['_s'])
 if i<3 or i>=len(m):continue
 x,y,z,v=m[i-3:i+1]
 if [q['Type'] for q in (x,y,z,v)]!=['HcPre','HcPost','HcPre','HcPost']:continue
 inside=[q for q in rows if q['Model ID']==r['Model ID'] and x['_s']<=q['_s'] and q['_e']<=y['_e']]
 if any(q['Type']=='SparseFlashMla' for q in inside):continue
 main=Counter(q['Type'] for q in inside if q['Stream ID']==r['Stream ID']);allc=Counter(q['Type'] for q in inside)
 accepted=all(main.get(t,0)>0 for t in ('ZerosLike','ConcatD')) and all(allc.get(t,0)>0 for t in ('hcom_allGather_','hcom_reduceScatter_')) and not any(main.get(t,0)>0 for t in ('MatMulV2','QuantBatchMatmulV3','InplacePartialRotaryMul'))
 if accepted:continue
 missing={'label':'v3_first_unclassified_no_exported_smla','start_ns':x['_s'],'attention_end_ns':y['_e'],'block_end_ns':v['_e'],'model':r['Model ID'],'stream':r['Stream ID'],'mc2_shape':r['Input Shapes']};break
if not missing or missing['attention_end_ns']-missing['start_ns']!=3204156:raise RuntimeError('EP7 v3 representative did not reproduce')
sc=sorted([x for x in json.loads(strict.read_text()) if x['rank']==rank],key=lambda x:x['start_ns'])
if not sc:raise RuntimeError('no strict EP7 block')
s=sc[0];mark=[x for x in markers[(s['model_id'],s['main_stream_id'])] if s['start_ns']<=x['_s'] and x['_e']<=s['end_ns']]
if len(mark)!=4 or [x['Type'] for x in mark]!=['HcPre','HcPost','HcPre','HcPost']:raise RuntimeError('strict block markers not unique')
present={'label':'first_strict_recorded_smla','start_ns':s['start_ns'],'attention_end_ns':mark[1]['_e'],'block_end_ns':s['end_ns'],'model':s['model_id'],'stream':s['main_stream_id'],'mc2_task_id':s['mc2_task_id']}
con=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row;con.execute('PRAGMA query_only=ON');con.execute('PRAGMA temp_store=MEMORY')
def limited(sql,args=(),cap=20):
 arr=[dict(x) for x in con.execute(sql,args).fetchmany(cap+1)]
 return {'rows':arr[:cap],'truncated':len(arr)>cap,'limit':cap}
opnames={'HcPre','HcPost','RmsNorm','MatMulV2','QuantBatchMatmulV3','DynamicQuant','InplacePartialRotaryMul','ScatterNdUpdateSk','SparseFlashMla','ZerosLike','ConcatD'}
safe_torch={'aten::slice','aten::index','aten::index_select','aten::zeros_like','aten::matmul','aten::mm','aten::linear','aten::rms_norm','aten::as_strided'}
shape_re=re.compile(r'^[0-9,; \[\]()?\-]{0,500}$')
def safe_api(row):
 name=row.pop('api_name',None)
 row['api_name']=name if name in safe_torch else None
 row['other_api_name_sha256']=None if name in safe_torch else hashlib.sha256(str(name).encode()).hexdigest()
 shape=row.pop('input_shape_text',None)
 row['input_shape_text']=shape if isinstance(shape,str) and shape_re.fullmatch(shape) else None
 return row
outputs=[]
for w in (missing,present):
 region=[r for r in rows if r['Model ID']==w['model'] and w['start_ns']<=r['_s'] and r['_e']<=w['attention_end_ns'] and r['Type'] in opnames]
 if len(region)>80:raise RuntimeError('bounded region too large')
 if w is present and not any(r['Type']=='SparseFlashMla' for r in region):raise RuntimeError('strict example has no selected attention')
 events=[]
 for r in region:
  target={'type':r['Type'],'stream':r['Stream ID'],'model':r['Model ID'],'task':r['Task ID'],
          'start_ns':r['_s'],'end_ns':r['_e'],'input_shapes':r['Input Shapes'],'input_dtypes':r['Input Data Types'],
          'output_shapes':r['Output Shapes'],'output_dtypes':r['Output Data Types']}
  tasks=limited('SELECT * FROM TASK WHERE modelId=? AND streamId=? AND taskId=? AND startNs=? AND endNs=?',
   (int(r['Model ID']),int(r['Stream ID']),int(r['Task ID']),r['_s'],r['_e']),2) if r['Stream ID'].isdigit() and r['Task ID'].isdigit() else {'rows':[],'truncated':False,'limit':2}
  links=[]
  for task in tasks['rows']:
   conn=task['connectionId'];sets=limited('SELECT id,connectionId FROM CONNECTION_IDS WHERE connectionId=?',(conn,),20)
   cann=limited('SELECT connectionId,startNs,endNs FROM CANN_API WHERE connectionId=?',(conn,),20)
   apis=[]
   for group in sets['rows']:
    result=limited('''SELECT a.connectionId,a.sequenceNumber,a.callchainId,n.value AS api_name,sh.value AS input_shape_text
      FROM PYTORCH_API a LEFT JOIN STRING_IDS n ON n.id=a.name LEFT JOIN STRING_IDS sh ON sh.id=a.inputShapes
      WHERE a.connectionId=?''',(group['id'],),20)
    result['rows']=[safe_api(x) for x in result['rows']];apis.append({'connection_set_id':group['id'],'matches':result})
   links.append({'connection_id':conn,'cann_exact':cann,'connection_sets':sets,'pytorch_via_sets':apis})
  events.append({'target':target,'exact_task_matches':tasks,'links':links})
 outputs.append({'window':w,'selected_events':events,'selected_events_complete':True})
print(json.dumps({'scope':'EP7 deterministic v3 no-exported-attention and earliest previously strict attention draft brackets',
 'source_sha256':{p.name:h for p,h in expected.items()},'outputs':outputs,
 'limitations':['This selects representative same-rank samples, not matched request, stage, work or shape.',
 'The no-attention bracket is based on exported SparseFlashMla rows only; other paths or capture gaps remain possible.',
 'Tensor shape is not the value of query_metadata.swa.num_actual_tokens; no claim of empty query or phase from shape alone.',
 'Exact connection mappings are candidate links subject to documented profiler semantics; graph/replay reuse can prevent unique attribution.',
 'No host-device time subtraction, cross-rank timing, request text or token contents are returned.']},ensure_ascii=False,indent=2))
con.close()
