"""Stream one already-exported Chrome trace to inspect graph execute args.

Raw 169MB JSON never enters the model context or memory as a single object.
The result is bounded and does not infer ownership from timestamp proximity.
"""
from pathlib import Path
from decimal import Decimal
import csv,json
T=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
ROOT=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces')
O=T/'graph-trace-argument-inspection';O.mkdir(exist_ok=False)
rows=list(csv.DictReader((T/'graph-submit-candidate-analysis/rank_layer0_submit_candidates.csv').open()))
chosen=[r for r in rows if '_ep0_' in r['rank'] and r['target_forward_ordinal']=='7' and r['candidate_count_in_2ms']=='1'];assert len(chosen)==1
r=chosen[0];candidate_ns=int(r['candidate_submit_start_ns']);p=ROOT/r['rank']/'ASCEND_PROFILER_OUTPUT/trace_view.json'
decoder=json.JSONDecoder();chunk=1024*1024;maximum_record=4*1024*1024;events=0;matching=0;near=[];samples=[]
with p.open('r',encoding='utf-8') as f:
 buf=f.read(chunk);pos=0;started=False;eof=False
 while True:
  while pos<len(buf) and (buf[pos].isspace() or buf[pos]==','):pos+=1
  if not started:
   assert pos<len(buf) and buf[pos]=='[';pos+=1;started=True;continue
  if pos>=len(buf):
   if eof:raise RuntimeError('Unexpected EOF in trace array')
   buf=f.read(chunk);pos=0;eof=not buf;continue
  if buf[pos]==']':break
  try:obj,end=decoder.raw_decode(buf,pos)
  except json.JSONDecodeError:
   remainder=buf[pos:];more=f.read(chunk)
   if not more:raise
   assert len(remainder)+len(more)<=maximum_record,'One trace item exceeds bounded parser memory'
   buf=remainder+more;pos=0;continue
  pos=end;events+=1
  if not isinstance(obj,dict):continue
  if 'aclmdlRIExecuteAsync' not in str(obj.get('name','')):continue
  matching+=1
  item={k:obj.get(k) for k in ['name','cat','ph','pid','tid','ts','dur','args'] if k in obj}
  if len(samples)<3:samples.append(item)
  try:event_ns=int(Decimal(str(obj.get('ts')))*1000)
  except Exception:continue
  if abs(event_ns-candidate_ns)<=50_000 and len(near)<10:near.append(item)
doc={'status':'STREAMED_ONE_TRACE','trace':str(p),'trace_bytes':p.stat().st_size,'events_scanned':events,
     'graph_execute_named_events':matching,'target_candidate_start_ns':candidate_ns,'candidate_nearby_events':near,
     'first_three_matching_events':samples,'record_buffer_limit_bytes':maximum_record,
     'interpretation':'Even a nearby event is a timestamp association until its arguments identify model/replay/request; absence of such an argument remains UNKNOWN.',
     'no_device_work':True}
assert len(json.dumps(doc).encode())<32768
(O/'summary.json').write_text(json.dumps(doc,indent=2)+'\n');print(json.dumps(doc,indent=2))
