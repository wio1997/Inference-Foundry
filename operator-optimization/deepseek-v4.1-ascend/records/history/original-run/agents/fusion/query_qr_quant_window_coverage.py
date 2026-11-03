#!/usr/bin/env python3
"""Read-only quant-shape/order inventory in frozen target windows; no tensor binding."""
import argparse,csv,hashlib,json,statistics
from pathlib import Path
from collections import Counter,defaultdict
from decimal import Decimal
p=argparse.ArgumentParser();p.add_argument('--task-root',required=True);p.add_argument('--trace-root',required=True);a=p.parse_args();T=Path(a.task_root);R=Path(a.trace_root)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def get(rel,checksum):
 p=T/rel
 if sha(p)!=checksum:raise ValueError('input hash '+rel)
 return json.loads(p.read_text())
wins=get('captured-layer-attribution/module-call-metrics.json','d98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a')
manifest=get('evidence/TRACE-EXPORT-SHA256.json','45ac6c7d22cc20601df2592ce0d78406df24a612f4aa56045566354d9ff46139')
expected={x['rank']:x['kernel_csv_sha256'] for x in manifest['exports']};windows=defaultdict(list)
for w in wins:
 if w['run_in_model'] in (1,2,3):windows[w['rank']].append(w)
if len(expected)!=16 or set(expected)!=set(windows) or any(len(x)!=120 for x in windows.values()):raise ValueError('windows')
records=defaultdict(list);actual={};special={2,8,14,20,24,28,32,36};issues=[]
for rank in sorted(expected):
 path=R/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv';actual[rank]=sha(path)
 if actual[rank]!=expected[rank]:raise ValueError('csv hash')
 ws=windows[rank];streams={str(w['main_stream_id']) for w in ws}
 if len(streams)!=1 or any(str(w['model_id'])!='49' for w in ws):raise ValueError('model/stream')
 with path.open(newline='') as f:
  reader=csv.DictReader(f)
  for z in reader:
   if z['Model ID']!='49' or z['Stream ID'] not in streams or z['Type'].strip() not in ('DynamicQuant','QuantBatchMatmulV3'):continue
   st=Decimal(z['Start Time(us)'])*1000; dur=Decimal(z['Duration(us)'])*1000
   if not(st.is_finite() and dur.is_finite() and dur>0 and st==st.to_integral_value() and dur==dur.to_integral_value()):issues.append('bad time');continue
   st=int(st);en=st+int(dur);hits=[w for w in ws if w['window_start_ns']<=st and en<=w['window_end_ns']]
   if len(hits)>1:raise ValueError('overlap')
   if not hits:continue
   w=hits[0];ins=z['Input Shapes'].strip('"');outs=z['Output Shapes'].strip('"');idt=z['Input Data Types'];odt=z['Output Data Types'];typ=z['Type'].strip();kind=None
   if typ=='DynamicQuant' and ins=='3,1280' and outs=='3,1280;3' and idt=='DT_BF16' and odt=='INT8;FLOAT':kind='D'
   elif typ=='QuantBatchMatmulV3' and ins.startswith('3,1280;') and outs in ('3,32768','3,4096') and idt.startswith('INT8;INT8;') and odt=='DT_BF16':kind='Q' if outs=='3,32768' else 'I'
   if kind:records[(rank,w['run_in_model'],w['layer'])].append((st,en,kind))
counts={};patterns=Counter();duration=[];dur_q1=[];dur_q2=[];samples=[]
for rank,ws in windows.items():
 for w in ws:
  key=(rank,w['run_in_model'],w['layer']);seq=sorted(records[key]);c=Counter(x[2] for x in seq);group='special8' if w['layer'] in special else 'other32';counts.setdefault(group,Counter())[f'D{c["D"]}-Q{c["Q"]}-I{c["I"]}']+=1
  if group=='special8':
   pat=''.join(x[2] for x in seq);patterns[pat]+=1
   if pat=='DQDI' and all(seq[i][1]<=seq[i+1][0] for i in range(3)):
    duration.append((seq[2][1]-seq[2][0])/1000);dur_q1.append((seq[0][1]-seq[0][0])/1000);dur_q2.append((seq[2][0]-seq[1][1])/1000)
    if len(samples)<2:samples.append({'rank':rank,'run':w['run_in_model'],'layer':w['layer'],'relative_ns':[[s-w['window_start_ns'],e-w['window_start_ns'],k] for s,e,k in seq]})
  
def stats(x):return {'n':len(x),'min':min(x),'median':statistics.median(x),'max':max(x)} if x else {'n':0}
out={'status':'COMPLETE' if not issues else 'INCOMPLETE','scope':'16 ranks x 3 sampled runs x 40 attributed target layer windows, fully-contained exported model49 main-stream rows only','special8_layers':sorted(special),'input_sha256':{'windows':'d98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a','manifest':'45ac6c7d22cc20601df2592ce0d78406df24a612f4aa56045566354d9ff46139','kernel_csv_by_rank':actual},'window_shape_count_distributions':{k:dict(v) for k,v in counts.items()},'special8_filtered_sequences':dict(patterns),'DQDI_second_quant_duration_us':stats(duration),'DQDI_first_quant_duration_us':stats(dur_q1),'DQDI_first_matmul_end_to_second_quant_start_us':stats(dur_q2),'examples':samples,'issues':issues,'limitations':['Only selected matching signatures, not complete task adjacency; unknown other operators may occur between.','Shape and temporal order are not tensor/storage identity, request/consumer linkage, historical source or binary binding.','Sample timings under profiler, not production latency or estimated benefit.','Excludes other streams, noncontained/unexported tasks and unknown rank/request relationships.']}
print(json.dumps(out,ensure_ascii=False,indent=2))
