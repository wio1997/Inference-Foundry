#!/usr/bin/env python3
"""Bounded, streaming raw TorchOp TLV check; never emit pointers or values."""
import pathlib,json,time,struct,os,hashlib
base=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt')
p=base/'FRAMEWORK/torch.op_range'; inf=json.loads((base/'profiler_info_3.json').read_text());b=p.stat();assert b.st_size==355292880
si=inf['start_info'];ei=inf['end_info'];assert si['syscnt_enable'] and si['freq']==100
# Approximate conversion using profiler fallback; CANN localtime offset may differ.
delta=int(ei['collectionTimeEnd'])-int(ei['MonotonicTimeEnd']);startcnt=int(si['start_cnt']);offset=int(si['start_monotonic'])
def conv(s):return int((s-startcnt)*1000/si['freq'])+offset+delta
T={'reshape':('aten::reshape',1793241329147421150,1793241329147536050),'view':('aten::view',1793241329147483840,1793241329147535770),'linear':('aten::linear',1793241329147679030,1793241329149931610)}
const=struct.Struct('<3q4QB?');assert const.size==58
out={k:[] for k in T};cnt=0;selected=0;complete=True;start=time.monotonic()
with p.open('rb',buffering=1024*1024) as f:
 while True:
  if time.monotonic()-start>25:complete=False;break
  h=f.read(6)
  if not h:break
  if len(h)!=6:raise RuntimeError('truncated top header')
  typ,n=struct.unpack('<HI',h)
  if n<58 or n>4*1024*1024:raise RuntimeError('unexpected record length')
  raw=f.read(58)
  if len(raw)!=58:raise RuntimeError('truncated constant')
  z=const.unpack(raw);ts=conv(z[0]);pid=z[3];tid=z[4];cnt+=1
  if pid!=2128 or tid!=2128 or not any(abs(ts-v[1])<20000000 for v in T.values()):
   f.seek(n-58,1);continue
  data=f.read(n-58);assert len(data)==n-58;selected+=1
  fields={};i=0
  while i<len(data):
   if i+6>len(data):raise RuntimeError('bad field')
   t,l=struct.unpack('<HI',data[i:i+6]);i+=6
   if i+l>len(data):raise RuntimeError('bad field length')
   # Read only op name, tensor metadata; ignore scalar values and all others.
   if t in (2,5):fields[t]=data[i:i+l].decode('utf-8',errors='replace')
   i+=l
  name=fields.get(2,'')
  for k,(target_name,target_ts,target_end) in T.items():
   if name==target_name:
    tensor=fields.get(5);entries=[]
    if tensor:
     for item in tensor.strip().split(')'):
      a=item.split(';')
      if len(a)==8:
       entries.append({'impl':a[0],'ptr':a[1], 'dtype':a[2],'shape':a[4],'stride':a[5]})
    out[k].append({'ts_error_ns':ts-target_ts,'end_error_ns':conv(z[1])-target_end,'field5_present':5 in fields,'parsed':entries})
a=p.stat();assert (a.st_size,a.st_mtime_ns,a.st_ino)==(b.st_size,b.st_mtime_ns,b.st_ino)
# Only accept matching if uniquely close under approximate timestamp conversion.
summary={};picked={}
for k,vals in out.items():
 matches=[v for v in vals if abs(v['ts_error_ns'])<=10000 and abs(v['end_error_ns'])<=10000]
 summary[k]={'candidates_in_20ms':len(vals),'both_endpoints_within_10us':len(matches),'min_abs_timestamp_error_ns':min((abs(v['ts_error_ns']) for v in vals),default=None)}
 if len(matches)==1:
  v=matches[0];summary[k].update(start_error_ns=v['ts_error_ns'],end_error_ns=v['end_error_ns'],field5_present=v['field5_present'],metadata_count=len(v['parsed']),shapes=[x['shape'] for x in v['parsed']],strides=[x['stride'] for x in v['parsed']]);picked[k]=v['parsed']
comparisons={}
print(json.dumps({'scope':'EP3 saved raw op_range fixed reshape, view, linear host targets; record headers streamed, payload only +/-20ms and only name/tensor metadata fields decoded','records_scanned':cnt,'candidate_records_examined':selected,'completed_scan':complete,'targets':summary,'comparisons':comparisons,'limits':'Approximate clock conversion; no pointer values emitted. Equal storage pointer does not prove allocation generation, offset, alias bounds or causal flow. No tensor values or scalar fields emitted.'},indent=2))
