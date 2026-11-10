#!/usr/bin/env python3
"""Inspect only completed traces; bind CP self operations to recorded callers."""
from pathlib import Path
from hashlib import sha256
import json,subprocess,sys
from read_common_frame_caller_logs import REMOTE as CALLERS
ROOT=Path('audit/execution-search');TIMING=Path('audit/framework-timing-171')
TAIL=r'''
from bisect import bisect_left
def merged(rows):
 result=[]
 for a,b in sorted(rows):
  if result and a<=result[-1][1]:result[-1][1]=max(result[-1][1],b)
  elif b>a:result.append([a,b])
 return result
def overlap(rows,a,b):return sum(max(0,min(y,b)-max(x,a)) for x,y in rows if x<b and y>a)
result=[]
for plan,r in zip(spec['ranks'],out):
 p=root/plan['trace']['path'];assert p.resolve().is_relative_to(root)
 raw=p.read_bytes();assert sha256(raw).hexdigest()==plan['trace']['sha256'];doc=json.loads(raw);del raw
 events=doc['traceEvents'] if isinstance(doc,dict) else doc
 pids={e['pid'] for e in events if e.get('ph')=='M' and e.get('name')=='process_name' and e.get('args',{}).get('name')=='Python'}
 cpu=sorted([e for e in events if e.get('ph')=='X' and e.get('dur') and e.get('cat')=='cpu_op' and e.get('pid') in pids and e.get('tid')==e.get('pid') and e.get('name','').startswith('aten::')],key=lambda e:(float(e['ts']),-float(e['dur'])))
 starts=[float(e['ts']) for e in cpu]
 free=merged([[float(e['ts']),float(e['ts'])+float(e['dur'])] for e in events if e.get('ph')=='X' and e.get('name')=='Free' and e.get('dur')])
 clock=r['clock'];origin=clock['origin_us'];offset=clock['offset_us'];scale=1+clock['slope_ppm']/1e6
 def mapped(ns):return origin+(ns/1000-origin-offset)/scale
 cps=sorted([s for s in r['selected'] if s['kind']=='cp' and s['caller']=='runner_build_attention'],key=lambda s:s['wall_ns'])
 seen=Counter();samples=[];phases=[];patterns=Counter()
 for cp in cps:
  a=mapped(cp['wall_ns']);b=mapped(cp['wall_ns']+cp['duration_ns'])
  children=[(mapped(s['wall_ns']),mapped(s['wall_ns']+s['duration_ns'])) for s in r['selected'] if s['kind']!='cp' and s['caller']==cp['caller'] and cp['wall_ns']<=s['wall_ns'] and s['wall_ns']+s['duration_ns']<=cp['wall_ns']+cp['duration_ns']]
  selected=[]
  for e in cpu[bisect_left(starts,a):bisect_left(starts,b)]:
   x=float(e['ts']);y=x+float(e['dur'])
   if y>b or any(v<=x and y<=w for v,w in children):continue
   selected.append(e)
  top=[];end=-1
  for e in selected:
   x=float(e['ts']);y=x+float(e['dur'])
   if y<=end:continue
   top.append(e);end=y
  signature=tuple(e['name'] for e in top);patterns[signature]+=1;seen[cp['parent']]+=1;ordinal=seen[cp['parent']]
  if len(samples)<2:
   samples.append(dict(id=cp['id'],frame=cp['parent'],ordinal=ordinal,ops=[dict(name=e['name'],relative_us=float(e['ts'])-a,duration_us=float(e['dur']),args=e.get('args')) for e in top]))
  clamps=[i for i,e in enumerate(top) if e['name']=='aten::clamp'];wheres=[i for i,e in enumerate(top) if e['name']=='aten::where']
  valid=len(clamps)==len(wheres)==1 and clamps[0]<wheres[0]
  segment=top[clamps[0]:wheres[0]+1] if valid else []
  valid=valid and [e['name'] for e in segment].count('aten::sub')==2 and [e['name'] for e in segment].count('aten::clamp_min')==1 and [e['name'] for e in segment].count('aten::gt')==1
  if valid:
   x=float(segment[0]['ts']);y=float(segment[-1]['ts'])+float(segment[-1]['dur']);guard=clock['required_enclosure_residual_us']+1
   phases.append(dict(id=cp['id'],frame=cp['parent'],ordinal=ordinal,first_retained=ordinal==1,start_us=x,end_us=y,host_envelope_us=y-x,free_overlap_us=overlap(free,x,y),guarded_free_overlap_us=overlap(free,x+guard,y-guard),op_names=[e['name'] for e in segment]))
  else:phases.append(dict(id=cp['id'],frame=cp['parent'],ordinal=ordinal,unresolved=True))
 result.append(dict(ep=r['ep'],window_ms=(plan['window_end_us']-plan['window_start_us'])/1000,trace=plan['trace'],caller_counts=r['counts'],target_frames=len(seen),frame_call_counts=dict(seen),samples=samples,patterns=[dict(calls=n,ops=list(k)) for k,n in patterns.items()],phases=phases,original_FT_body_misses=r['FT_body_unmatched_in_original_analysis'],clock=clock))
 del cpu,starts,events,doc
print(json.dumps(dict(status='READ_ONLY_CP_SELF_OPERATOR_SEQUENCE',ranks=result,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[])))
'''

def main():
 fine=json.loads((TIMING/'function-analysis.json').read_text());inv={r['path']:r for r in json.loads((TIMING/'raw-inventory.json').read_text())};receipt=json.loads((TIMING/'receipt.json').read_text());assert receipt['status']=='CAPTURED_RESTORED'
 wanted=[int(v) for v in sys.argv[1:]] if len(sys.argv)>1 else list(range(16));plans=[]
 for r in fine['ranks']:
  if r['ep'] not in wanted:continue
  trace=next(v for k,v in inv.items() if f'_ep{r["ep"]}_' in k and k.endswith('/trace_view.json'))
  plans.append(dict(ep=r['ep'],clock=r['clock_alignment'],window_start_us=r['window_start_us'],window_end_us=r['window_end_us'],unmatched=r['body_marker_unmatched'],files=[inv['timing/spans-'+str(pid)+'.jsonl'] for pid in r['python_pids']],trace=trace))
 spec=dict(root=receipt['output_host'],ranks=plans)
 remote=CALLERS.rsplit('print(json.dumps',1)[0]+TAIL
 p=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',remote,json.dumps(spec)],capture_output=True,text=True,timeout=120)
 if p.returncode:raise RuntimeError(p.stderr)
 d=json.loads(p.stdout);d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),caller_reader_sha256=sha256(Path('tools/read_common_frame_caller_logs.py').read_bytes()).hexdigest(),inputs=[dict(path=str(x),sha256=sha256(x.read_bytes()).hexdigest()) for x in [TIMING/'function-analysis.json',TIMING/'raw-inventory.json',TIMING/'receipt.json']],qualification='Saved marked diagnostic only. CPU aten envelope/Free intersection is not device-kernel cost or unprofiled request delta; source/pattern qualification is mandatory. Existing clocks/misses retained, no new capture.')
 suffix='all' if wanted==list(range(16)) else '-'.join(map(str,wanted))
 dest=ROOT/f'common-device-trace-{suffix}.json';dest.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(path=str(dest),ranks=len(d['ranks']),calls=sum(len(r['phases']) for r in d['ranks']),unresolved=sum(p.get('unresolved',False) for r in d['ranks'] for p in r['phases']))))
if __name__=='__main__':main()
