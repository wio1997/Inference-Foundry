#!/usr/bin/env python3
"""Price the final target CP builder using existing pinned diagnostic files only."""
from pathlib import Path
from hashlib import sha256
import json, subprocess
from read_common_frame_caller_logs import REMOTE as CALLERS
ROOT=Path('audit/execution-search'); TIMING=Path('audit/framework-timing-171')
TAIL=r'''
import bisect
def merge(xs):
 out=[]
 for a,b in sorted(xs):
  if b<=a:continue
  if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
  else:out.append([a,b])
 return out
def inter(a,b):
 out=[];i=j=0
 while i<len(a) and j<len(b):
  x=max(a[i][0],b[j][0]);y=min(a[i][1],b[j][1])
  if y>x:out.append([x,y])
  if a[i][1]<b[j][1]:i+=1
  else:j+=1
 return out
def sub(a,b):
 out=[];j=0
 for x,y in a:
  while j<len(b) and b[j][1]<=x:j+=1
  k=j
  while k<len(b) and b[k][0]<y:
   if b[k][0]>x:out.append([x,min(y,b[k][0])])
   x=max(x,b[k][1])
   if x>=y:break
   k+=1
  if x<y:out.append([x,y])
 return out
def dur(xs):return sum(b-a for a,b in xs)/1000
def span(e):return [float(e['ts']),float(e['ts'])+float(e['dur'])]
def pinned(pin):
 p=root/pin['path'];assert p.resolve().is_relative_to(root)
 st=p.stat();raw=p.read_bytes();end=p.stat()
 assert (st.st_size,st.st_mtime_ns)==(end.st_size,end.st_mtime_ns)
 assert len(raw)==pin['bytes'] and sha256(raw).hexdigest()==pin['sha256']
 return raw
result=[]
for plan,old in zip(spec['ranks'],out):
 raw=pinned(plan['trace']);doc=json.loads(raw);del raw
 es=doc['traceEvents'] if isinstance(doc,dict) else doc
 py={e['pid'] for e in es if e.get('ph')=='M' and e.get('name')=='process_name' and e.get('args',{}).get('name')=='Python'}
 main=[e for e in es if e.get('ph')=='X' and e.get('dur') and e.get('pid') in py and e.get('tid')==e.get('pid')]
 log_by=defaultdict(list);byid={}
 for pin in plan['files']:
  for line in pinned(pin).splitlines():
   if not line:continue
   r=json.loads(line)
   if r.get('kind')!='adapter' and r.get('mode')=='detailed':log_by[r['name']].append(r);byid[r['id']]=r
 for rows in log_by.values():rows.sort(key=lambda r:r['wall_ns'])
 clock=old['clock'];origin=clock['origin_us'];offset=clock['offset_us'];scale=1+clock['slope_ppm']/1e6
 def mapped(ns):return origin+(ns/1000-origin-offset)/scale
 starts={n:[mapped(r['wall_ns']) for r in rows] for n,rows in log_by.items()}
 marks=[];unmatched=[];matched=0
 for e in main:
  if not e.get('name','').startswith('FT::'):continue
  n=e['name'][4:];a,b=span(e);rs=log_by[n];i=bisect.bisect_left(starts.get(n,[]),a-10)
  choices=[r for r in rs[i:i+4] if mapped(r['wall_ns'])>=a-10 and mapped(r['wall_ns']+r['duration_ns'])<=b+10]
  if not choices:unmatched.append(dict(name=n,interval=[a,b]));continue
  r=min(choices,key=lambda r:abs(mapped(r['wall_ns'])-a));matched+=1
  marks.extend(sub([[a,b]],[[mapped(r['wall_ns']),mapped(r['wall_ns']+r['duration_ns'])]]))
 assert len(unmatched)==plan['unmatched'];assert matched==plan['matched']
 marks=merge(marks);window=[[plan['window_start_us'],plan['window_end_us']]]
 free=inter(merge([span(e) for e in es if e.get('ph')=='X' and e.get('dur') and e.get('name')=='Free']),window)
 prep=inter(merge([span(e) for e in main if e.get('name')=='prepare input']),window);pf=inter(free,prep)
 batches=defaultdict(list)
 for r in old['selected']:
  if r['kind']=='cp' and r['caller']=='runner_build_attention':batches[r['parent']].append(r)
 calls=[];guard=clock['required_enclosure_residual_us']+1
 for parent,rs in sorted(batches.items()):
  assert len(rs)==26 and all(r['fully_enclosed'] for r in rs)
  r=max(rs,key=lambda r:r['wall_ns']);a=mapped(r['wall_ns']);b=mapped(r['wall_ns']+r['duration_ns']);pr=byid[parent];end=mapped(pr['wall_ns']+pr['duration_ns'])
  assert end>=b and end<=window[0][1]
  cs=[s for s in old['selected'] if s['kind']!='cp' and r['wall_ns']<=s['wall_ns'] and s['wall_ns']+s['duration_ns']<=r['wall_ns']+r['duration_ns']]
  assert sorted(s['kind'] for s in cs)==['base','base','local']
  body=[[a,b]];outer=[[a,end]];exposed=inter(body,pf);application=sub(exposed,marks);guarded=sub(inter([[a+guard,b-guard]],pf),marks)
  outer_app=sub(inter(outer,pf),marks)
  calls.append(dict(id=r['id'],parent_id=parent,ordinal=26,child_ids=[s['id'] for s in cs],body_ns=r['duration_ns'],body_us=[a,b],relaxed_to_parent_end_us=[a,end],free_prepare_us=exposed,marker_overlap_us=inter(exposed,marks),application_free_prepare_us=application,guarded_application_free_prepare_us=guarded,relaxed_outer_application_us=outer_app,body_wall_ms=(b-a)/1000,free_prepare_ms=dur(exposed),marker_ms=dur(inter(exposed,marks)),application_ms=dur(application),guarded_application_ms=dur(guarded),relaxed_outer_ms=dur(outer_app)))
 result.append(dict(ep=plan['ep'],window_ms=dur(window),trace=plan['trace'],files=plan['files'],clock=clock,marker_matched=matched,unmatched=unmatched,calls=calls,totals_ms={k:sum(c[k] for c in calls) for k in ['body_wall_ms','free_prepare_ms','marker_ms','application_ms','guarded_application_ms','relaxed_outer_ms']}))
 del es,doc,main,log_by,byid
print(json.dumps(dict(status='READ_ONLY_PINNED_LAST_TARGET_CP_METADATA_EXPOSURE',ranks=result,new_profile=0,new_NPU_work=0,service_requests=0,remote_writes=[])))
'''
def main():
 fine=json.loads((TIMING/'function-analysis.json').read_text());inv={r['path']:r for r in json.loads((TIMING/'raw-inventory.json').read_text())};receipt=json.loads((TIMING/'receipt.json').read_text());assert receipt['status']=='CAPTURED_RESTORED'
 plans=[]
 for r in fine['ranks']:
  trace=next(v for k,v in inv.items() if f'_ep{r["ep"]}_' in k and k.endswith('/trace_view.json'))
  plans.append(dict(ep=r['ep'],clock=r['clock_alignment'],window_start_us=r['window_start_us'],window_end_us=r['window_end_us'],matched=r['body_marker_matched'],unmatched=r['body_marker_unmatched'],files=[inv['timing/spans-'+str(pid)+'.jsonl'] for pid in r['python_pids']],trace=trace))
 spec=dict(root=receipt['output_host'],ranks=plans)
 remote=CALLERS.rsplit('print(json.dumps',1)[0]+TAIL
 p=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',remote,json.dumps(spec)],capture_output=True,text=True,timeout=300)
 if p.returncode:raise RuntimeError(p.stderr)
 d=json.loads(p.stdout);d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),caller_reader_sha256=sha256(Path('tools/read_common_frame_caller_logs.py').read_bytes()).hexdigest(),inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in [TIMING/'function-analysis.json',TIMING/'raw-inventory.json',TIMING/'receipt.json',ROOT/'target-draft-metadata-primary.json']],qualification='Same detailed balanced C8 diagnostic only. Whole final CP body, including both base and helper, with all matched FT boundary unions removed. Relaxed outer includes necessary unpadding/task submit and is only an optimistic parent envelope; queued work after parent return unpriced. Saved clock guard, 2 unmatched ranges retained. Not unprofiled net/request gain, original first C8 or ordinary-load price; source/bit/lifetime gates separate.')
 (ROOT/'target-draft-metadata-price-raw.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(ranks=len(d['ranks']),calls=sum(len(r['calls']) for r in d['ranks']),new_profile=0,remote_writes=[])))
if __name__=='__main__':main()
