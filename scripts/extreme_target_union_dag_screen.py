#!/usr/bin/env python3
"""Observed device interval unions inside frozen target scopes, not a bound."""
import argparse,csv,glob,json,statistics
from pathlib import Path

def union(xs):
    xs=sorted(xs)
    if not xs:return 0.0
    acc=0.0;lo,hi=xs[0]
    for a,b in xs[1:]:
        if a>hi:acc+=hi-lo;lo,hi=a,b
        elif b>hi:hi=b
    return acc+hi-lo

def run(root):
 windows=[];invalid=[]
 for rank in range(8):
  fs=sorted(glob.glob(str(root/f'rank{rank}_*/ASCEND_PROFILER_OUTPUT/kernel_details.csv')))
  if not fs:invalid.append((rank,'missing'));continue
  f=Path(fs[-1]);a=json.loads((f.parent/'trace_view.json').read_text())
  scopes=sorted((e for e in a if e.get('name')=='extreme::target' and e.get('cat')=='cpu_op'),key=lambda e:float(e['ts']))
  if len(scopes)!=2:invalid.append((rank,'scopes',len(scopes)));continue
  with f.open() as h:rows=list(csv.DictReader(h))
  for cyc,s in enumerate(scopes):
   start=float(s['ts']);end=start+float(s['dur']);comp=[];comm=[];count=0
   for r in rows:
    if not str(r['Task ID']).isdigit():continue
    try:ts=float(r['Start Time(us)'].strip());dur=float(r['Duration(us)'])
    except (ValueError,KeyError):continue
    if not start<=ts<end or dur<=0:continue
    count+=1
    if r['Type'].startswith('hcom_'):continue
    else:comp.append((ts,ts+dur))
   comm=[(float(e['ts']),float(e['ts'])+float(e['dur'])) for e in a if e.get('ph')=='X' and isinstance(e.get('args'),dict) and 'size(Byte)' in e['args'] and start<=float(e['ts'])<end]
   if not comp or not comm:invalid.append((rank,cyc,'empty'));continue
   allx=comp+comm;first=min(x[0] for x in allx);last=max(x[1] for x in allx)
   c=union(comp);h=union(comm);total=union(allx)
   windows.append({'rank':rank,'cycle':cyc,'device_span_ms':(last-first)/1000,'compute_copy_union_ms':c/1000,'hcom_union_ms':h/1000,'overlap_ms':(c+h-total)/1000,'all_busy_union_ms':total/1000,'device_gap_ms':((last-first)-total)/1000,'kernel_rows':count,'hcom_rows':len(comm)})
 return windows,invalid

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--profile-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 windows,invalid=run(a.profile_dir)
 keys=('device_span_ms','compute_copy_union_ms','hcom_union_ms','overlap_ms','all_busy_union_ms','device_gap_ms','hcom_rows')
 summary={k:{'median':statistics.median(w[k] for w in windows),'min':min(w[k] for w in windows),'max':max(w[k] for w in windows)} for k in keys} if windows else {}
 out={'status':'valid' if len(windows)==16 and not invalid else 'invalid','windows':windows,'summary':summary,'invalid':invalid,'meaning':'Observed synchronized-profiler device interval unions in latest target scopes across eight ranks; a scheduling diagnosis, not necessary work, exposed saving, physical resource bound or E2E ceiling. Event waits/Host submission not modeled. Kernels with nonnumeric Task ID excluded.'}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'summary':summary,'invalid':invalid},indent=2))
if __name__=='__main__':main()
