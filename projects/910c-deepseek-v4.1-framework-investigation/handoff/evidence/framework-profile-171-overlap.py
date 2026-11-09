from pathlib import Path
from collections import Counter
import concurrent.futures,json,sys

def merge(xs):
 out=[]
 for a,b in sorted(xs):
  if not out or a>out[-1][1]:out.append([a,b])
  else:out[-1][1]=max(out[-1][1],b)
 return out

def length(xs):return sum(b-a for a,b in xs)
def overlap(a,b):
 i=j=0;v=0
 while i<len(a) and j<len(b):
  v+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
  if a[i][1]<b[j][1]:i+=1
  else:j+=1
 return v

def one(path):
 es=json.load(open(path));es=es.get('traceEvents',[]) if isinstance(es,dict) else es
 keys={'gloo:all_reduce':'gloo','Event::synchronize':'event_sync','Free':'free','Computing':'compute','Communication(Not Overlapped)':'communication','DispatchFFNCombine':'dfc'}
 xs={k:[] for k in keys.values()};counts=Counter();procs={e['pid']:e.get('args',{}).get('name') for e in es if e.get('ph')=='M' and e.get('name')=='process_name'}
 for e in es:
  if e.get('ph')!='X' or e.get('name') not in keys:continue
  a=float(e['ts']);b=a+float(e.get('dur',0));k=keys[e['name']];xs[k].append((a,b));counts[k]+=1
 xs={k:merge(v) for k,v in xs.items()};device=merge(xs['free']+xs['compute']+xs['communication']);span=length(device)
 output={'rank_dir':path.parent.parent.name,'device_window_ms':span/1000,'process_names':procs,'calls':dict(counts),'union_ms':{k:length(v)/1000 for k,v in xs.items()},'device_window_pct':{k:100*overlap(v,device)/span for k,v in xs.items()},'gloo_and_free_ms':overlap(xs['gloo'],xs['free'])/1000,'event_sync_and_free_ms':overlap(xs['event_sync'],xs['free'])/1000,'gloo_largest_ms':sorted([(b-a)/1000 for a,b in xs['gloo']],reverse=True)[:6]}
 return output

if __name__=='__main__':
 root=Path(sys.argv[1]);paths=list((root/'traces').rglob('trace_view.json'))
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  data=list(pool.map(one,paths))
 (root/'framework-overlap.json').write_text(json.dumps({'files':data},indent=2))
 print('done',len(data),flush=True)
