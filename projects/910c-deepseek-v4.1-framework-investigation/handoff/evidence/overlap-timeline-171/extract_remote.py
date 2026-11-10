"""Read completed traces only; export exact intervals and representative tasks."""
from pathlib import Path
from collections import defaultdict
import bisect,concurrent.futures,csv,hashlib,json,re,statistics,sys

def merge(xs):
 out=[]
 for a,b in sorted(xs):
  if b<=a:continue
  if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
  else:out.append([a,b])
 return out

def intersect(xs,ys):
 out=[];i=j=0
 while i<len(xs) and j<len(ys):
  a=max(xs[i][0],ys[j][0]);b=min(xs[i][1],ys[j][1])
  if b>a:out.append([a,b])
  if xs[i][1]<ys[j][1]:i+=1
  else:j+=1
 return out

def diff(xs,ys):
 out=[];j=0
 for a,b in xs:
  while j<len(ys) and ys[j][1]<=a:j+=1
  k=j
  while k<len(ys) and ys[k][0]<b:
   if ys[k][0]>a:out.append([a,min(b,ys[k][0])])
   a=max(a,ys[k][1]);k+=1
   if a>=b:break
  if a<b:out.append([a,b])
 return out

def length(xs):return sum(b-a for a,b in xs)/1000

def one(path):
 raw=json.load(open(path));events=raw if isinstance(raw,list) else raw['traceEvents'];ep=int(re.search(r'_ep(\d+)_',str(path))[1])
 proc={e['pid']:e.get('args',{}).get('name') for e in events if e.get('ph')=='M' and e.get('name')=='process_name'}
 py={p for p,n in proc.items() if n=='Python'}
 es=[e for e in events if e.get('ph')=='X' and e.get('dur')];names=['prepare input','forward','post process','sample_token','draft_token','async_state_update']
 host=[e for e in es if e.get('pid') in py and e.get('pid')==e.get('tid') and e.get('name') in names]
 starts=sorted(float(e['ts']) for e in es if e.get('name')=='MODEL_EXECUTE');assert len(starts)==12
 lo,hi=starts[0],starts[-1];window=[[lo,hi]]
 span=lambda e:[float(e['ts']),float(e['ts'])+float(e['dur'])]
 activity={k:intersect(merge([span(e) for e in es if e.get('name')==k]),window) for k in ['Free','Computing','Communication(Not Overlapped)']}
 scopes={k:merge([span(e) for e in host if e['name']==k]) for k in names};prep=scopes['prepare input'];prep_free=intersect(activity['Free'],prep)
 cycles=[]
 for i,(a,b) in enumerate(zip(starts,starts[1:])):
  w=[[a,b]];p=intersect(prep,w);c=intersect(activity['Computing'],w)
  cycles.append({'index':i,'start_ms':(a-lo)/1000,'end_ms':(b-lo)/1000,'period_ms':(b-a)/1000,
    'compute_ms':length(c),'free_ms':length(intersect(activity['Free'],w)),
    'communication_ms':length(intersect(activity['Communication(Not Overlapped)'],w)),
    'prepare_wall_ms':length(p),'prepare_free_ms':length(intersect(prep_free,w)),
    'compute_during_prepare_ms':length(intersect(c,p))})
 # Tasks retained only within three central cycles, plus exact activity for all.
 tasks=[]
 if ep in [0,8]:
  a,b=starts[3],starts[6]
  for r in csv.DictReader(open(path.parent/'kernel_details.csv')):
   try:x=float(r['Start Time(us)']);y=x+float(r['Duration(us)'])
   except (ValueError,KeyError):continue
   if y<a or x>b:continue
   target=str(r.get('Model ID','')) not in ['','4294967295','-1']
   tasks.append({'name':r.get('Type',r.get('Name')),'start_ms':(x-lo)/1000,'end_ms':(y-lo)/1000,'stream':r.get('Stream ID'),'task':r.get('Task ID'),'target_graph':target,
                 'shape':r.get('Input Shapes',''),'model':r.get('Model ID')})
 return {'ep':ep,'dp':ep//8,'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'origin_us':lo,
  'window_ms':(hi-lo)/1000,'cycles':cycles,
  'activity':{k:[[round((a-lo)/1000,6),round((b-lo)/1000,6)] for a,b in v] for k,v in activity.items()},
  'host':[{'name':e['name'],'start_ms':(span(e)[0]-lo)/1000,'end_ms':(span(e)[1]-lo)/1000} for e in host if span(e)[1]>lo and span(e)[0]<hi],
  'totals':{'compute_ms':length(activity['Computing']),'free_ms':length(activity['Free']),'comm_ms':length(activity['Communication(Not Overlapped)']),
            'prepare_free_ms':length(prep_free),'prepare_wall_ms':length(intersect(prep,window)),'compute_during_prepare_ms':length(intersect(activity['Computing'],prep))},'tasks':tasks}

if __name__=='__main__':
 root=Path(sys.argv[1]);files=sorted((root/'traces').rglob('trace_view.json'));assert len(files)==16
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:ranks=list(pool.map(one,files))
 print(json.dumps({'schema':1,'source_capture':str(root),'new_capture':False,'ranks':ranks},separators=(',',':')))
