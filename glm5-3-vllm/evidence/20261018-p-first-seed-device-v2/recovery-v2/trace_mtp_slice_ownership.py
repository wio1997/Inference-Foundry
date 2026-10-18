from pathlib import Path
from collections import Counter,defaultdict
from bisect import bisect_left,bisect_right
import csv,hashlib,json,statistics,tarfile
r=Path(__file__).resolve().parent
base=Path('/Users/wio/work/glm53-execution-evidence-backup/20261018-p-first-seed-device-v2/recovery-v2/mtp-moe-live1-profile');p=base/'rank0-trace_view.json';csvp=base/'rank0-kernel_details.csv'
if not csvp.exists():
 with tarfile.open(base/'node33-mtp_moe_graph_export_v1-eb104ea3dcbb.tar.gz','r|gz') as tar:
  for m in tar:
   if m.name=='artifacts/rank0/kernel_details.csv':csvp.write_bytes(tar.extractfile(m).read());break
assert csvp.exists()
x=json.loads(p.read_text());e=x['traceEvents'] if isinstance(x,dict) else x;compute=[z for z in e if z.get('ph')=='X' and z.get('name')=='aclnnIndex_SliceAiCore_Slice' and z.get('args',{}).get('Model Id')==4294967295]
launch={z.get('args',{}).get('connection_id'):z for z in e if z.get('name')=='Node@launch' and z.get('ph')=='X'}
dequeues=defaultdict(list);cpu=defaultdict(list);enqueues={}
for z in e:
 if z.get('cat')=='dequeue':dequeues[z['tid']].append(z)
 if z.get('cat')=='cpu_op':cpu[(z['pid'],z['tid'])].append(z)
 if z.get('cat')=='enqueue':enqueues[z.get('args',{}).get('correlation_id')]=z
for v in dequeues.values():v.sort(key=lambda z:float(z['ts']))
dt={k:[float(z['ts']) for z in v] for k,v in dequeues.items()}
for v in cpu.values():v.sort(key=lambda z:float(z['ts']))
ct={k:[float(z['ts']) for z in v] for k,v in cpu.items()}
bytask=defaultdict(list)
with csvp.open() as f:
 for z in csv.DictReader(f):
  if z['Name']=='aclnnIndex_SliceAiCore_Slice':bytask[(int(z['Task ID']),int(z['Stream ID']))].append(z)
rows=[]
for z in compute:
 node=launch.get(z['args']['connection_id']);assert node
 t=float(node['ts']);tid=node['tid'];arr=dequeues[tid];i=bisect_right(dt[tid],t)-1;deq=arr[i];assert t<=float(deq['ts'])+float(deq['dur']);enq=enqueues[deq['args']['correlation_id']];key=(enq['pid'],enq['tid']);t=float(enq['ts']);parents=[v for v in cpu[key][bisect_left(ct[key],t-100000):bisect_right(ct[key],t)] if t<=float(v['ts'])+float(v['dur'])]
 options=bytask[(int(z['args']['Task Id']),int(z['args']['Physic Stream Id']))];row=min(options,key=lambda v:abs(float(v['Start Time(us)'])-float(z['ts'])));assert abs(float(row['Start Time(us)'])-float(z['ts']))<1
 rows.append({'duration_us':z['dur'],'task':z['args']['Task Id'],'device_ts':z['ts'],'Input Shapes':row['Input Shapes'],'Output Shapes':row['Output Shapes'],'Input Data Types':row['Input Data Types'],'cpu_parent_names':[v['name'] for v in sorted(parents,key=lambda v:float(v['dur']),reverse=True)],'enqueue':enq['name'],'dequeue':deq['name']})
groups=defaultdict(list)
for z in rows:groups[(z['Input Shapes'],z['Output Shapes'],tuple(z['cpu_parent_names']))].append(z)
summary=[{'input_shapes':k[0],'output_shapes':k[1],'cpu_parent_names':list(k[2]),'n':len(v),'median_us':statistics.median(z['duration_us'] for z in v),'sum_us':sum(z['duration_us'] for z in v),'sample':v[0]} for k,v in groups.items()];summary.sort(key=lambda z:z['sum_us'],reverse=True)
report={'completed':True,'source_trace_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_csv_sha256':hashlib.sha256(csvp.read_bytes()).hexdigest(),'matched':len(rows),'groups':summary,'scope':'Existing stopped rank0 trace+CSV CPU-only ownership. Kernel duration sum may overlap; parent CPU op names and exact input/output shapes support selecting the next fragment, not end-to-end gain or complete logical call-stack proof.'}
(r/'mtp-slice-ownership-v1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
