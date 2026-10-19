from pathlib import Path
import collections,hashlib,json
r=Path(__file__).resolve().parent;b=Path('/Users/wio/work/glm53-execution-evidence-backup/20261018-p-first-seed-device-v2/recovery-v2/cumsum-i32-supply-profile');p=b/'trace_view.json';x=json.loads(p.read_text());events=x['traceEvents'] if isinstance(x,dict) else x;tasks=[z for z in events if 'Task Type' in z.get('args',{})];types=collections.Counter(z['args']['Task Type'] for z in tasks);links=collections.Counter()
def walk(x):
 if isinstance(x,dict):
  for k,v in x.items():
   if k in ['Transport Type','transport_type']:links[str(v)]+=1
   walk(v)
 elif isinstance(x,list):
  for v in x:walk(v)
walk(json.loads((b/'communication_matrix.json').read_text()))
rdma=[z for z in tasks if 'RDMA' in z['args']['Task Type'].upper() or 'RDMA' in z.get('name','').upper()]
report={'source_trace_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'SDMA_task_count':sum(v for k,v in types.items() if 'SDMA' in k),'SDMA_task_types':{k:v for k,v in types.items() if 'SDMA' in k},'RDMA_named_task_count':len(rdma),'hccl_batchGet_event_count':sum('batchget' in z.get('name','').lower() for z in events),'matrix_transport_types':dict(links),'parser_basis':'Native Task Type SDMA_SQE and matrix Transport Type keys; do not require literal SDMA or lowercase-only keys.','scope':'Only current candidate PD8 rank0 exported trace. Same original SDMA configuration and previously validated full32 baseline reused; not a new all32 coverage or every-byte/PD-read attribution proof.'}
assert report['SDMA_task_count']>0 and not rdma and set(links)<=set(['LOCAL','SIO','HCCS']) and links
(r/'cumsum-i32-sdma-sanity-v1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
