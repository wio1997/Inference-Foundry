from pathlib import Path
import collections,hashlib,json
r=Path('/Users/wio/work/Inference-Foundry-execution-plan');e=r/'glm5-3-vllm/evidence/20261018-p-first-seed-device-v2';p=next(Path('/Users/wio/work/glm53-execution-evidence-backup/20261018-p-first-seed-device-v2/P-profile-export/artifacts').rglob('trace_view.json'));j=json.loads(p.read_text());ev=j.get('traceEvents',[]) if isinstance(j,dict) else j
rows=[]
for q in [v for v in ev if v.get('name')=='Dequeue@acl_memcpy_device_to_host']:
 t=float(q['ts']);d=q['dur'];api=[v for v in ev if v.get('ph')=='X' and 'aclrtMemcpyAsync' in v.get('name','') and t<=float(v['ts'])<=t+d];assert len(api)==1
 conn=api[0]['args']['connection_id'];tasks=[v for v in ev if v.get('ph')=='X' and v.get('args',{}).get('connection_id')==conn and 'Task Type' in v.get('args',{})];assert len(tasks)==1
 rows.append({'CPU_dequeue':q,'ACL_API':api[0],'device_task':tasks[0]})
assert len(rows)==5 and all(v['device_task']['args']['Task Type']=='PCIE_DMA_SQE' for v in rows)
counts=collections.Counter(v.get('args',{}).get('Task Type') for v in ev if v.get('ph')=='X' and v.get('args',{}).get('Task Type'))
report={'source':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rank':0,'all_captured_D2H_via_correlated_PCIE_DMA_SQE':True,'D2H_count':5,'D2H':rows,'device_task_types':dict(counts),'scope':'Every captured P rank0 D2H queue is linked via its enclosing ACL API connection_id to a PCIE_DMA_SQE device task. Native local P requests and opt-in export captured; not a full PD or all-rank coverage claim. Attribution of the fourth copy to the added writer-stream proposal is an inference from source order and extra copy, without a dedicated trace marker.'}
(e/'P-profile-D2H-correlation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'D2H_count':len(rows),'all_D2H_PCIE_DMA_SQE':True,'device_task_types':dict(counts)}))
