"""Correlate the graph input copy through native InplaceCopy, not a DD queue name."""
from pathlib import Path
from bisect import bisect_left,bisect_right
from collections import Counter,defaultdict
import hashlib
import json

source=Path('/data/wio/glm5-3-vllm-execution/20261017-inventory/node33-mtp_moe_graph_export_v1-eb104ea3dcbb/artifacts')
out=Path(__file__).resolve().parent/'artifacts';out.mkdir(exist_ok=False)
report={'completed':False,'ranks':[],'scope':'CPU-only all16 stopped trace analysis. Candidate opaque MoE calls have no native GMM enqueue; each added input copy is matched to Enqueue/Dequeue InplaceCopy via correlation_id, then CANN API connection_id to physical device tasks. The initial exact acl_memcpy_device_to_device queue filter missed this native InplaceCopy path; no GPU rerun.'}
for rank in range(16):
    p=source/('rank%d'%rank)/'trace_view.json';raw=p.read_bytes();obj=json.loads(raw);events=obj['traceEvents'] if isinstance(obj,dict) else obj
    e=[x for x in events if x.get('ph')=='X'];ops=[x for x in e if x.get('name')=='vllm::moe_forward_shared']
    assert ops,(rank,'No opaque CPU events')
    main=sorted([x for x in e if x.get('pid')==ops[0]['pid'] and x.get('tid')==ops[0]['tid']],key=lambda x:float(x['ts']))
    ts=[float(x['ts']) for x in main]
    dq={(x['name'],x.get('args',{}).get('correlation_id')):x for x in e if x.get('cat')=='dequeue'}
    api_by_tid=defaultdict(list);tasks=defaultdict(list)
    for event in e:
        if event.get('name','').startswith('AscendCL@'):api_by_tid[event['tid']].append(event)
        if 'Task Type' in event.get('args',{}) and 'connection_id' in event['args']:tasks[event['args']['connection_id']].append(event)
    api_times={}
    for tid in api_by_tid:
        api_by_tid[tid].sort(key=lambda x:float(x['ts']));api_times[tid]=[float(x['ts']) for x in api_by_tid[tid]]
    native=0;copies=[]
    for op in ops:
        begin=float(op['ts']);end=begin+float(op['dur']);kids=main[bisect_left(ts,begin):bisect_right(ts,end)]
        if any('GroupedMatmul' in x.get('name','') for x in kids):native+=1;continue
        cpu=[x for x in kids if x.get('name')=='aten::copy_'];enq=[x for x in kids if x.get('name')=='Enqueue@aclnnInplaceCopy']
        rec={'op':op,'cpu_copies':cpu,'enqueue':enq,'device_tasks':[]}
        if len(enq)==1:
            deq=dq.get(('Dequeue@aclnnInplaceCopy',enq[0]['args']['correlation_id']));rec['dequeue']=deq
            if deq:
                t=float(deq['ts']);tid=deq['tid'];apis=api_by_tid[tid][bisect_left(api_times[tid],t):bisect_right(api_times[tid],t+float(deq['dur']))]
                rec['APIs']=apis;seen=set()
                for api in apis:
                    for task in tasks.get(api.get('args',{}).get('connection_id'),[]):
                        key=(task.get('name'),task.get('pid'),task.get('tid'),str(task.get('ts')))
                        if key not in seen:seen.add(key);rec['device_tasks'].append(task)
        data=[x for x in rec['device_tasks'] if any(k in x['args']['Task Type'] for k in ['DMA','AI_CORE','VECTOR','MIX_','AI_CPU'])]
        rec['data_tasks']=data
        rec['copy_chain_complete']=len(cpu)==1 and len(enq)==1 and bool(data)
        rec['all_data_tasks_SDMA']=bool(data) and all(x['args']['Task Type'] in ['SDMA_SQE','PCIE_DMA_SQE'] for x in data)
        copies.append(rec)
    types=Counter(x['args']['Task Type'] for rec in copies for x in rec['data_tasks'])
    result={'rank':rank,'trace_bytes':len(raw),'trace_sha256':hashlib.sha256(raw).hexdigest(),
            'native_opaque_calls':native,'candidate_opaque_calls':len(copies),
            'all_copy_chains_complete':bool(copies) and all(x['copy_chain_complete'] for x in copies),
            'all_added_copy_data_tasks_SDMA':bool(copies) and all(x['copy_chain_complete'] and x['all_data_tasks_SDMA'] for x in copies),
            'data_task_types':dict(types),'sample':copies[:1]}
    (out/('rank%d-copy-chains.json'%rank)).write_text(json.dumps(copies,indent=2)+'\n')
    (out/('rank%d-summary.json'%rank)).write_text(json.dumps(result,indent=2)+'\n')
    report['ranks'].append(result)
    if not result['all_added_copy_data_tasks_SDMA']:
        report['failure']='Unproven or non-SDMA copy chain at rank '+str(rank);break
    del raw,obj,events,e,main,ts,dq,api_by_tid,api_times,tasks,copies,ops
report['all16_added_copy_SDMA']=len(report['ranks'])==16 and all(x['all_added_copy_data_tasks_SDMA'] for x in report['ranks'])
report['completed']=report['all16_added_copy_SDMA']
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END')
assert report['completed'],report.get('failure')
