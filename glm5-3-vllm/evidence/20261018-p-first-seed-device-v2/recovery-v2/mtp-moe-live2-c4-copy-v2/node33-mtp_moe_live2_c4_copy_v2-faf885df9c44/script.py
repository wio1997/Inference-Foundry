"""Correlate the graph input copy through native InplaceCopy, not a DD queue name."""
from pathlib import Path
from bisect import bisect_left,bisect_right
from collections import Counter,defaultdict
import hashlib
import json

source=Path('/data/wio/glm5-3-vllm-execution/20261017-inventory/node33-mtp_moe_live2_c4_export_v1-df4d1eec4e9f/artifacts')
model_root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/mtp-moe-graph-live2-20261018/node33')
profile=json.loads((model_root/'mtp-moe-graph-c4-profile-v1/result.json').read_text())
prefixes=[x['D_response']['id'] for x in profile['rows'] if x['profiled']]
out=Path(__file__).resolve().parent/'artifacts';out.mkdir(exist_ok=False)
report={'completed':False,'ranks':[],'scope':'CPU-only all16 stopped trace analysis. Profile core actions are aligned in CPU order after the native profiler delay; replay has no native GMM and capture copies are first/last around the original core; each added input copy is matched to Enqueue/Dequeue InplaceCopy via correlation_id, then CANN API connection_id to physical device tasks. The initial exact acl_memcpy_device_to_device queue filter missed this native InplaceCopy path; no GPU rerun.'}
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
    ops.sort(key=lambda x:float(x['ts']))
    frames=[]
    for path in (model_root/'mtp-moe-graph-probe').glob('rank-%d-*.jsonl'%rank):
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if any(any(rid==prefix or rid.startswith(prefix+'-') for prefix in prefixes) for rid in row['ids']):frames.append(row)
    frames.sort(key=lambda x:x['monotonic_ns'])
    actions=[call for frame in frames for call in frame['calls']]
    dropped=len(actions)-len(ops)
    assert 0<=dropped<=15,(rank,'Opaque/core count mismatch',len(actions),len(ops))
    actions=actions[dropped:]
    native=0;copies=[]
    for op,action in zip(ops,actions):
        begin=float(op['ts']);end=begin+float(op['dur']);kids=main[bisect_left(ts,begin):bisect_right(ts,end)]
        gmm=any('GroupedMatmul' in x.get('name','') for x in kids)
        if action['action'] not in ['capture-and-replay','copy-and-replay']:
            assert gmm,(rank,'Expected native GMM',action);native+=1;continue
        if action['action']=='copy-and-replay':assert not gmm,(rank,'Expected only graph replay',action)
        enq=[x for x in kids if x.get('name')=='Enqueue@aclnnInplaceCopy']
        cpu=[x for x in kids if x.get('name')=='aten::copy_']
        if action['action']=='copy-and-replay':
            assert len(cpu)==len(enq)==1,(rank,action,len(cpu),len(enq))
            selected_enqueues=enq
        else:
            assert len(enq)>=2,(rank,'Capture needs pre/post copies',action,len(enq))
            selected_enqueues=[enq[0],enq[-1]]
        for enqueue in selected_enqueues:
            rec={'op':op,'source_action':action,'enqueue':enqueue,'device_tasks':[]}
            deq=dq.get(('Dequeue@aclnnInplaceCopy',enqueue['args']['correlation_id']));rec['dequeue']=deq
            if deq:
                t=float(deq['ts']);tid=deq['tid'];apis=api_by_tid[tid][bisect_left(api_times[tid],t):bisect_right(api_times[tid],t+float(deq['dur']))]
                rec['APIs']=apis;seen=set()
                for api in apis:
                    for task in tasks.get(api.get('args',{}).get('connection_id'),[]):
                        key=(task.get('name'),task.get('pid'),task.get('tid'),str(task.get('ts')))
                        if key not in seen:seen.add(key);rec['device_tasks'].append(task)
            data=[x for x in rec['device_tasks'] if any(k in x['args']['Task Type'] for k in ['DMA','AI_CORE','VECTOR','MIX_','AI_CPU'])]
            rec['data_tasks']=data;rec['copy_chain_complete']=deq is not None and bool(data)
            rec['all_data_tasks_SDMA']=bool(data) and all(x['args']['Task Type'] in ['SDMA_SQE','PCIE_DMA_SQE'] for x in data)
            copies.append(rec)
    types=Counter(x['args']['Task Type'] for rec in copies for x in rec['data_tasks'])
    result={'rank':rank,'trace_bytes':len(raw),'trace_sha256':hashlib.sha256(raw).hexdigest(),
            'native_opaque_calls':native,'candidate_input_copies':len(copies),'profile_delay_prefix_core_calls':dropped,'observed_selected_shapes':sorted(set(tuple(a['shape']) for a in actions if a['action'] in ['capture-and-replay','copy-and-replay'])),
            'all_copy_chains_complete':bool(copies) and all(x['copy_chain_complete'] for x in copies),
            'all_added_copy_data_tasks_SDMA':bool(copies) and all(x['copy_chain_complete'] and x['all_data_tasks_SDMA'] for x in copies),
            'data_task_types':dict(types),'sample':copies[:1],
            'closed_copy_chains':sum(z['copy_chain_complete'] for z in copies),
            'unclosed_copy_chains':sum(not z['copy_chain_complete'] for z in copies),
            'copy_coverage_by_shape':[{'shape':list(shape),'selected':sum(tuple(z['source_action']['shape'])==shape for z in copies),'closed_SDMA':sum(tuple(z['source_action']['shape'])==shape and z['copy_chain_complete'] and z['all_data_tasks_SDMA'] for z in copies)} for shape in sorted(set(tuple(z['source_action']['shape']) for z in copies))],
            'all_observed_copy_data_tasks_SDMA':bool(types) and set(types)<=set(['SDMA_SQE','PCIE_DMA_SQE'])}
    (out/('rank%d-copy-chains.json'%rank)).write_text(json.dumps(copies,indent=2)+'\n')
    (out/('rank%d-summary.json'%rank)).write_text(json.dumps(result,indent=2)+'\n')
    report['ranks'].append(result)
    del raw,obj,events,e,main,ts,dq,api_by_tid,api_times,tasks,copies,ops
report['all16_added_copy_SDMA']=len(report['ranks'])==16 and all(x['all_added_copy_data_tasks_SDMA'] for x in report['ranks'])
report['all16_selected_shape_SDMA_samples']=len(report['ranks'])==16 and all(z['all_observed_copy_data_tasks_SDMA'] and all(s['closed_SDMA']>0 for s in z['copy_coverage_by_shape']) for z in report['ranks'])
report['completed']=len(report['ranks'])==16
report['limitation']='Some replay copy CPU/API records have no device task connection in this stopped trace. Those instances remain unproven; representative per-shape chains do not prove every instance. Preserve complete-chain flag separately. No device rerun.'
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END')
assert report['completed'],report.get('failure')
