#!/usr/bin/env python3
"""Offline fail-closed Run507 cohort certificate; no device imports."""
import argparse,json,math,hashlib,re
from pathlib import Path
LABELS=('T','R0','R1','H','U')
EDGES=('T_R0_ms','R0_R1_ms','R1_H_ms','H_U_ms')
PINS=dict(c10d='31155344233ae77360a630bbeacf1b43cfa18f1923cf2103cd5b3142db39ec5c',
 logger='b7b5866e4d3c3847e24f3fe488f187cbd90ac94955a24a98dc10f0d96c99b649',
 torch_npu='83fb9a0eb249aef6bca7f8463047fcb4d3062cc4849f17ddf31a5e050c838842',
 ascendcl='123d67c313f743e5d6f2e856f40c24d5edac376975cf0a7a1019e079b4171480',
 runtime='7bc2b610dff552df2f114c95701156b4aba43ffcba24e06e3c249e1cf57bbc21',
 current_stream='608b5984180706f4179c7d6427242ec9163b840b5d4fb56459f92d359c3a84b0',
 event='241741b356db5006797dbec119007afb33b73b0e1753bfb06ab27c989c5bfac4',
 graph_replay='eb11cac181087766e8489cf3a1da0c0fa150933b7e74454dba4a47db6114164e',
 graph_call='6d2ee39b2980a8554717524d04d5ddb5672633cc0bf107eac3daee39d3441199')
SOURCE_ROWS=[
 ('runtime','/data/wio/Inference_Foundry/runtime/extreme_decode.py','eb4b5142cc475a51e06c6eec8238e9cfeb918b568e8a5f27b0a4bc7d26963499','e9f531e3d53571eb0b1241140c97961ad982143709371de718ce6c07ebde728f'),
 ('serving','/data/wio/Inference_Foundry/runtime/fixed_serving.py','137f1cc6a460700e6f26e5837aa062022eb621198e33099b85988f8b7ef6743a','212455832f138fb99413897784479436d2881985b14870343684ecd432204223'),
 ('target','/data/wio/Inference_Foundry/runtime/target_adapter.py','c46d60318098077064c0ff57eae98bc937dcf943bda531918388be4e9128acd5','61bcab596b831fab7e0e7d81ef71fb76d3f08f4be170c9f7a1845d029c17e60a'),
 ('handoff','/data/wio/Inference_Foundry/bootstrap/vllm_target_handoff.py','2053dafbd46638d0da23e14b96c59ea01995ec2f0441d01928f1c9484c7b4ed5','1d2ed9af027c3fba1e70be55ae91cd5988cb9ab69fd2607f46925a98f100fa45'),
 ('graph','/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/compilation/acl_graph.py','6396ca409d633ee61c4433a762982cf901b893f252c864c0ff05ac17bdbdb3f6','6d2ee39b2980a8554717524d04d5ddb5672633cc0bf107eac3daee39d3441199'),
 ('comm','/data/wio/vllm_ascend_26/framework/vllm/vllm/distributed/device_communicators/base_device_communicator.py','c4fafc71bbb3a7652ecdf425bf116e3a9ad203ba44baf005cbab5db88a8b8221','e486febeeac426a509414293a3202b90aa8891b0fd9a9c191c8ec59963db8c46')]
SOURCE_MANIFEST=[dict(key=k,source=p,original=o,patched=n) for k,p,o,n in SOURCE_ROWS]
def hex64(x):return isinstance(x,str) and re.fullmatch(r'[0-9a-f]{64}',x) is not None
def source_cert(x):
    return (isinstance(x,dict) and all(isinstance(x.get(k),str) and x[k] for k in
            ('module','qualname','path')) and hex64(x.get('sha256')))
def callable_cert(x,id_key):
    return (isinstance(x,dict) and exact_int(x.get(id_key),1) and
            (x.get('owner_id') is None or exact_int(x.get('owner_id'),1)) and
            source_cert(x.get('source')))

class Rejected(ValueError):pass
def gate(x,msg):
    if not x:raise Rejected(msg)
def exact_int(x,minimum=0):return type(x) is int and x>=minimum
def descriptor(x,device,shape=None,dtype=None):
    gate(isinstance(x,dict) and set(x)=={'data','storage','storage_bytes','offset','shape','stride','dtype','device'},'tensor descriptor fields')
    gate(exact_int(x['data'],1) and exact_int(x['storage'],1) and
         exact_int(x['storage_bytes'],1) and exact_int(x['offset']) and
         isinstance(x['shape'],list) and all(exact_int(v,1) for v in x['shape']) and
         isinstance(x['stride'],list) and len(x['stride'])==len(x['shape']) and
         all(type(v) is int for v in x['stride']) and x['device']==device and
         isinstance(x['dtype'],str),'tensor descriptor values')
    if shape is not None:gate(x['shape']==shape,'tensor shape')
    if dtype is not None:gate(x['dtype']==dtype,'tensor dtype')
    return x
def leaves(x,device):
    gate(isinstance(x,dict) and x.get('kind') in ('tensor','list','tuple'),'output tree node')
    if x['kind']=='tensor':
        gate(set(x)=={'kind','value'},'output tensor tree fields')
        return [descriptor(x['value'],device)]
    gate(set(x)=={'kind','children'} and isinstance(x['children'],list),'output container tree fields')
    out=[]
    for child in x['children']:out.extend(leaves(child,device))
    return out
def validate_detail(d):
    rank=d['rank'];device=f'npu:{rank}';stream=d['stream']
    gate(isinstance(stream,list) and len(stream)==2 and stream[0]==device and
         isinstance(stream[1],dict) and set(stream[1])=={'stream_id','device_index','device_type'} and
         exact_int(stream[1]['stream_id']) and type(stream[1]['device_index']) is int and
         stream[1]['device_index']==rank and type(stream[1]['device_type']) is int and
         stream[1]['device_type']==20,'logical stream rank/device/type')
    setup=d['setup']
    gate(setup.get('stream')==stream and setup.get('events_initialized')==5 and
         setup.get('completion')=='setup-only event.synchronize','setup event completion')
    config=d.get('config',{})
    gate(config.get('batch_size')==12 and config.get('target_tokens_per_request')==8 and
         config.get('target_token_count')==96 and config.get('max_output_tokens')==1024 and
         exact_int(config.get('max_cycles'),65),'frozen Target/serving configuration')
    source_pins=setup.get('source_pins',{})
    gate(exact_int(source_pins.get('c10d_callable_id'),1) and
         isinstance(source_pins.get('bindings'),dict) and
         set(source_pins['bindings'])=={'graph_call','current_stream','event','graph_replay'} and
         all(isinstance(b,dict) and isinstance(b.get('path'),str) and
             len(b.get('sha256',''))==64 for b in source_pins['bindings'].values()),
         'Python graph/stream/event binding provenance')
    gate(all(source_pins['bindings'][name]['sha256']==PINS[name] for name in
             ('graph_call','current_stream','event','graph_replay')) and
         source_pins.get('graph_debug_dump',{}).get('sha256')==PINS['graph_replay'],
         'installed Torch-NPU Python binding hashes')
    flags=d.get('runtime_flags',{})
    gate(flags.get('schedule_mode')=='off' and
         all(flags.get(k) is False for k in ('profile_dag','diagnose','cycle_profiler',
             'profile_scopes','kv_slot_audit','target_page_audit','schedule_verify',
             'diagnostic_refresh','competing_capture')) and
         flags.get('cycle_profile_dir') is None and
         flags.get('competing_capture_env')=={},'runtime mode/competing diagnostic')
    inp=descriptor(d.get('input'),device,[96],'torch.int32')
    descriptor(d.get('positions'),device,[96],'torch.int64')
    descriptor(d.get('indices'),device,[96],'torch.int64')
    sample=descriptor(d.get('sample_hidden'),device)
    gate(sample['shape'][0]==96 and len(sample['shape'])==2 and
         sample['dtype']=='torch.bfloat16','sampled hidden shape/dtype')
    descriptor(d.get('active_mask'),device,[12],'torch.bool')
    event_ids=set();host=[]
    for name in LABELS:
        marker=d['markers'][name]
        gate(exact_int(marker.get('event'),1) and exact_int(marker.get('host_ns'),1),
             'marker event/Host timestamp')
        event_ids.add(marker['event']);host.append(marker['host_ns'])
        gate(descriptor(marker.get('tensor'),device)==(sample if name=='U' else inp),
             'marker tensor lineage')
    gate(len(event_ids)==5 and host==sorted(host),'per-slice events/Host order')
    intervals=d.get('intervals')
    gate(isinstance(intervals,dict) and set(intervals)==set(EDGES) and
         all(type(intervals[k]) in (int,float) and math.isfinite(intervals[k]) and
             intervals[k]>=0 for k in EDGES),'finite local event intervals')
    sync=d['sync'];replay=d['replay']
    gate(sync.get('stream')==stream[1] and
         host[0]<=sync['begin_ns']<=sync['end_ns']<=host[1] and
         replay.get('entry_id')==sync.get('entry_id') and
         exact_int(replay.get('entry_id'),1) and exact_int(replay.get('graph_id'),1) and
         exact_int(replay.get('capture_generation'),1) and
         exact_int(replay.get('selected_observation_ordinal'),1),'sync/replay provenance')
    gate(replay.get('original_branch')==dict(need_sync=True,is_draft_eagle=False,
         enable_enpu=False,runtime_mode='FULL'),'ordinary FULL replay branch')
    gate(sync.get('original_branch')==replay['original_branch'] and
         replay.get('completion')=='conditional: caller stream/child join/source contract' and
         isinstance(replay.get('arguments'),dict) and
         replay.get('capture_scope',{}).get('target') is True and
         replay['capture_scope'].get('runtime_mode')=='FULL' and
         replay['capture_scope'].get('enable_enpu') is False and
         replay['capture_scope'].get('is_draft_model') is False and
         replay.get('batch_descriptor')==replay['capture_scope'].get('batch_descriptor') and
         replay['capture_scope'].get('origin') in ('startup_full_decode96','runtime_target') and
         replay['capture_scope'].get('batch_num_tokens')==96 and
         replay['capture_scope'].get('batch_num_reqs')==12 and
         replay['capture_scope'].get('uniform') is True and
         exact_int(replay['capture_scope'].get('forward_model_id'),1) and
         replay['capture_scope'].get('bound_runtime_target_model_id')==
             replay['capture_scope']['forward_model_id'] and
         replay['capture_scope'].get('bound_runtime_target_batch')==
             replay['capture_scope']['batch_descriptor'],
         'actual Target FULL capture/replay scope and conditional completion')
    args=replay['arguments']
    gate(args.get('kind')=='dict' and isinstance(args.get('items'),dict) and
         args['items'].get('args')==dict(kind='tuple',children=[]) and
         args['items'].get('kwargs',{}).get('kind')=='dict',
         'captured/replayed full argument tree')
    kwargs=args['items']['kwargs'].get('items',{})
    gate(kwargs.get('input_ids')==dict(kind='tensor',value=inp) and
         kwargs.get('positions')==dict(kind='tensor',value=d['positions']),
         'Target input/position capture generation')
    graph=d['graph_output'];post=d['post_gather']
    source=leaves(graph,device);dest=leaves(post,device)
    gate(len(source)==len(dest)==4 and post['kind']==graph['kind'] and
         all(x['shape']==[12,4096] and x['dtype']=='torch.bfloat16' for x in source) and
         all(x['shape']==[96,4096] and x['dtype']=='torch.bfloat16' for x in dest),
         'hidden/aux tree shape and leaf order')
    gate(d['pre_gather']==graph==replay.get('output_owner'),'graph output ownership')
    gate(d.get('flash_comm') is True and exact_int(d.get('pad_size')) and
         d.get('num_tokens')==96 and d.get('padded')==96+d['pad_size'],
         'frozen hidden gather context')
    hidden=d['hidden'];native=d['native_calls']
    gate(len(hidden)==len(native)==len(source),'one native gather per hidden leaf')
    gate(all(native[i]['return_ns']<=native[i+1]['entry_ns']
             for i in range(len(native)-1)), 'ordered synchronous native gathers')
    for pre,after,row,call in zip(source,dest,hidden,native):
        gate(row.get('gather') is True and row.get('stream')==stream and
             row.get('input')==pre and row.get('output')==after and
             row.get('native_output_same_storage') is True,'hidden/aux leaf lineage')
        a=descriptor(call.get('input'),device);b=descriptor(call.get('output'),device)
        gate(a==pre and len(a['shape'])==len(b['shape']) and
             b['shape']==[8*a['shape'][0]]+a['shape'][1:] and
             a['dtype']==b['dtype']==after['dtype'] and b['storage']==after['storage'] and
             b['data']==after['data'] and b['storage_bytes']==after['storage_bytes'] and
             b['stride']==after['stride'] and
             after['shape']==[b['shape'][0]-d['pad_size']]+b['shape'][1:] and
             after['offset']==b['offset'],'native concat/view/slice relation')
        group=call.get('group',{});dispatch=call.get('dispatch',{})
        gate(group.get('members')==list(range(8)) and group.get('backend')=='hccl' and
             group.get('member_rank')==rank and dispatch.get('ordinary') is True and
             dispatch.get('async_op') is False and dispatch.get('coalescing') is False and
             dispatch.get('capture') is False and dispatch.get('compiled') is False and
             dispatch.get('modes_clean') is True and
             call.get('dispatch_gate')==dict(user_stack=0,functional=False,
                                            proxy=False,pre_dispatch=False) and
             call.get('torch_function_override') is False and
             exact_int(call.get('entry_ns'),1) and exact_int(call.get('return_ns'),1) and
             host[2]<=call['entry_ns']<=call['return_ns']<=host[3],
             'ordinary native TP provenance')
    gate(dest[0]['shape'][0]==96 and sample['shape'][1:]==dest[0]['shape'][1:],
         'Target hidden/sample dimensions')
    updates=d.get('graph_update')
    gate(isinstance(updates,list) and len(updates)<=1,'graph update branch count')
    bound=setup.get('graph_update_bound',{})
    gate(d.get('graph_update_branch')==bound.get('branch') and
         bound.get('branch')=='after' and len(updates)==1 and
         bound.get('private_stream_relation')=='unresolved',
         'graph update setup/selected branch')
    actual=d.get('actual_update',{})
    gate(isinstance(actual,dict) and actual.get('return_count')==1 and
         actual.get('branch')=='after' and
         actual.get('backend_id')==bound.get('backend',{}).get('object_id') and
         exact_int(actual.get('impl_id'),1) and
         callable_cert(bound.get('backend'),'object_id') and
         callable_cert(bound.get('get_impl_cls'),'callable_id') and
         isinstance(actual.get('update_callable'),dict) and
         exact_int(actual['update_callable'].get('callable_id'),1) and
         (actual['update_callable'].get('owner_id') is None or
          exact_int(actual['update_callable']['owner_id'],1)) and
         actual.get('configured_update_stream')==bound.get('configured_update_stream'),
         'selected actual implementation call certificate')
    for update in updates:
        gate(update.get('phase') in ('before','after') and
             exact_int(update.get('begin_ns'),1) and exact_int(update.get('end_ns'),1) and
             update['begin_ns']<=update['end_ns'] and
             update.get('caller_stream')==stream[1] and
             update.get('private_stream_relation')=='unresolved' and
             update.get('callable_id')==bound.get('callable_id') and
             update.get('callable')==bound.get('callable') and
             update.get('configured_update_stream')==bound.get('configured_update_stream'),
             'graph update caller/private-stream scope')
        if update['phase']=='before':
            gate(host[0]<=update['begin_ns']<=update['end_ns']<=host[1],
                 'pre-replay graph update Host placement')
        else:
            gate(host[2]<=update['begin_ns']<=update['end_ns']<=native[0]['entry_ns'],
                 'post-replay graph update Host placement')
def validate(rows):
    gate(len(rows)==40,'40 rank/cohort rows required')
    ids=set(); initial={}; remaining={}
    for d in rows:
        rank=d.get('rank');cohort=d.get('cohort')
        gate(type(rank) is int and 0<=rank<8 and type(cohort) is int and 1<=cohort<=5,'rank/cohort range')
        key=(rank,cohort);gate(key not in ids,'duplicate rank/cohort');ids.add(key)
        gate(d.get('schema')==1 and d.get('phase')==('diagnostic' if cohort==5 else 'warmup'),'schema/phase')
        gate(d.get('cycles',0)>64 and d.get('cycle_index')==d.get('start_cycle',-100)+64,'selected cycle64')
        gate(d.get('run_id') and isinstance(d['run_id'],str),'run ID')
        pins=d.get('setup',{}).get('source_pins',{})
        gate(pins.get('c10d',{}).get('c10d_sha256')==PINS['c10d'] and
             pins.get('c10d',{}).get('logger_sha256')==PINS['logger'] and
             pins.get('torch_npu',{}).get('sha256')==PINS['torch_npu'] and
             pins.get('cann',{}).get('libascendcl.so',{}).get('sha256')==PINS['ascendcl'] and
             pins.get('cann',{}).get('libruntime.so',{}).get('sha256')==PINS['runtime'],
             'loaded source/library pins')
        gate(tuple(d.get('markers',{}))==LABELS,'marker sequence')
        stream=d.get('stream')
        gate(isinstance(stream,(list,tuple)) and len(stream)==2,'stream identity')
        events=set()
        for name in LABELS:
            m=d['markers'][name];gate((m.get('device'),m.get('stream'))==(stream[0],stream[1]),'marker stream mismatch')
            ev=m.get('event');gate(ev and ev not in events,'duplicate/missing event ID')
            events.add(ev)
        validate_detail(d)
        sync=d.get('sync'); replay=d.get('replay')
        gate(sync and sync.get('begin_ns') and sync.get('end_ns') and sync['end_ns']>=sync['begin_ns'],'existing FULL sync')
        gate(replay and replay.get('entry_id')==sync.get('entry_id') and replay.get('capture_generation',0)>0,'replay generation/entry')
        gate(d.get('graph_output')==d.get('pre_gather')==replay.get('output_owner'),'graph output ownership')
        gate(d.get('flash_comm') is True and d.get('post_gather') and d.get('hidden'),'hidden gather')
        gate(len(d.get('hidden',[]))==len(d.get('native_calls',[])),'hidden native count')
        gate(all(h.get('stream')==stream and h.get('gather') and h.get('native_output_same_storage') is True and h.get('output') for h in d['hidden']),'hidden lineage')
        gate(all(n.get('return_ns') and n.get('group',{}).get('members')==list(range(8)) for n in d['native_calls']),'native TP return')
        counts=d.get('accepted_counts');gate(isinstance(counts,list) and len(counts)==d['cycles'],'accepted count shape')
        init=d.get('initial_output_counts');rem=d.get('remaining')
        gate(isinstance(init,list) and isinstance(rem,list) and len(init)==len(rem)==12,'frozen12 slots')
        gate(all(type(v) is int and v==0 for v in init) and
             all(type(v) is int and v>0 for v in rem) and
             all(a+b==1024 for a,b in zip(init,rem)),'initial+remaining=1024')
        gate(all(isinstance(row,list) and len(row)==len(init) and
                 all(type(v) is int and 0<=v<=8 for v in row) for row in counts),
             'accepted count shape/range')
        gate(all(sum(row[i] for row in counts)>=rem[i] for i in range(12)),
             'staged accepted counts short of remaining output')
        initial.setdefault(cohort,init);remaining.setdefault(cohort,rem)
        gate(initial[cohort]==init and remaining[cohort]==rem,'cross-rank request count mismatch')
        gate(d.get('request_ids_available') is False and d.get('request_ids') is None,
             'request ID availability declaration')
    gate(len({d['run_id'] for d in rows})==1,'run ID mismatch')
    gate(ids=={(r,c) for r in range(8) for c in range(1,6)},'cohort grid')
    return dict(run_id=rows[0]['run_id'],rows=40,events=200,scope='conditional source-side frontier; no finite Bound or TPS claim')
def native_tasks(nodes):
    gate(isinstance(nodes,list) and bool(nodes),'nonempty graph task list')
    ids=set();models=set();streams={}
    supported={'EVENT_RECORD','EVENT_RESET','EVENT_WAIT','NOTIFY_RECORD','NOTIFY_WAIT',
               'MEMCPY','MEMCPY_ASYNC','MEMSET'}
    for node in nodes:
        gate(isinstance(node,dict) and isinstance(node.get('name'),str) and
             bool(node['name'].strip()) and isinstance(node.get('args'),dict),
             'native task name/args')
        args=node['args'];model=args.get('Model Id');stream=args.get('Stream Id');task=args.get('Task Id')
        kind=args.get('Task Type')
        gate(all(type(x) is int and x>=0 for x in (model,stream,task)) and
             isinstance(kind,str) and bool(kind.strip()),'native task fields')
        key=(model,stream,task);gate(key not in ids,'duplicate native task identity');ids.add(key)
        if kind.startswith('KERNEL_'):
            gate(isinstance(args.get('Kernel Args'),str) and bool(args['Kernel Args'].strip()) and
                 type(args.get('Kernel Args Size')) is int and args['Kernel Args Size']>0 and
                 type(args.get('Numblocks')) is int and args['Numblocks']>0 and
                 type(args.get('Schem Mode')) is int and args['Schem Mode']>=0,
                 'kernel native args')
        else:gate(kind in supported,'unsupported native task type')
        models.add(model);streams[stream]=streams.get(stream,0)+1
    gate(len(models)==1 and bool(streams),'native model/stream set')
    return dict(native_model_ids=sorted(models),task_count=len(nodes),
                stream_ids=sorted(str(x) for x in streams),
                stream_task_counts={str(k):v for k,v in sorted(streams.items())})

def validate_dumps(root,rows):
    dump_dir=root/'graph_dump'
    files=sorted(dump_dir.glob('rank*_cohort5_acl_graph.json'))
    metas=sorted(dump_dir.glob('rank*_cohort5_acl_graph.meta.json'))
    gate(len(files)==len(metas)==8,'exact8 graph dumps/meta')
    by_key={(d['rank'],d['cohort']):d for d in rows}
    gate(all(by_key[r,c].get('debug_dump') is None for r in range(8) for c in range(1,5)),
         'warmup graph dump prohibited')
    for rank in range(8):
        graph=dump_dir/f'rank{rank}_cohort5_acl_graph.json'
        meta_path=dump_dir/f'rank{rank}_cohort5_acl_graph.meta.json'
        gate(graph in files and meta_path in metas,'rank graph dump missing')
        row=by_key[rank,5];meta=json.loads(meta_path.read_text());raw=graph.read_bytes()
        nodes=json.loads(raw)
        task=native_tasks(nodes)
        digest=hashlib.sha256(raw).hexdigest()
        gate(meta.get('run_id')==row['run_id'] and meta.get('rank')==rank and
             meta.get('cohort')==5 and meta.get('path')==str(graph) and
             meta.get('bytes')==len(raw)>0 and meta.get('sha256')==digest and
             meta.get('node_count')==len(nodes) and
             all(meta.get(k)==v for k,v in task.items()),
             'graph dump bytes/node metadata')
        replay=row['replay']
        gate((meta.get('entry_id'),meta.get('capture_generation'),meta.get('graph_id'))==
             (replay.get('entry_id'),replay.get('capture_generation'),replay.get('graph_id')) and
             meta.get('model_id')==replay.get('capture_scope',{}).get('bound_runtime_target_model_id') and
             meta.get('batch_descriptor')==replay.get('batch_descriptor') and
             meta.get('output_owner_sha256')==hashlib.sha256(json.dumps(row['graph_output'],sort_keys=True).encode()).hexdigest(),
             'graph dump exact selected producer/owner join')
        gate(source_cert(row['setup']['source_pins'].get('graph_debug_dump')) and
             meta.get('debug_dump_binding')==row['setup']['source_pins']['graph_debug_dump'] and
             meta.get('debug_dump_callable_id')==row['setup']['source_pins'].get('graph_debug_dump_callable_id') and
             exact_int(meta.get('debug_dump_callable_id'),1) and
             meta.get('debug_dump_owner_id')==replay.get('graph_id') and
             exact_int(meta.get('debug_dump_owner_id'),1) and
             meta.get('completion')=='post ordinary counts_cpu drain; graph state after further replays, not cycle64 dynamic parameters',
             'graph dump binding/completion')
        backend=meta.get('graph_update_backend',{})
        bound=row['setup']['graph_update_bound'];actual=row['actual_update']
        gate(backend.get('private_stream_relation')=='unresolved' and
             backend.get('configured_update_stream')==bound['configured_update_stream'] and
             backend.get('selected_actual_update')==actual and
             backend.get('attn_backend')==bound.get('backend') and
             backend.get('get_impl_cls')==bound.get('get_impl_cls') and
             callable_cert(backend.get('attn_backend'),'object_id') and
             callable_cert(backend.get('get_impl_cls'),'callable_id') and
             callable_cert(backend.get('impl'),'object_id') and
             callable_cert(backend.get('update_graph_params'),'callable_id') and
             backend['impl']['object_id']==actual['impl_id'] and
             {k:backend['update_graph_params'][k] for k in ('callable_id','owner_id')}==
                 actual['update_callable'],
             'selected/post-drain backend identity and source provenance')
        gate(row.get('debug_dump')==dict(path=str(graph),meta_path=str(meta_path),
              sha256=digest,bytes=len(raw),node_count=len(nodes)),
             'capture row graph dump join')
    return 8

def validate_root(root):
    run_uuid=(root/'run_uuid.txt').read_text().strip()
    gate(bool(run_uuid),'missing run UUID manifest')
    paths=sorted((root/'capture').glob('rank*_cohort*.json'))
    gate(len(paths)==40,'exact40 capture files')
    rows=[json.loads(p.read_text()) for p in paths]
    gate(all(p.name==f"rank{d['rank']}_cohort{d['cohort']}.json" for p,d in zip(paths,rows)),
         'capture file/rank/cohort mismatch')
    result=validate(rows)
    gate(result['run_id']==run_uuid,'run UUID join')
    dump_count=validate_dumps(root,rows)
    by_key={(d['rank'],d['cohort']):d for d in rows}
    for c in range(1,6):
        cohort=[by_key[r,c] for r in range(8)]
        gate(len({d['cycles'] for d in cohort})==1 and
             all(d['accepted_counts']==cohort[0]['accepted_counts'] and
                 d['initial_output_counts']==cohort[0]['initial_output_counts'] and
                 d['remaining']==cohort[0]['remaining'] for d in cohort),
             'all-rank cohort cycle/trajectory identity')
    for name,count in (('warmup48.json',48),('bench.json',12)):
        client=json.loads((root/name).read_text())
        requests=client['requests']
        gate(len(requests)==count and client['summary']['concurrency']==12 and
             client['summary']['max_tokens']==1024 and
             all(r['error'] is None and r['output_tokens']==1024 and
                 r['end']>=r['start'] for r in requests),
             'frozen client request/length contract')
        sweep=[]
        for r in requests:sweep.extend(((r['start'],1),(r['end'],-1)))
        active=peak=0
        for _,delta in sorted(sweep):active+=delta;peak=max(peak,active)
        gate(active==0 and peak==12,'client c12 sweep')
    gate(int((root/'server_post_count.txt').read_text().strip())==60 and
         int((root/'server_http200_count.txt').read_text().strip())==60,
         'exact60 HTTP200 POSTs')
    runtime_paths=sorted((root/'runtime').glob('rank*_cohort*.json'))
    gate(len(runtime_paths)==40,'exact40 Runtime reports')
    runtime={}
    for path in runtime_paths:
        d=json.loads(path.read_text());key=(d.get('rank'),d.get('cohort'))
        gate(key in by_key and key not in runtime and
             path.name==f'rank{key[0]}_cohort{key[1]}.json' and
             d.get('pass') is True and d.get('target_graph_mode')=='FULL' and
             d.get('oracle_target_calls_after_handoff')==0 and
             d.get('host_mirror_exact') is True and
             d.get('generated_output_counts')==[1024]*12 and
             d.get('staged_output_counts')==[
                 sum(row[i] for row in by_key[key]['accepted_counts'])
                 for i in range(12)] and
             d.get('overshoot_tokens')==sum(
                 d['staged_output_counts'][i]-1024 for i in range(12)) and
             d.get('cycles')==by_key[key]['cycles'] and
             len(d.get('req_ids',[]))==12 and len(set(d['req_ids']))==12,
             'Runtime FULL/ownership/output/request join')
        runtime[key]=d
    all_ids=set()
    for c in range(1,6):
        ids=runtime[0,c]['req_ids']
        gate(all(runtime[r,c]['req_ids']==ids for r in range(8)) and
             not all_ids.intersection(ids),'Runtime all-rank/cohort request IDs')
        all_ids.update(ids)
    gate(len(all_ids)==60,'60 distinct Runtime requests')
    return dict(valid=True,scope=result['scope'],run_id=run_uuid,slices=40,cohorts=5,graph_dumps=dump_count)
def final_admit(root):
    local=json.loads((root/'validation.json').read_text())
    gate(local.get('valid') is True,'local validation failed')
    status=dict(line.split('=',1) for line in (root/'cleanup_status.txt').read_text().splitlines())
    for key in ('run_exit','stop_exit','stop_verify_exit','restore_exit',
                'sha_exit','sha_compare_exit','final_exit'):
        gate(status.get(key)=='0','cleanup '+key)
    expected_sha=''.join(f'{o}  {p}\n' for _,p,o,_ in SOURCE_ROWS).encode()
    gate((root/'source_before.sha256').read_bytes()==expected_sha and
         (root/'source_after.sha256').read_bytes()==expected_sha,
         'six pinned source before/after SHA mismatch')
    source_check=json.loads((root/'source_check.json').read_text())
    install=json.loads((root/'install.json').read_text())
    restore=json.loads((root/'restore.json').read_text())
    manifest=json.loads((root/'patch_state/manifest.json').read_text())
    gate(all(hex64(h) for _,_,o,n in SOURCE_ROWS for h in (o,n)) and
         all(hex64(v) for v in PINS.values()),'pinned source digest format')
    gate(source_check.get('action')=='check' and
         source_check.get('files')==SOURCE_MANIFEST and
         install.get('action')=='install' and install.get('files')==SOURCE_MANIFEST and
         restore.get('action')=='restore' and restore.get('files')==SOURCE_MANIFEST and
         manifest==SOURCE_MANIFEST,'six-source check/install/restore provenance')
    gate(validate_root(root)['valid'],'post-cleanup acquisition changed')
    return dict(valid=True,scope='post-cleanup conditional Target frontier diagnostic only')
def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--final-only',action='store_true');a=p.parse_args()
    try:result=final_admit(a.root) if a.final_only else validate_root(a.root)
    except Exception as exc:result=dict(valid=False,error=f'{type(exc).__name__}: {exc}')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    if not result['valid']:raise SystemExit(1)
if __name__=='__main__':main()
