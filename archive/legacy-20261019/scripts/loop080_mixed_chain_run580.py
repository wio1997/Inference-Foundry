"""Run580 terminal fixed-work GMM × current-order TP8 HCCL resource probe."""
from __future__ import annotations
import hashlib,json,statistics,time
from pathlib import Path
import torch
import torch.distributed as dist
import torch_npu

ROOT=Path('/data/wio/Inference_Foundry')
LEDGER=ROOT/'evidence/20260927_loop077_bound/run378/ledger.json'
LEDGER_SHA='c5bbe55c0853b011188cd0f2e3e6e0a673951b8e95c5b6fd625ca80e45827211'
DTYPES={'BFP16':torch.bfloat16,'FP32':torch.float32}


def run(gmm_graph,gmm_stream,gmm_outputs,valid_rows,gmm_expected,
        rank,out,tp_group,wait_all):
    pg=tp_group.device_group
    if dist.get_world_size(group=pg)!=8 or dist.get_rank(group=pg)!=rank:
        raise RuntimeError('Run580 TP8 group identity drift')
    raw=LEDGER.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=LEDGER_SHA:
        raise RuntimeError('Run580 ordered HCCL ledger SHA drift')
    ordered=json.loads(raw)['ordered_tasks']
    if len(ordered)!=265:
        raise RuntimeError('Run580 HCCL chain length drift')
    kinds={kind:sum(row['kind']==kind for row in ordered)
           for kind in ('hcom_allGather','hcom_reduceScatter','hcom_alltoall')}
    if kinds!={'hcom_allGather':135,'hcom_reduceScatter':87,'hcom_alltoall':43}:
        raise RuntimeError('Run580 HCCL chain kind counts drift')
    buffers=[]
    for i,row in enumerate(ordered):
        count=row['elements'];dtype=DTYPES[row['dtype']]
        value=rank+1+i%10
        if row['kind']=='hcom_allGather':
            src=torch.full((count,),value,dtype=dtype,device=f'npu:{rank}')
            dst=torch.empty((count*8,),dtype=dtype,device=f'npu:{rank}')
        elif row['kind']=='hcom_reduceScatter':
            src=torch.full((count*8,),value,dtype=dtype,device=f'npu:{rank}')
            dst=torch.empty((count,),dtype=dtype,device=f'npu:{rank}')
        elif row['kind']=='hcom_alltoall':
            src=torch.full((count*8,),value,dtype=dtype,device=f'npu:{rank}')
            dst=torch.empty_like(src)
        else:raise RuntimeError('Run580 unknown collective')
        buffers.append((src,dst))
    free, total=torch.npu.mem_get_info()
    if free<3*1024**3:
        raise RuntimeError(f'Run580 scratch HBM guard after HCCL buffers: {free}')

    def chain():
        for row,(src,dst) in zip(ordered,buffers):
            kind=row['kind']
            if kind=='hcom_allGather':
                dist.all_gather_into_tensor(dst,src,group=pg)
            elif kind=='hcom_reduceScatter':
                dist.reduce_scatter_tensor(dst,src,group=pg)
            else:
                dist.all_to_all_single(dst,src,group=pg)

    # Eager HCCL initialization and all8 agreement precede Graph capture.
    for _ in range(2):chain()
    torch.npu.synchronize()
    dist.barrier(group=pg)
    torch.npu.synchronize()
    comm_stream=torch.npu.Stream()
    join_stream=torch.npu.Stream()
    hgraph=torch.npu.NPUGraph()
    with torch.npu.graph(hgraph,stream=comm_stream):
        chain()
    torch.npu.synchronize()
    for _ in range(3):
        with torch.npu.stream(comm_stream):hgraph.replay()
    torch.npu.synchronize()
    wait_all(out,rank,'graphs_captured')

    # Fresh generation plus poisoned outputs: join consumer is submitted before
    # any all-device sync, then only its end event is synchronized.
    checked=[]
    for generation in (0,1,2):
        expected_peers=[(torch.arange(1,9,device=f'npu:{rank}',dtype=DTYPES[row['dtype']])+
                         (i%10)+2*generation).view(8,1)
                        if row['kind']!='hcom_reduceScatter' else None
                        for i,row in enumerate(ordered)]
        torch.npu.synchronize()  # expected tensors ready before fresh replay
        with torch.npu.stream(comm_stream):
            for i,((src,dst),row) in enumerate(zip(buffers,ordered)):
                src.fill_(rank+1+i%10+2*generation)
                dst.fill_(-17)
            hgraph.replay()
            branch_done=torch.npu.Event(enable_timing=True)
            branch_done.record()
        with torch.npu.stream(join_stream):
            join_stream.wait_event(branch_done)
            # Tail first makes a missing completion edge easier to detect.
            tail=ordered[-1];tail_dst=buffers[-1][1]
            tail_mismatch=(tail_dst.view(8,tail['elements'])[:,0]!=expected_peers[-1][:,0]).to(torch.int32).sum()
            parts=[tail_mismatch]
            for i,(row,(_,dst)) in enumerate(zip(ordered,buffers)):
                if row['kind']=='hcom_reduceScatter':
                    expected=36+8*(i%10+2*generation)
                    mismatch=(dst!=expected).to(torch.int32).sum()
                else:
                    mismatch=(dst.view(8,row['elements'])!=expected_peers[i]).to(torch.int32).sum()
                parts.append(mismatch)
            total_mismatch=torch.stack(parts).sum()
            consumer_end=torch.npu.Event(enable_timing=True)
            consumer_end.record()
        consumer_end.synchronize()
        bad=int(total_mismatch.item())
        if bad!=0:raise RuntimeError(f'Run580 rank{rank} generation{generation} HCCL mismatches={bad}')
        checked.append({'generation':generation,'all265_outputs_exact':True,'mismatch':bad})
        dist.barrier(group=pg)
    torch.npu.synchronize()
    wait_all(out,rank,'fresh_validated')

    # Fresh mixed execution must prove both branches after their join. All
    # consumers are queued before the sole end-event synchronization.
    generation=3
    expected_peers=[(torch.arange(1,9,device=f'npu:{rank}',dtype=DTYPES[row['dtype']])+
                     (i%10)+2*generation).view(8,1)
                    if row['kind']!='hcom_reduceScatter' else None
                    for i,row in enumerate(ordered)]
    with torch.npu.stream(gmm_stream):
        for tensor in gmm_outputs:tensor.fill_(-17)
    with torch.npu.stream(comm_stream):
        for i,((src,dst),row) in enumerate(zip(buffers,ordered)):
            src.fill_(rank+1+i%10+2*generation)
            dst.fill_(-17)
    torch.npu.synchronize()  # fresh inputs and poison ready before common start
    mixed_start=torch.npu.Event(enable_timing=True)
    mixed_gdone=torch.npu.Event(enable_timing=True)
    mixed_hdone=torch.npu.Event(enable_timing=True)
    with torch.npu.stream(join_stream):mixed_start.record()
    with torch.npu.stream(gmm_stream):
        gmm_stream.wait_event(mixed_start)
        gmm_graph.replay()
        mixed_gdone.record()
    with torch.npu.stream(comm_stream):
        comm_stream.wait_event(mixed_start)
        hgraph.replay()
        mixed_hdone.record()
    with torch.npu.stream(join_stream):
        join_stream.wait_event(mixed_gdone)
        join_stream.wait_event(mixed_hdone)
        gmm_parts=[(~torch.isclose(actual[:n],expected,rtol=0.01,atol=0.05)).to(torch.int32).sum()
                   for actual,n,expected in zip(gmm_outputs,valid_rows,gmm_expected)]
        hccl_parts=[]
        for i,(row,(_,dst)) in enumerate(zip(ordered,buffers)):
            if row['kind']=='hcom_reduceScatter':
                expected=36+8*(i%10+2*generation)
                mismatch=(dst!=expected).to(torch.int32).sum()
            else:
                mismatch=(dst.view(8,row['elements'])!=expected_peers[i]).to(torch.int32).sum()
            hccl_parts.append(mismatch)
        gmm_bad=torch.stack(gmm_parts).sum()
        hccl_bad=torch.stack(hccl_parts).sum()
        mixed_end=torch.npu.Event(enable_timing=True)
        mixed_end.record()
    mixed_end.synchronize()
    mixed_correctness={'generation':generation,
                       'gmm_valid_rows_close':int(gmm_bad.item())==0,
                       'gmm_mismatch':int(gmm_bad.item()),
                       'all265_outputs_exact':int(hccl_bad.item())==0,
                       'hccl_mismatch':int(hccl_bad.item()),
                       'joint_ms':mixed_start.elapsed_time(mixed_end),
                       'gmm_branch_done_ms':mixed_start.elapsed_time(mixed_gdone),
                       'hccl_branch_done_ms':mixed_start.elapsed_time(mixed_hdone)}
    if not mixed_correctness['gmm_valid_rows_close'] or not mixed_correctness['all265_outputs_exact']:
        raise RuntimeError(f'Run580 rank{rank} fresh mixed correctness {mixed_correctness}')
    dist.barrier(group=pg)
    torch.npu.synchronize()
    wait_all(out,rank,'mixed_validated')

    def one_sample(arm,mark_submits=False):
        start=torch.npu.Event(enable_timing=True)
        gdone=torch.npu.Event(enable_timing=True)
        hdone=torch.npu.Event(enable_timing=True)
        end=torch.npu.Event(enable_timing=True)
        host_start=time.perf_counter_ns()
        with torch.npu.stream(join_stream):start.record()
        if arm in ('gmm','serial','concurrent'):
            with torch.npu.stream(gmm_stream):
                gmm_stream.wait_event(start)
                if mark_submits:
                    with record_function('run580_gmm_graph_replay_submit'):
                        gmm_graph.replay()
                else:gmm_graph.replay()
                gdone.record()
        if arm in ('hccl','serial','concurrent'):
            with torch.npu.stream(comm_stream):
                comm_stream.wait_event(start)
                if arm=='serial':comm_stream.wait_event(gdone)
                if mark_submits:
                    with record_function('run580_hccl_graph_replay_submit'):
                        hgraph.replay()
                else:hgraph.replay()
                hdone.record()
        with torch.npu.stream(join_stream):
            if arm in ('gmm','serial','concurrent'):join_stream.wait_event(gdone)
            if arm in ('hccl','serial','concurrent'):join_stream.wait_event(hdone)
            end.record()
        end.synchronize()
        host_end=time.perf_counter_ns()
        result={'joint_ms':start.elapsed_time(end),
                'host_submit_wait_ms':(host_end-host_start)/1e6}
        if arm in ('gmm','serial','concurrent'):
            result['gmm_branch_done_ms']=start.elapsed_time(gdone)
        if arm in ('hccl','serial','concurrent'):
            result['hccl_branch_done_ms']=start.elapsed_time(hdone)
        if any(v<=0 or v>1000 for v in result.values()):
            raise RuntimeError(f'Run580 invalid event sample {arm}: {result}')
        return result

    # Warm every arm, then balanced forward/reverse blocks in one model load.
    for arm in ('gmm','hccl','serial','concurrent'):
        for _ in range(2):one_sample(arm)
    torch.npu.synchronize()
    wait_all(out,rank,'timed')
    samples={key:[] for key in ('gmm','hccl','serial','concurrent')}
    order=('gmm','hccl','serial','concurrent','concurrent','serial','hccl','gmm')
    host_start=time.perf_counter_ns()
    for arm in order:
        dist.barrier(group=pg)
        torch.npu.synchronize()
        for _ in range(10):samples[arm].append(one_sample(arm))
    host_end=time.perf_counter_ns()
    medians={key:statistics.median(x['joint_ms'] for x in values)
             for key,values in samples.items()}
    checkpoint={'status':'unprofiled_four_arm_checkpoint',
                'run_tag':'LOOP080-RUN580-B','rank':rank,
                'arm_order':list(order),'samples':samples,
                'rank_median_joint_ms':medians,
                'host_start_ns':host_start,'host_end_ns':host_end}
    (out/f'rank{rank}_unprofiled_checkpoint.json').write_text(
        json.dumps(checkpoint,indent=2)+'\n')
    wait_all(out,rank,'unprofiled_checkpoint')
    # Mechanism trace only. The unprofiled four-arm samples above remain the
    # service-time reference; disable automatic online CANN analysis after
    # Run578's parser failure and analyse the raw timeline after full stop.
    from torch_npu.profiler import (profile, ProfilerActivity, ProfilerLevel,
        _ExperimentalConfig, tensorboard_trace_handler)
    from torch.profiler import record_function
    trace_dir=out/'profile'/f'rank{rank}'
    trace_dir.mkdir(parents=True,exist_ok=False)
    torch.npu.synchronize()
    wait_all(out,rank,'profile')
    profiled_steps=[]
    experimental=_ExperimentalConfig(profiler_level=ProfilerLevel.Level0)
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.NPU],
                 schedule=torch_npu.profiler.schedule(wait=0,warmup=1,active=2,repeat=1),
                 on_trace_ready=tensorboard_trace_handler(str(trace_dir),analyse_flag=False),
                 experimental_config=experimental) as prof:
        for label,arm in (('warmup_serial','serial'),
                          ('active_serial','serial'),
                          ('active_concurrent','concurrent')):
            dist.barrier(group=pg)
            torch.npu.synchronize()
            with record_function('run580_'+label):
                measurement=one_sample(arm,mark_submits=True)
            profiled_steps.append({'label':label,'arm':arm,'event_sample':measurement})
            prof.step()
    torch.npu.synchronize()
    raw_files=[p for p in trace_dir.rglob('*') if p.is_file()]
    raw_bytes=sum(p.stat().st_size for p in raw_files)
    if len(raw_files)<3 or raw_bytes<1000:
        raise RuntimeError(f'Run580 rank{rank} raw profiler missing: {len(raw_files)} files, {raw_bytes} bytes')
    wait_all(out,rank,'profiled')
    free_after,_=torch.npu.mem_get_info()
    return {'status':'all8_terminal_mixed_resource_service',
            'ledger_sha256':LEDGER_SHA,'collective_count':265,'kinds':kinds,
            'distinct_input_output_buffers_per_collective':True,
            'collective_correctness':checked,
            'mixed_correctness':mixed_correctness,
            'gmm_graph_stream':'private','hccl_graph_stream':'private',
            'joint_event_stream':'private',
            'arms':'gmm,hccl,serial_gmm_then_hccl,concurrent_dual_graph',
            'arm_order':list(order),'samples':samples,
            'rank_median_joint_ms':medians,
            'host_start_ns':host_start,'host_end_ns':host_end,
            'profile_status':'raw_timeline_only_no_online_analysis',
            'profile_steps':profiled_steps,
            'profile_raw_file_count':len(raw_files),
            'profile_raw_bytes':raw_bytes,
            'profile_root':str(trace_dir.relative_to(ROOT)),
            'free_hbm_after_hccl_buffers_bytes':free,
            'free_hbm_after_samples_bytes':free_after,
            'total_hbm_bytes':total,
            'limits':['Independent already-ready GMM and HCCL branches; no production producer/consumer dependency.',
                      'Private synthetic activation/collective inputs; no formal W0 or E2E TPS.',
                      'Rank-local event durations are not calibrated cross-rank makespan.',
                      'Attained mixed Engineering service, not strict C+ or Product Bound.']}
