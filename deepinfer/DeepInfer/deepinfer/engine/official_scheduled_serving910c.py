"""Original C++ Scheduler/Worker serving bridge for verified hot fixed-four batches.

Short/cold/internal jobs use NativeServingRunner's existing path. No model
precision or sampling algorithm is replaced here; original commits own tokens.
"""
from pathlib import Path
from types import SimpleNamespace as NS
import json,os,time
import torch
import torch.distributed as dist


def initialize(runner):
    from .zos910c.scheduled_worker910c import create_scheduler_groups
    runner._scheduled_groups=create_scheduler_groups(runner.tp_size)


def try_run(runner,job,directory):
    if os.environ.get('DEEPINFER_910C_ZOS_CONTINUOUS_ADMISSION')=='1':
        from .continuous_scheduled_serving910c import try_run as try_continuous
        return try_continuous(runner,job,directory)
    requests=job['requests']
    if (len(requests)!=runner.slots*runner.dp_size or runner.slots!=4
            or any(len(r['token_ids'])!=32768 or r['max_tokens']!=1024 or not r.get('ignore_eos',False) for r in requests)):
        return False
    owned=[(i,r) for i,r in enumerate(requests) if i%runner.dp_size==runner.dp_rank]
    cached=all(any(tuple(r['token_ids'][:p]) in runner.prefix_cache for p in runner.prefix_points) for _,r in owned)
    if runner._global_busy(not cached):
        return False
    if not hasattr(runner,'_scheduled_groups'):
        raise RuntimeError('original Scheduler groups were not initialized before readiness')
    curve=runner.model.dspark_graphs.curve
    if not curve:
        raise RuntimeError('original Scheduler requires the actual warmed decode performance curve')
    directory=Path(directory);started=time.monotonic()
    runner._scheduled_owned=owned
    runner._scheduled_official_yoco=os.environ.get('DEEPINFER_910C_YOCO_OFFICIAL_PREFILL')=='1'
    if runner._scheduled_official_yoco:
        from .official_yoco_metadata910c import OfficialYocoMetadata,configure_consumer
        configure_consumer(runner.model)
        if runner.official_yoco_metadata is None:runner.official_yoco_metadata=OfficialYocoMetadata(cp_size=runner.tp_size)
        runner.official_yoco_metadata.begin_job([r for _,r in owned])
    file=(directory/f'events-dp{runner.dp_rank}.jsonl').open('a',buffering=1) if runner.tp_rank==0 else None
    def emit(event):
        event.setdefault('elapsed',time.monotonic()-started)
        if file is not None:file.write(json.dumps(event,separators=(',',':'))+'\n')
    spec={'initial':[dict(r,seq_id=1000+i) for i,r in enumerate(requests)],'eos_token_id':job['eos_token_id'],
          'scratch_token':requests[0]['token_ids'][0],'curve':[list(x) for x in curve],
          'max_steps':max(r['max_tokens'] for r in requests)+64,'replacement':None}
    try:
        actual,hits,projected,profile=execute(runner,runner.official_yoco_metadata.api if runner._scheduled_official_yoco else __import__('deepinfer')._C.engine,directory,spec,emit=emit)
        generated=[actual[str(1000+i)] for i,_ in owned]
        cached=[hits[1000+i] for i,_ in owned]
        for (i,r),tokens,hit in zip(owned,generated,cached,strict=True):
            if len(tokens)!=r['max_tokens']:raise RuntimeError('original Scheduler output budget incomplete')
            emit({'type':'done','index':i,'prompt_tokens':len(r['token_ids']),'cached_tokens':hit,
                  'output_tokens':len(tokens),'token_ids':tokens,'finish_reason':'length'})
        if runner.tp_rank==0:
            stats=dict(runner.model.runtime_stats(),zos_official_scheduler_used=True,
                       zos_official_scheduler_steps=len(projected),zos_official_scheduler_prompt_steps=sum(x['prompt'] for x in projected),
                       zos_profile_step_count=len(profile['step_wall']),zos_profile_device_count=len(profile['device_events']),
                       zos_async_sample_inputs=profile['sample_input_staging'])
            result={'state':'PASS','dp_rank':runner.dp_rank,'internal_replica_work':False,
                    'request_indices':[i for i,_ in owned],'cached_tokens':cached,'generated_token_ids':generated,
                    'seconds':time.monotonic()-started,'runtime_stats':stats,'peak_npu_allocated_bytes':torch.npu.max_memory_allocated()}
            (directory/f'result-dp{runner.dp_rank}.json').write_text(json.dumps(result,indent=2))
    finally:
        if runner._scheduled_official_yoco:runner.official_yoco_metadata.end_job()
        runner._scheduled_official_yoco=False
        if file is not None:file.close()
    dist.barrier()
    return True


def execute(runner, engine, directory, fixture, *, emit, trace=None):
    import numpy as np
    import torch
    import torch.distributed as dist

    from .zos910c.fixed_run_ledger910c import FixedRunLedger
    from .zos910c.native_scheduled_runner910c import ScheduledNativeModelRunner
    from .zos910c.native_scheduled_target910c import ScheduledNativeTarget
    from .zos910c.official_phase_core910c import NativeEventRegistry, create_worker
    from .zos910c.scheduled_admission910c import ScheduledAdmission
    from .zos910c.scheduled_prefill910c import initialize_prompt_cursor
    from .zos910c.scheduled_worker910c import ScheduledDriver, create_scheduler_groups, run_scheduled_step
    from .zos910c.zos_slot_handoff910c import SlotBinding, SlotHandoff, validate_prefix

    model = runner.model
    dp_group, host_tp, control = runner._scheduled_groups
    is_driver = runner.tp_rank == 0
    eos = fixture["eos_token_id"]
    ledger = FixedRunLedger(runner, engine, eos=eos)
    target = ScheduledNativeTarget(runner, engine, debug_c2=False,
        async_sample_inputs=os.environ.get('DEEPINFER_910C_ZOS_ASYNC_SAMPLE_INPUTS')=='1',
        multigraph=os.environ.get('DEEPINFER_910C_ZOS_OFFICIAL_MULTIGRAPH')=='1')
    groups = host_tp, control
    sequences, bindings, hits = {}, {}, {}
    mapping = [int(x) for x in model.histories[0].token_map]
    if is_driver:
        config = NS(max_model_len=model.max_tokens-12, runtime_max_model_len=model.max_tokens,
                    max_num_batched_tokens=model.chunk_size*4, max_num_seqs=4,
                    stop_preempted_request=True, chunked_prefill=True, engine_type=0,
                    prefill_attempt_interval=2, prefill_any_attempt_interval=3)
        scheduler = engine.Scheduler(128, engine.BlockManager(2048), engine.BlockManager(16),
            eos, dp_group, config, 0, 0, 4*model.max_tokens, None, mapping, 8, model.cfg["vocab_size"], False)
        scheduler.decode_perf_curve = [tuple(x) for x in fixture["curve"]]
        driver = ScheduledDriver(engine, scheduler, engine.SequenceMetadataPool(), nextn=5,
                                 on_commit=lambda *_: None, on_finished=lambda *_: None)
    else:
        driver = None
    registry = NativeEventRegistry()
    seam = ScheduledNativeModelRunner(runner, target, ledger, driver=driver, groups=groups,
                                     registry=registry, scratch_token=fixture["scratch_token"])
    callbacks = NS(post_process_fn=seam.post_already_committed,
                   pre_sampler_sync_fn=None if driver is None else driver.pre_sampler_sync)
    worker = create_worker(seam, callbacks, NS(DEEPINFER_ENABLE_MTP_VARLEN=True,
        DEEPINFER_ZERO_OVERHEAD_SCHEDULING=True), registry, is_driver=is_driver, backend=torch)
    seam.worker = worker
    seam.trace = trace

    def cached(request):
        ids = request["token_ids"]
        snapshot = next((runner.prefix_cache[tuple(ids[:point])] for point in reversed(runner.prefix_points)
                         if len(ids)>point and tuple(ids[:point]) in runner.prefix_cache), None)
        if snapshot is None:
            raise RuntimeError("isolated fixed-graph gate requires a real warm prefix")
        validate_prefix(model, ids, snapshot)
        return snapshot

    def add(slot, binding, request, position):
        bindings[slot] = binding
        hits[binding.seq_id] = position
        ledger.register(slot, binding, request, hit=position)
        if driver is None:
            return
        ids = np.asarray(request["token_ids"], dtype=np.int32)
        compressed = np.asarray([mapping[i] for i in ids], dtype=np.int32)
        sampling = NS(temperature=0., top_p=1., top_k=-1, seq_seed=0, top_logprobs=0,
            prompt_logprobs=False, first_n_token_no_eos=0, eos_id=eos,
            ignore_eos=request["ignore_eos"], max_tokens=request["max_tokens"])
        seq = engine.Sequence(binding.seq_id, ids, compressed, None, 128,
                              str(binding.seq_id), "isolated", sampling, 0., None)
        initialize_prompt_cursor(seq, position=position, chunk_size=model.chunk_size, prefix_points=runner.prefix_points)
        sequences[binding.seq_id] = seq
        scheduler.add_sequence(seq)

    model.reset()
    for slot, request in enumerate(fixture["initial"][runner.dp_rank::runner.dp_size]):
        snapshot = cached(request)
        model.restore_prefix(snapshot, slot)
        add(slot, SlotBinding(request["seq_id"], 0), request, model.next_positions[slot])
    replacement = fixture.get("replacement")
    replace_here = replacement is not None and replacement["dp"] == runner.dp_rank
    if replace_here:
        slot = replacement["slot"]
        old = bindings[slot]
        gate = SlotHandoff(model, worker, slot, old, slots_per_dp=4)
        seam.admission = ledger.admission = ScheduledAdmission(runner, worker, {slot: gate}, close_pipeline=target.close)
    admitted = False
    projected = []
    step_wall=[]
    runner._scheduled_hits=[hits[request['seq_id']] for request in fixture['initial'][runner.dp_rank::runner.dp_size]]
    publish=ledger.publish_committed
    def publish_and_emit(step, rows, stop):
        offsets={seq_id:len(ledger.results[seq_id]) for seq_id,_,_,_ in rows}
        publish(step,rows,stop)
        for seq_id,tokens,_,_ in rows:
            for offset,token in enumerate(tokens,offsets[seq_id]):
                emit({'type':'token','index':seq_id-1000,'token_id':token,'step':offset})
    ledger.publish_committed=publish_and_emit
    for slot,request in enumerate(fixture['initial'][runner.dp_rank::runner.dp_size]):
        emit({'type':'start','index':request['seq_id']-1000,'cached_tokens':hits[request['seq_id']]})

    for number in range(fixture["max_steps"]):
        wall_start=time.monotonic_ns()
        busy = torch.tensor([int(is_driver and scheduler.has_unfinished_seqs())], dtype=torch.int32)
        dist.all_reduce(busy, op=dist.ReduceOp.MAX, group=control)
        if not busy.item():
            break
        positions = seam.admission.scheduled_positions() if seam.admission else model.next_positions
        args = dict(bindings=bindings, sequences=sequences, positions=positions,
                    slots=4, chunk_size=model.chunk_size)
        run_scheduled_step(worker, seam, number, driver=driver, tp_group=host_tp,
                           control_group=control, root=runner.root, schedule=args)
        ledger.finish_step()
        step = seam.step
        step_wall.append((number,step.metadata.is_prompt,wall_start,time.monotonic_ns()))
        pending = worker.prev_step_state
        projected.append(dict(step=number, prompt=step.metadata.is_prompt,
                              rows=[(r.seq_id, r.slot, r.epoch, r.q, r.scheduled_k,
                                     r.mtp_read_slot, r.mtp_write_slot) for r in step.metadata.rows],
                              sample_q=None if pending is None else pending.sampler_output.target.logical_q,
                              seed_drafts=step.seed_drafts))
        if replace_here and not admitted and old.seq_id in seam.step.retired_ids:
            request = replacement["request"]
            snapshot = cached(request)
            reserved = seam.admission.reserve(slot, old, request["token_ids"], snapshot,
                                              seq_id=request["seq_id"])
            add(slot, reserved.binding, request, reserved.start)
            admitted = True
    else:
        raise RuntimeError("isolated scheduled fixture exceeded its step limit")
    worker.flush_inflight_reqs()
    if driver is not None:
        driver.release_metadata()
    sample_input_stats = target.sample_input_stats()
    target.close()
    torch.npu.synchronize()
    if replace_here and not admitted:
        raise RuntimeError("fixture did not exercise a physical slot replacement")
    actual = {str(k): v for k,v in ledger.results.items()}
    differences = {}
    profile=dict(step_wall=step_wall,phase_marks=seam.phase_marks,
                 cpu_timer=worker._local_cpu_timer.samples,device_events=seam.device_timings,
                 initial_curve=fixture['curve'],fixed_physical_frame=24,
                 sample_input_staging=sample_input_stats,
                 timing_policy='reuse already-read original event times;no extra GPU event or synchronization')
    rank = dist.get_rank()
    path = directory / f"scheduled-rank{rank}.json"
    scratch = path.with_suffix(".json.tmp")
    with scratch.open("x") as f:
        json.dump(dict(rank=rank, state="TOKEN_MISMATCH" if differences else "PASS",
                       actual=actual, hits=hits, trace=ledger.trace, projected=projected,profile=profile,
                       first_differences=differences,
                       local_slot_drains=seam.admission.drains if seam.admission else [],
                       kv_positions=model.next_positions[:4]), f)
    scratch.replace(path)
    # Archive all ranks before any mismatching rank raises into torchrun.
    dist.barrier()
    return actual,hits,projected,profile
