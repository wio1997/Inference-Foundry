"""Run401 sparse diagnostic. No device synchronization except cohort-end export.
Sequential fixed-cohort process only. Event timing semantics require external preflight.
"""
import json
import os
from pathlib import Path
import time

ENV = 'EXTREME_COUNT_MARKER_DIR'
D = None
LABELS = ('R_BEGIN_64', 'R_DONE_64', 'W_PRE_65', 'W_POST_65')
HOST = ('commit_entry', 'sync_pre', 'sync_post', 'count_add_pre', 'count_add_post',
        'seq_add_pre', 'seq_add_post', 'mirrors_done', 'launch64', 'launch65',
        'progress64', 'progress65', 'clone_pre', 'clone_post', 'downstream_pre', 'downstream_post')


def storage(t):
    return dict(ptr=t.data_ptr(), storage_ptr=t.untyped_storage().data_ptr(),
                storage_bytes=t.untyped_storage().nbytes(), offset=t.storage_offset(),
                shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype),
                device=str(t.device), nbytes=t.numel()*t.element_size())


def start(serving):
    global D
    if not os.getenv(ENV):
        D = None
        return
    import torch
    import torch_npu
    runtime, adapter = serving.runtime, serving.runtime.proposer
    run_id = os.environ['EXTREME_COUNT_MARKER_RUN_ID']
    if os.getenv('EXTREME_BOUND_EVENT_DIR') or os.getenv('EXTREME_BOUND_TOKEN_CAPTURE_DIR'):
        raise RuntimeError('Run401 excludes other event/route capture conditions')
    if runtime._profile_dag or runtime._diagnose or runtime._cycle_profiler:
        raise RuntimeError('Run401 excludes competing runtime diagnostics/profilers')
    if adapter._host_count_copy is None or not adapter._host_count_copy.is_pinned():
        raise RuntimeError('Run401 requires original pinned host destination')
    seq = adapter._unique_host_mirrors(adapter.common_attn_metadata,
            ('seq_lens_cpu', '_seq_lens_cpu', 'seq_lens_cpu_upper_bound'))
    computed = adapter._unique_host_mirrors(adapter.common_attn_metadata,
            ('num_computed_tokens_cpu', '_num_computed_tokens_cpu'))
    D = dict(run_id=run_id, torch_version=torch.__version__, torch_npu_version=torch_npu.__version__, start_cycle=runtime.state.cycle_index, entry=None,
        serving=serving, adapter=adapter, generation=None, commit_generation=None,
        events={k: torch.npu.Event(enable_timing=True) for k in LABELS},
        anchor=torch.npu.Event(enable_timing=True),
        recorded={k: False for k in LABELS},
        device_records={k: dict(event=None, stream=None, device=None, entry=None, generation=64, storage=None) for k in LABELS},
        host={k: dict(ns=None, generation=None, entry=None, ptr=None) for k in HOST}, parking=False, reason=None, commit_pending=None,
        mirror_ptrs=[x.data_ptr() for x in seq], derived_ptr=None,
        seq_mirrors=[storage(x) for x in seq], computed_mirrors=[storage(x) for x in computed],
        source=storage(runtime.state.num_sampled), destination=storage(adapter._host_count_copy),
        production_event=id(adapter._host_copy_event),
        copy_stream=int(adapter._host_copy_stream.npu_stream),
        schedule_mode=runtime._schedule_mode, draft_graph=adapter.proposer.use_cuda_graph,
        first_clone=None, downstream=None, generation_error=None)
    if len({id(x) for x in D['events'].values()} | {id(D['anchor']), D['production_event']}) != 6:
        raise RuntimeError('Run401 event identities must be distinct')


def step(runtime):
    if D is not None:
        D['entry'] = runtime.state.cycle_index - D['start_cycle']
        if D['entry'] in (64, 65) and any(D['serving']._parked):
            D['parking'] = True


def event(label, tensor):
    if D is None or D['entry'] != (64 if label.startswith('R_') else 65):
        return
    import torch
    if D['recorded'][label]:
        raise RuntimeError('Run401 diagnostic event reused')
    stream = torch.npu.current_stream(tensor.device)
    handle = getattr(stream, 'npu_stream', None)
    if handle is None:
        raise RuntimeError('Run401 cannot resolve actual stream identity')
    D['events'][label].record()
    D['recorded'][label] = True
    row = D['device_records'][label]
    row['event'] = id(D['events'][label])
    row['stream'] = int(handle)
    row['device'] = str(tensor.device)
    row['entry'] = D['entry']
    row['storage'] = D['source']
    if tensor.data_ptr() != D['source']['ptr']:
        D['generation_error'] = 'source/overwrite tensor pointer changed'


def host(label, mirror=None):
    if D is None or D['entry'] != 65 or D['commit_generation'] != 64:
        return
    row = D['host'][label]
    if row['ns'] is None:
        row['ns'] = time.monotonic_ns()
        row['generation'] = 64
        row['entry'] = 65
        row['ptr'] = mirror.data_ptr() if mirror is not None else None


def launch(adapter, counts):
    if D is None:
        return
    if D['entry'] in (64, 65):
        label = f"launch{D['entry']}"
        row = D['host'][label]
        row['ns'] = time.monotonic_ns()
        row['generation'] = D['entry']
        row['entry'] = D['entry']
        if adapter._host_count_copy.data_ptr() != D['destination']['ptr']:
            D['generation_error'] = 'pinned destination storage changed'
        if counts.data_ptr() != D['source']['ptr']:
            D['generation_error'] = 'copy source storage changed'
        if D['entry'] == 65 and D['host']['mirrors_done']['ns'] is None:
            D['generation_error'] = 'launch65 before generation64 consumption'
    D['generation'] = D['entry']


def commit(adapter, reason):
    if D is None:
        return
    if reason == 'park_completed_slots' and D['entry'] in (64, 65):
        D['parking'] = True
    D['commit_generation'] = D['generation'] if adapter._host_copy_pending else None
    if D['entry'] == 65 and D['commit_generation'] == 64:
        D['reason'] = reason
        D['commit_pending'] = bool(adapter._host_copy_pending)
        host('commit_entry')


def progress():
    if D is not None and D['entry'] in (64, 65):
        row = D['host'][f"progress{D['entry']}"]
        row['ns'] = time.monotonic_ns()
        row['generation'] = D['commit_generation']
        row['entry'] = D['entry']


def _matched_lineage(tensor):
    if tensor.device.type != 'cpu':
        return None
    if tensor.data_ptr() in D['mirror_ptrs']:
        return 'direct_mirror'
    if tensor.data_ptr() == D['derived_ptr']:
        return 'tracked_clone'
    return None


def clone_before(tensor, site):
    if D is None or D['entry'] != 65 or D['commit_generation'] != 64 or D['first_clone'] is not None:
        return False
    lineage = _matched_lineage(tensor)
    if lineage is None:
        return False
    D['first_clone'] = dict(site=site, source_pointer=tensor.data_ptr(), output_pointer=None,
                            field='seq_lens_cpu', lineage=lineage)
    host('clone_pre', tensor)
    return True


def clone_after(selected, derived):
    if selected and D is not None:
        host('clone_post', derived)
        D['derived_ptr'] = derived.data_ptr()
        D['first_clone']['output_pointer'] = derived.data_ptr()


def downstream_before(tensor, site):
    if D is None or D['entry'] != 65 or D['commit_generation'] != 64 or D['host']['downstream_pre']['ns'] is not None:
        return False
    if not site.endswith(('.add_', '.max_item')):
        raise RuntimeError('Run401 downstream requires an actual numeric operation')
    lineage = _matched_lineage(tensor)
    if lineage is None:
        return False
    D['downstream'] = dict(site=site, pointer=tensor.data_ptr(), field='seq_lens_cpu', lineage=lineage)
    host('downstream_pre', tensor)
    return True


def downstream_after(selected):
    if selected and D is not None:
        host('downstream_post')


def export(serving, counts_cpu, cycles):
    global D
    if D is None:
        return
    import torch
    if not all(D['recorded'].values()):
        raise RuntimeError('Run401 incomplete selected events')
    # Entire loop and ordinary history drain have completed here.
    for ev in D['events'].values():
        ev.synchronize()
    D['anchor'].record()
    D['anchor'].synchronize()
    before = {label: ev.elapsed_time(D['anchor']) for label, ev in D['events'].items()}
    direct = {}
    for name, a, b in [('margin', 'R_DONE_64', 'W_PRE_65'),
                       ('read_envelope', 'R_BEGIN_64', 'R_DONE_64'),
                       ('write_envelope', 'W_PRE_65', 'W_POST_65')]:
        try:
            direct[name] = dict(ms=D['events'][a].elapsed_time(D['events'][b]))
        except RuntimeError as exc:
            direct[name] = dict(unsupported=type(exc).__name__)
    rank = torch.distributed.get_rank()
    out = Path(os.environ[ENV]); out.mkdir(parents=True, exist_ok=True)
    cohort = 1 + len(list(out.glob(f'rank{rank}_cohort*.json')))
    result = {k: D[k] for k in ('run_id', 'torch_version', 'torch_npu_version', 'start_cycle', 'device_records', 'host', 'parking',
        'reason', 'commit_pending', 'source', 'destination', 'production_event', 'copy_stream', 'schedule_mode', 'draft_graph',
        'seq_mirrors', 'computed_mirrors', 'first_clone', 'downstream', 'generation_error')}
    result.update(schema=2, rank=rank, cohort=cohort, cycles=cycles,
        phase='warmup' if cohort <= 4 else 'diagnostic',
        slot_identity='cohort-local slot; external request ledger required',
        initial_output_counts=list(serving.initial_output_counts), remaining=list(serving.remaining),
        accepted_counts=counts_cpu.tolist(), before_anchor_ms=before, direct=direct)
    with (out/f'rank{rank}_cohort{cohort}.json').open('x') as f:
        json.dump(result, f)
    D = None
