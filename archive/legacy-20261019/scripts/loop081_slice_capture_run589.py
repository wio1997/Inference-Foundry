"""Capture-time native tags and typed identity for one layer0 Target TP slice.

No device reads, stream getters or synchronization in the hot path. This
module is diagnostic and produces no latency or Bound value.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path

ENV = 'EXTREME_RUN589_DIR'
SIDECAR_PATH = Path('/data/wio/Inference_Foundry/evidence/20260928_loop081_bound/run588/bound_native_tag_run588.so')
SIDECAR_SHA = '2bbd70afbfef79d5834f33a4c7115dcdec8e6ebf4681f3dab24460674a707e6b'
STACK = []
SERIAL = 0
GRAPH = {}
ENTRIES = {}
WRITTEN = set()
TARGET_CYCLE = None
COHORT = None
REQUEST_IDS = None
START_CYCLE = None


def sidecar():
    if not active():
        return None
    if not hasattr(sidecar, '_loaded'):
        raw = SIDECAR_PATH.read_bytes()
        if hashlib.sha256(raw).hexdigest() != SIDECAR_SHA:
            raise RuntimeError('Run589 sidecar binary SHA drift')
        spec = importlib.util.spec_from_file_location('bound_native_tag_run588', str(SIDECAR_PATH))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sidecar._loaded = module
    return sidecar._loaded


def active():
    return bool(os.getenv(ENV))


def key(entry):
    return (id(entry), repr(entry.batch_descriptor))


def graph_begin(entry, owner):
    global SERIAL
    if not active():
        return
    k = key(entry)
    if k in GRAPH:
        raise RuntimeError('Run583 duplicate Graph entry capture')
    SERIAL += 1
    descriptor=entry.batch_descriptor
    fields={name:getattr(descriptor,name,None)
            for name in ('num_tokens','num_reqs','uniform','has_lora','num_active_loras')}
    GRAPH[k] = dict(capture_serial=SERIAL, entry_id=k[0],
                    descriptor=k[1], descriptor_fields=fields,
                    wrapper_type=type_name(owner),
                    runnable_type=type_name(getattr(owner,'runnable',None)),
                    layer0={}, replay_submissions=0)
    ENTRIES[k] = entry
    STACK.append(k)


def graph_end():
    if active():
        if not STACK:
            raise RuntimeError('Run583 unbalanced Graph capture')
        STACK.pop()


def target_begin(runtime):
    global TARGET_CYCLE
    if active():
        if TARGET_CYCLE is not None:
            raise RuntimeError('Run583 nested target phase')
        TARGET_CYCLE=int(runtime.state.cycle_index)


def bind_cohort(cohort, request_ids, run_tag, runtime):
    global COHORT, REQUEST_IDS, START_CYCLE
    if not active():
        return
    if type(cohort) is not int or cohort <= 0:
        raise RuntimeError('Run589 invalid cohort')
    if not isinstance(request_ids, tuple) or len(request_ids)!=12 or len(set(request_ids))!=12:
        raise RuntimeError('Run589 requires 12 unique request IDs')
    if run_tag != 'LOOP081-RUN589-B':
        raise RuntimeError('Run589 run tag mismatch')
    COHORT=cohort
    REQUEST_IDS=list(request_ids)
    START_CYCLE=int(runtime.state.cycle_index)


def target_end():
    global TARGET_CYCLE
    if active():
        if TARGET_CYCLE is None:
            raise RuntimeError('Run583 target end without begin')
        TARGET_CYCLE=None


def meta(t):
    import torch
    if not isinstance(t, torch.Tensor):
        raise TypeError('Run583 expected Tensor')
    return dict(shape=list(t.shape), stride=list(t.stride()),
                dtype=str(t.dtype), device=str(t.device),
                data_ptr=int(t.data_ptr()),
                storage_ptr=int(t.untyped_storage().data_ptr()),
                storage_offset=int(t.storage_offset()),
                storage_nbytes=int(t.untyped_storage().nbytes()),
                element_size=int(t.element_size()))


def type_name(obj):
    if obj is None:
        return None
    cls = type(obj)
    return f'{cls.__module__}.{cls.__qualname__}'


def current():
    import torch
    if not active() or not torch.npu.is_current_stream_capturing():
        return None
    if not STACK:
        raise RuntimeError('Run583 selected capture lacks Graph entry')
    return GRAPH[STACK[-1]]['layer0']


def layer0(name):
    return isinstance(name, str) and re.search(r'(?:^|\.)layers\.0\.', name) is not None


def attention_before(layer_name, impl, proj_input, output, full_weight):
    if not layer0(layer_name):
        return
    dst = current()
    if dst is None:
        return
    from vllm.distributed import get_tp_group
    from vllm_ascend.utils import oproj_tp_enable, enable_sp, enable_dsa_cp
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX
    op = getattr(impl.wo_b, 'custom_op', None)
    if 'attention_before' in dst:
        raise RuntimeError('Run583 duplicate layer0 attention record')
    group=get_tp_group()
    dst['attention_before'] = dict(layer_name=layer_name,
        full_weight=bool(full_weight),
        custom_op=type_name(op),
        input_is_parallel=(None if op is None else bool(op.input_is_parallel)),
        quant_method=type_name(impl.wo_b.quant_method),
        inner_quant_method=type_name(getattr(impl.wo_b.quant_method,'quant_method',None)),
        oproj_tp=bool(oproj_tp_enable()), sp=bool(enable_sp()),
        dsa_cp=bool(enable_dsa_cp()),
        flash_comm=bool(_EXTRA_CTX.flash_comm_v1_enabled),
        mmrs=bool(_EXTRA_CTX.mmrs_fusion), pad=int(_EXTRA_CTX.pad_size),
        tp_rank=int(group.rank_in_group),
        tp_size=int(group.world_size),
        tp_members=[int(x) for x in group.ranks],
        proj_input=meta(proj_input), output_destination=meta(output))


def projected(layer_name, value):
    if not layer0(layer_name):
        return value
    dst=current()
    if dst is not None:
        if 'projected' in dst:
            raise RuntimeError('Run583 duplicate projected output')
        dst['projected']=meta(value)
    return value


def attention_after(layer_name, output):
    if not layer0(layer_name):
        return
    dst = current()
    if dst is None:
        return
    if 'attention_after' in dst:
        raise RuntimeError('Run583 duplicate attention copy record')
    dst['attention_after'] = dict(output=meta(output))
    tag_native('copy', dst['attention_after']['output'], dst)


def tag_native(role, tensor_meta, dst):
    if role in dst.get('native_tags', {}):
        raise RuntimeError('Run589 duplicate native role')
    from vllm.distributed import get_tp_group
    rank=int(get_tp_group().rank_in_group)
    nbytes=tensor_meta['element_size']*__import__('math').prod(tensor_meta['shape'])
    if role=='copy':
        src=dst.get('projected')
        if src is None or src['shape']!=tensor_meta['shape'] or src['dtype']!=tensor_meta['dtype']:
            raise RuntimeError('Run589 copy source geometry missing')
        payload=('RUN589|r=%d|g=%d|role=copy|src=0x%016x|dst=0x%016x|bytes=%d' %
                 (rank, GRAPH[STACK[-1]]['capture_serial'],
                  src['data_ptr'],tensor_meta['data_ptr'],nbytes))
    else:
        payload=('RUN589|r=%d|g=%d|role=%s|ptr=0x%016x|bytes=%d' %
                 (rank, GRAPH[STACK[-1]]['capture_serial'], role,
                  tensor_meta['data_ptr'],nbytes))
    sidecar().enqueue_tag(payload)
    dst.setdefault('native_tags', {})[role]=payload


def before_sequence_rs(prefix, x, partial, flash_comm, mmrs):
    if not layer0(prefix) or 'wo_b' not in prefix:
        return
    dst=current()
    if dst is None:
        return
    if not flash_comm or mmrs:
        raise RuntimeError('Run589 unexpected row-parallel branch')
    info=meta(partial)
    dst['partial_pre_rs']=info
    tag_native('partial', info, dst)


def sequence_rs(prefix, x, partial, reduced, flash_comm, mmrs):
    if not layer0(prefix) or 'wo_b' not in prefix:
        return
    dst = current()
    if dst is None:
        return
    if 'sequence_rs' in dst:
        raise RuntimeError('Run583 duplicate layer0 RS record')
    dst['sequence_rs'] = dict(prefix=prefix,
        flash_comm=bool(flash_comm), mmrs=bool(mmrs),
        input=meta(x), partial=meta(partial), reduced=meta(reduced))


def graph_replay(entry, owner):
    if not active():
        return
    k = key(entry)
    if k not in GRAPH or entry.aclgraph is None:
        raise RuntimeError('Run583 replay lacks capture identity')
    GRAPH[k]['replay_submissions']+=1
    if (k in WRITTEN or TARGET_CYCLE is None or COHORT!=5):
        return
    row = GRAPH[k]
    # Only emit graphs that actually included the Target layer0 slice.
    if 'attention_before' not in row['layer0']:
        return
    from vllm.distributed import get_tp_group
    rank = int(get_tp_group().rank_in_group)
    out = Path(os.environ[ENV])
    out.mkdir(parents=True, exist_ok=True)
    row = dict(row, rank=rank, graph_object_id=id(entry.aclgraph),
               replay_submission_ordinal=GRAPH[k]['replay_submissions'],
               replay_submitted=True, replay_completed=False,
               replay_owner_type=type_name(owner),
               runtime_phase='target', runtime_cycle=TARGET_CYCLE,
               cohort=COHORT, request_ids=REQUEST_IDS,
               cohort_start_cycle=START_CYCLE,
               relative_cycle=TARGET_CYCLE-START_CYCLE,
               run_tag=os.getenv('RUN_TS'))
    with (out / f'rank{rank}_captures.jsonl').open('a') as f:
        f.write(json.dumps(row, separators=(',', ':')) + '\n')
    WRITTEN.add(k)


def cohort_end(cohort, request_ids, runtime, cycles):
    if not active() or cohort!=5:
        return
    if COHORT!=cohort or list(request_ids)!=REQUEST_IDS or cycles<=0:
        raise RuntimeError('Run589 cohort end identity mismatch')
    from vllm.distributed import get_tp_group
    rank=int(get_tp_group().rank_in_group)
    selected=[k for k in WRITTEN if GRAPH[k]['descriptor_fields']==
              {'num_tokens':96,'num_reqs':12,'uniform':True,
               'has_lora':False,'num_active_loras':0}]
    if len(selected)!=1:
        raise RuntimeError('Run589 selected FULL96 Graph not unique')
    k=selected[0]
    entry=ENTRIES[k]
    out=Path(os.environ[ENV]);out.mkdir(parents=True,exist_ok=True)
    dump=out/f'rank{rank}_cohort5_acl_graph.json'
    entry.aclgraph.debug_dump(str(dump))
    raw=dump.read_bytes();nodes=json.loads(raw)
    meta_row=dict(rank=rank,cohort=cohort,request_ids=REQUEST_IDS,
                  start_cycle=START_CYCLE,end_cycle=int(runtime.state.cycle_index),
                  cycles=cycles,capture_serial=GRAPH[k]['capture_serial'],
                  entry_id=k[0],graph_object_id=id(entry.aclgraph),
                  replay_submissions=GRAPH[k]['replay_submissions'],
                  graph_dump_path=str(dump),graph_dump_sha256=hashlib.sha256(raw).hexdigest(),
                  graph_task_count=len(nodes),completion='post cohort torch.npu.synchronize')
    with (out/f'rank{rank}_cohort5_graph_meta.json').open('x') as f:
        json.dump(meta_row,f,indent=2)
