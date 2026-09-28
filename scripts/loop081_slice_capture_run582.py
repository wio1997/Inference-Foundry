"""Capture-time scalar/descriptor identity for one layer0 Target TP slice.

No device reads, tensor retention, stream getters or synchronization.  This
module is diagnostic and produces no latency or Bound value.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

ENV = 'EXTREME_RUN582_DIR'
STACK = []
SERIAL = 0
GRAPH = {}
WRITTEN = set()
TARGET_CYCLE = None


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
        raise RuntimeError('Run582 duplicate Graph entry capture')
    SERIAL += 1
    descriptor=entry.batch_descriptor
    fields={name:getattr(descriptor,name,None)
            for name in ('num_tokens','num_reqs','uniform','has_lora','num_active_loras')}
    GRAPH[k] = dict(capture_serial=SERIAL, entry_id=k[0],
                    descriptor=k[1], descriptor_fields=fields,
                    wrapper_type=type_name(owner),
                    runnable_type=type_name(getattr(owner,'runnable',None)),
                    layer0={}, replay_submissions=0)
    STACK.append(k)


def graph_end():
    if active():
        if not STACK:
            raise RuntimeError('Run582 unbalanced Graph capture')
        STACK.pop()


def target_begin(runtime):
    global TARGET_CYCLE
    if active():
        if TARGET_CYCLE is not None:
            raise RuntimeError('Run582 nested target phase')
        TARGET_CYCLE=int(runtime.state.cycle_index)


def target_end():
    global TARGET_CYCLE
    if active():
        if TARGET_CYCLE is None:
            raise RuntimeError('Run582 target end without begin')
        TARGET_CYCLE=None


def meta(t):
    import torch
    if not isinstance(t, torch.Tensor):
        raise TypeError('Run582 expected Tensor')
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
        raise RuntimeError('Run582 selected capture lacks Graph entry')
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
        raise RuntimeError('Run582 duplicate layer0 attention record')
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
            raise RuntimeError('Run582 duplicate projected output')
        dst['projected']=meta(value)
    return value


def attention_after(layer_name, output):
    if not layer0(layer_name):
        return
    dst = current()
    if dst is None:
        return
    if 'attention_after' in dst:
        raise RuntimeError('Run582 duplicate attention copy record')
    dst['attention_after'] = dict(output=meta(output))


def sequence_rs(prefix, x, partial, reduced, flash_comm, mmrs):
    if not layer0(prefix) or 'wo_b' not in prefix:
        return
    dst = current()
    if dst is None:
        return
    if 'sequence_rs' in dst:
        raise RuntimeError('Run582 duplicate layer0 RS record')
    dst['sequence_rs'] = dict(prefix=prefix,
        flash_comm=bool(flash_comm), mmrs=bool(mmrs),
        input=meta(x), partial=meta(partial), reduced=meta(reduced))


def hc_consumer(layer_idx, hidden, residual, post, comb):
    if layer_idx != 0:
        return
    dst = current()
    if dst is None:
        return
    if 'hc_consumer' in dst:
        raise RuntimeError('Run582 duplicate layer0 HC-post record')
    dst['hc_consumer'] = dict(hidden=meta(hidden), residual=meta(residual),
                              post=meta(post), comb=meta(comb))


def graph_replay(entry, owner):
    if not active():
        return
    k = key(entry)
    if k not in GRAPH or entry.aclgraph is None:
        raise RuntimeError('Run582 replay lacks capture identity')
    GRAPH[k]['replay_submissions']+=1
    if k in WRITTEN or TARGET_CYCLE is None:
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
               runtime_phase='target', runtime_cycle=TARGET_CYCLE)
    with (out / f'rank{rank}_captures.jsonl').open('a') as f:
        f.write(json.dumps(row, separators=(',', ':')) + '\n')
    WRITTEN.add(k)
