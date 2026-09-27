"""Run576 one-cycle post-park operand witness; diagnostic, never TPS evidence.

The source patch calls these hooks only when the opt-in directory is set.
Device snapshots are enqueued on the producer stream; export follows drain.
"""
import json
import os
import re
from pathlib import Path

ENV = 'EXTREME_POSTPARK_CAPTURE_DIR'
TARGET = {}
ATTENTION = {}
PENDING = {}
PHASE = None
CYCLE = None
RECORDS = {}
START = None
COHORT = 0
SELECTED = None
PARK = None
REQUEST_IDS = None
RUN_TAG = None
GRAPH_STACK = []
REPLAYED = []
REPLAY_META = []
GRAPH_GENERATIONS = {}
CAPTURE_SERIAL = 0
TRACK_TARGET = False


def enabled():
    return bool(os.getenv(ENV))


def graph_begin(entry):
    global CAPTURE_SERIAL
    if enabled():
        key = (id(entry), repr(entry.batch_descriptor))
        if key in GRAPH_GENERATIONS:
            raise RuntimeError('Run576 same Graph entry captured twice')
        CAPTURE_SERIAL += 1
        GRAPH_GENERATIONS[key] = CAPTURE_SERIAL
        GRAPH_STACK.append(key)


def graph_end():
    if enabled():
        GRAPH_STACK.pop()


def graph_replay(entry):
    if enabled() and TRACK_TARGET:
        key = (id(entry), repr(entry.batch_descriptor))
        if key not in GRAPH_GENERATIONS or entry.aclgraph is None:
            raise RuntimeError('Run576 replay lacks captured Graph generation')
        REPLAYED.append(key)
        REPLAY_META.append(dict(key=key, capture_serial=GRAPH_GENERATIONS[key],
                                graph_object_id=id(entry.aclgraph)))


def before_target(runtime):
    global CYCLE, TRACK_TARGET, REPLAYED, REPLAY_META
    if enabled():
        CYCLE = runtime.state.cycle_index - START
        TRACK_TARGET = COHORT == 5 and CYCLE == SELECTED
        REPLAYED = []
        REPLAY_META = []


def identity(layer):
    name = getattr(layer, 'layer_name', None)
    if not isinstance(name, str):
        raise RuntimeError('Run399 requires explicit RoutedExperts.layer_name')
    target = re.fullmatch(r'(?:.*\.)?layers\.(\d+)\.mlp\.experts', name)
    draft = re.fullmatch(r'mtp\.(\d+)\.mlp\.experts', name)
    if target and 0 <= int(target[1]) < 43:
        return name, int(target[1]), 'target'
    if draft and 0 <= int(draft[1]) < 3:
        return name, 43 + int(draft[1]), 'draft'
    raise RuntimeError(f'Run399 unknown layer identity: {name}')



def route(layer, ids, log2phy, force_balance, dynamic_eplb,
          hidden, router_logits, weights, quant_type, mega_moe):
    if not enabled():
        return
    import torch
    capturing = torch.npu.is_current_stream_capturing()
    if not capturing and PHASE != 'draft':
        return
    name, ordinal, model = identity(layer)
    if not ((capturing and model == 'target' and tuple(ids.shape) == (96, 6))
            or (PHASE == 'draft' and model == 'draft' and CYCLE == SELECTED)):
        return
    if log2phy is not None or force_balance or dynamic_eplb:
        raise RuntimeError('Run576 supports frozen non-EPLB, non-force-balanced routing only')
    if ids.dtype != torch.int32 or ids.ndim != 2 or ids.shape[1] != 6:
        raise RuntimeError('Run399 route shape/dtype changed')
    if model == 'draft' and capturing:
        raise RuntimeError('Run399 draft must be eager')
    if capturing and not GRAPH_STACK:
        raise RuntimeError('Run399 capture has no explicit ACLGraph entry')
    PENDING[ids.data_ptr()] = dict(name=name, ordinal=ordinal, model=model,
                                  ids=ids, ids_ptr=ids.data_ptr(), capturing=capturing,
                                  graph_key=GRAPH_STACK[-1] if capturing else None,
                                  capture_serial=(GRAPH_GENERATIONS[GRAPH_STACK[-1]]
                                                  if capturing else None),
                                  hidden_meta=tensor_meta(hidden),
                                  router_logits_meta=tensor_meta(router_logits),
                                  weights_ref=weights, quant_type=str(quant_type),
                                  mega_moe=bool(mega_moe))


def attention(layer_name, q, q_prefix, key_lengths, ratio, full_gather):
    if not enabled():
        return
    import torch
    if not torch.npu.is_current_stream_capturing():
        return
    if not GRAPH_STACK:
        raise RuntimeError('Run576 attention capture has no Graph entry')
    match = re.search(r'(?:^|\.)layers\.(\d+)\.', layer_name)
    if not match or not 0 <= int(match[1]) < 43:
        raise RuntimeError(f'Run576 unknown attention layer: {layer_name}')
    key = (GRAPH_STACK[-1], int(match[1]))
    if key in ATTENTION:
        raise RuntimeError(f'Run576 duplicate attention: {key}')
    ATTENTION[key] = dict(ordinal=int(match[1]), graph_key=GRAPH_STACK[-1],
                          capture_serial=GRAPH_GENERATIONS[GRAPH_STACK[-1]],
                          q_meta=tensor_meta(q), q_prefix_ref=q_prefix,
                          key_lengths_ref=key_lengths,
                          ratio=int(ratio), full_gather=bool(full_gather))


def dispatched(ids, output, dispatcher, mapping):
    if not enabled():
        return
    item = PENDING.pop(ids.data_ptr(), None)
    if item is None:
        return
    from vllm.distributed import get_ep_group
    if type(dispatcher).__name__ != 'TokenDispatcherWithAllGather':
        raise RuntimeError('Run399 requires AllGather dispatcher')
    if output.group_list_type != 1 or tuple(output.group_list.shape) != (32,):
        raise RuntimeError('Run399 requires 32 local count-mode experts')
    if mapping is None or mapping.numel() != 256:
        raise RuntimeError('Run399 requires explicit 256-expert EP map')
    item.update(group=output.group_list, group_ptr=output.group_list.data_ptr(), expert_map=mapping,
                ep_rank=get_ep_group().rank_in_group)
    if item['capturing']:
        key = (item['graph_key'], item['name'])
        old = TARGET.get(key)
        if old is not None and old['ids'].data_ptr() != ids.data_ptr():
            raise RuntimeError(f'Run399 same graph entry/layer changed route storage: {key}')
        TARGET[key] = item
    else:
        dst = RECORDS[CYCLE].setdefault('draft', [])
        if any(x['name'] == item['name'] for x in dst):
            raise RuntimeError('Run399 duplicate draft layer')
        dst.append(snapshot(item))


def snapshot(item):
    result = {k: (v.clone() if k in ('ids', 'group', 'expert_map') else v)
              for k, v in item.items()
              if k not in ('capturing', 'weights_ref')}
    result['hidden'] = result.pop('hidden_meta')
    result['router_logits'] = result.pop('router_logits_meta')
    result['weights'] = {name: [tensor_meta(x, physical=True) for x in values] if values is not None else None
                         for name, values in item['weights_ref'].items()}
    return result


def tensor_meta(t, physical=False):
    import torch
    if not isinstance(t, torch.Tensor):
        raise RuntimeError('Run576 expected Tensor operand')
    value = dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype),
                device=str(t.device), data_ptr=t.data_ptr(),
                storage_ptr=t.untyped_storage().data_ptr(),
                storage_offset=t.storage_offset(), element_size=t.element_size(),
                storage_nbytes=t.untyped_storage().nbytes())
    if physical and t.device.type == 'npu':
        import torch_npu
        value['npu_format'] = int(torch_npu.get_npu_format(t))
    return value


def attention_snapshot(item):
    return dict(ordinal=item['ordinal'], graph_key=item['graph_key'],
                capture_serial=item['capture_serial'],
                q=item['q_meta'],
                q_prefix=item['q_prefix_ref'].clone(),
                key_lengths=item['key_lengths_ref'].clone(),
                ratio=item['ratio'], full_gather=item['full_gather'])


def cohort_start(runtime):
    global START, RECORDS, PHASE, CYCLE, SELECTED, PARK
    if not enabled():
        return
    if getattr(runtime, 'target_page_audit', None) is not None and runtime.target_page_audit.self_replay:
        raise RuntimeError('Run399 does not support Target self-replay audit')
    START = runtime.state.cycle_index
    if COHORT <= 0 or REQUEST_IDS is None or RUN_TAG is None:
        raise RuntimeError('Run576 ModelRunner cohort context missing')
    SELECTED = None
    PARK = None
    RECORDS = {}
    PHASE = CYCLE = None


def bind_cohort(cohort, request_ids, run_tag):
    global COHORT, REQUEST_IDS, RUN_TAG
    if not enabled():
        return
    if type(cohort) is not int or cohort <= 0:
        raise RuntimeError('Run576 invalid cohort index')
    if not isinstance(request_ids, tuple) or len(request_ids) != 12 or len(set(request_ids)) != 12:
        raise RuntimeError('Run576 requires twelve unique request IDs')
    if not isinstance(run_tag, str) or not run_tag.startswith('LOOP080-RUN576-'):
        raise RuntimeError('Run576 run tag missing')
    COHORT = cohort
    REQUEST_IDS = tuple(request_ids)
    RUN_TAG = run_tag


def after_park(serving, slots):
    global SELECTED, PARK
    if not enabled() or COHORT != 5 or not slots:
        return
    if SELECTED is not None:
        return
    selected = serving.runtime.state.cycle_index - START
    if selected <= 0 or selected >= serving.max_cycles:
        raise RuntimeError('Run576 invalid first post-park cycle')
    if not any(serving._parked) or all(serving._parked):
        raise RuntimeError('Run576 expected partial natural parking')
    SELECTED = selected
    PARK = dict(completed_cycle=selected - 1, next_target_cycle=selected,
                newly_parked_slots=list(slots), parked=list(serving._parked),
                progress=list(serving._committed_progress()))


def after_target(runtime):
    global CYCLE, PHASE, TRACK_TARGET
    if not enabled():
        return
    CYCLE = runtime.state.cycle_index - START
    PHASE = None
    TRACK_TARGET = False
    if COHORT != 5 or CYCLE != SELECTED:
        return
    import torch
    # Diagnostic only: complete producers before copying dynamic Graph buffers.
    torch.npu.synchronize()
    if len(REPLAYED) != len(set(REPLAYED)):
        raise RuntimeError('Run399 graph entry replayed twice in one Target')
    selected = [x for x in TARGET.values() if x['graph_key'] in REPLAYED]
    if sorted(x['ordinal'] for x in selected) != list(range(43)):
        raise RuntimeError('Run399 missing exact 43 captured Target layers')
    state = runtime.state
    attn = [x for x in ATTENTION.values() if x['graph_key'] in REPLAYED]
    if sorted(x['ordinal'] for x in attn) != list(range(43)):
        raise RuntimeError('Run576 missing exact 43 captured attention layers')
    RECORDS[CYCLE] = dict(target=[snapshot(x) for x in selected],
        attention=[attention_snapshot(x) for x in attn],
        replayed_graphs=list(REPLAYED),
        replay_meta=list(REPLAY_META),
        target_positions=state.target_positions.clone(),
        target_input_ids=state.target_input_ids.clone(),
        target_query_start_loc=state.target_query_start_loc.clone(),
        target_logits_indices=state.target_logits_indices.clone(),
        target_seq_lens=state.target_seq_lens.clone(),
        active_mask=state.active_mask.clone(),
        absolute_cycle=state.cycle_index)


def after_acceptance(acceptance):
    if enabled() and COHORT == 5 and CYCLE == SELECTED:
        RECORDS[CYCLE]['raw_acceptance_counts'] = acceptance.num_sampled.clone()


def draft_begin():
    global PHASE
    if enabled() and COHORT == 5 and CYCLE == SELECTED:
        PHASE = 'draft'


def draft_inputs(proposer, num_query_total, indices, cad):
    if not enabled() or PHASE != 'draft':
        return
    RECORDS[CYCLE]['draft_metadata'] = dict(
        q=int(proposer.num_query_per_req),
        sample_from_anchor=bool(proposer.sample_from_anchor),
        positions=proposer.positions[:num_query_total].clone(),
        input_ids=proposer.input_ids[:num_query_total].clone(),
        sample_indices=indices.clone(),
        query_start_loc=cad.query_start_loc.clone())


def draft_end(next_draft):
    global PHASE
    if enabled() and PHASE == 'draft':
        RECORDS[CYCLE]['draft_output'] = next_draft.clone()
    PHASE = None


def export(serving, counts_cpu, cycles):
    if not enabled() or COHORT != 5:
        return
    import torch
    if SELECTED is None or sorted(RECORDS) != [SELECTED] or PARK is None:
        raise RuntimeError('Run576 selected post-park cycle not reached')
    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.cpu().tolist()
        if isinstance(value, dict):
            return {str(k): cpu(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cpu(v) for v in value]
        return value
    out = Path(os.environ[ENV])
    out.mkdir(parents=True, exist_ok=True)
    rank = torch.distributed.get_rank()
    data = dict(schema=2, rank=rank, cohort=COHORT, run_tag=RUN_TAG,
        request_ids=list(REQUEST_IDS), cycles=cycles,
        start_cycle=START, slot_identity='cohort-local slot; external request IDs unavailable',
        park=PARK, selected_cycle=SELECTED,
        initial_output_counts=list(serving.initial_output_counts),
        remaining=list(serving.remaining), initial_positions=list(serving._initial_positions),
        counts=counts_cpu.tolist(), records=cpu(RECORDS))
    path = out / f'rank{rank}_cohort{COHORT}.json'
    with path.open('x') as f:
        json.dump(data, f)
