"""Run399 diagnostic hooks. Import is inert unless the patch calls a hook.
All device snapshots stay on the producing stream; export is cohort-end only.
"""
import json
import os
import re
from pathlib import Path

ENV = 'EXTREME_BOUND_TOKEN_CAPTURE_DIR'
TARGET = {}
PENDING = {}
PHASE = None
CYCLE = None
RECORDS = {}
START = None
GRAPH_STACK = []
REPLAYED = []
TRACK_TARGET = False


def enabled():
    return bool(os.getenv(ENV))


def graph_begin(entry):
    if enabled():
        GRAPH_STACK.append((id(entry), repr(entry.batch_descriptor)))


def graph_end():
    if enabled():
        GRAPH_STACK.pop()


def graph_replay(entry):
    if enabled() and TRACK_TARGET:
        REPLAYED.append((id(entry), repr(entry.batch_descriptor)))


def before_target(runtime):
    global CYCLE, TRACK_TARGET, REPLAYED
    if enabled():
        CYCLE = runtime.state.cycle_index - START
        TRACK_TARGET = CYCLE in (64, 65)
        REPLAYED = []


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



def route(layer, ids, log2phy, force_balance, dynamic_eplb):
    if not enabled():
        return
    import torch
    capturing = torch.npu.is_current_stream_capturing()
    if not capturing and PHASE != 'draft':
        return
    name, ordinal, model = identity(layer)
    if not ((capturing and model == 'target' and tuple(ids.shape) == (96, 6))
            or (PHASE == 'draft' and model == 'draft' and CYCLE in (64, 65))):
        return
    if log2phy is not None or force_balance or dynamic_eplb:
        raise RuntimeError('Run399 supports frozen non-EPLB, non-force-balanced routing only')
    if ids.dtype != torch.int32 or ids.ndim != 2 or ids.shape[1] != 6:
        raise RuntimeError('Run399 route shape/dtype changed')
    if model == 'draft' and capturing:
        raise RuntimeError('Run399 draft must be eager')
    if capturing and not GRAPH_STACK:
        raise RuntimeError('Run399 capture has no explicit ACLGraph entry')
    PENDING[ids.data_ptr()] = dict(name=name, ordinal=ordinal, model=model,
                                  ids=ids, ids_ptr=ids.data_ptr(), capturing=capturing,
                                  graph_key=GRAPH_STACK[-1] if capturing else None)


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
    return {k: (v.clone() if k in ('ids', 'group', 'expert_map') else v)
            for k, v in item.items() if k != 'capturing'}


def cohort_start(runtime):
    global START, RECORDS, PHASE, CYCLE
    if not enabled():
        return
    if getattr(runtime, 'target_page_audit', None) is not None and runtime.target_page_audit.self_replay:
        raise RuntimeError('Run399 does not support Target self-replay audit')
    START = runtime.state.cycle_index
    RECORDS = {}
    PHASE = CYCLE = None


def after_target(runtime):
    global CYCLE, PHASE, TRACK_TARGET
    if not enabled():
        return
    CYCLE = runtime.state.cycle_index - START
    PHASE = None
    TRACK_TARGET = False
    if CYCLE not in (64, 65):
        return
    if len(REPLAYED) != len(set(REPLAYED)):
        raise RuntimeError('Run399 graph entry replayed twice in one Target')
    selected = [x for x in TARGET.values() if x['graph_key'] in REPLAYED]
    if sorted(x['ordinal'] for x in selected) != list(range(43)):
        raise RuntimeError('Run399 missing exact 43 captured Target layers')
    state = runtime.state
    RECORDS[CYCLE] = dict(target=[snapshot(x) for x in selected],
        replayed_graphs=list(REPLAYED),
        target_positions=state.target_positions.clone(),
        target_input_ids=state.target_input_ids.clone(),
        active_mask=state.active_mask.clone(),
        absolute_cycle=state.cycle_index)


def after_acceptance(acceptance):
    if enabled() and CYCLE in (64, 65):
        RECORDS[CYCLE]['raw_acceptance_counts'] = acceptance.num_sampled.clone()


def draft_begin():
    global PHASE
    if enabled() and CYCLE in (64, 65):
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
    if not enabled():
        return
    import torch
    if sorted(RECORDS) != [64, 65]:
        raise RuntimeError('Run399 selected cycles not reached')
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
    cohort = 1 + len(list(out.glob(f'rank{rank}_cohort*.json')))
    data = dict(schema=1, rank=rank, cohort=cohort, cycles=cycles,
        start_cycle=START, slot_identity='cohort-local slot; external request IDs unavailable',
        initial_output_counts=list(serving.initial_output_counts),
        remaining=list(serving.remaining), initial_positions=list(serving._initial_positions),
        counts=counts_cpu.tolist(), records=cpu(RECORDS))
    path = out / f'rank{rank}_cohort{cohort}.json'
    with path.open('x') as f:
        json.dump(data, f)
