"""Run577 terminal diagnostic: production W4A8 weights, private nonzero input.

No inference continuation is permitted after this terminal measurement. It is
isolated attained service, never compulsory work or a strict C+ certificate.
"""
from __future__ import annotations

import json
import hashlib
import os
import re
import statistics
import time
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
ENV = 'EXTREME_RUN577_DIR'
WEIGHTS = {}
BENCHMARKED = False
CENSUS = ROOT/'evidence/20260928_loop080_bound/run576/operand_census.json'
CENSUS_SHA = '615aeb42086b035903867261e45642a6390a13ba396d91ec31d7cb62320ca1e8'


def reject_new_cohort(req_ids, served_cohorts):
    if os.getenv(ENV) and BENCHMARKED and req_ids not in served_cohorts:
        raise RuntimeError('Run577 terminal stage forbids a new model cohort')


def observe(layer, ids, w1, w2, ws1, ws2, b1, b2, quant_type, mega_moe):
    if not os.getenv(ENV):
        return
    import torch
    if not torch.npu.is_current_stream_capturing() or tuple(ids.shape) != (96, 6):
        return
    name = getattr(layer, 'layer_name', None)
    match = re.fullmatch(r'(?:.*\.)?layers\.(\d+)\.mlp\.experts', name or '')
    if not match or not 0 <= int(match[1]) < 43:
        return
    if str(quant_type) != 'QuantType.W4A8' or mega_moe:
        raise RuntimeError('Run577 requires non-Mega W4A8 Target')
    operands = dict(w1=w1, w2=w2, w1_scale=ws1, w2_scale=ws2,
                    w1_scale_bias=b1, w2_scale_bias=b2)
    if any(not isinstance(v, list) or len(v) != 1 for v in operands.values()):
        raise RuntimeError('Run577 requires one resident tensor per operand')
    ordinal = int(match[1])
    if ordinal in WEIGHTS:
        if any(WEIGHTS[ordinal][key][0].data_ptr() != value[0].data_ptr()
               for key, value in operands.items()):
            raise RuntimeError('Run577 duplicate layer changed operand storage')
        return
    WEIGHTS[ordinal] = dict(operands, swiglu_limit=float(layer.swiglu_limit))


def _meta(t):
    import torch_npu
    return dict(shape=list(t.shape), dtype=str(t.dtype),
                format=int(torch_npu.get_npu_format(t)),
                storage_nbytes=t.untyped_storage().nbytes(),
                data_ptr=t.data_ptr())


def _wait_all(out, rank, phase, timeout_s=120):
    ready = out / ('ready_' + phase)
    ready.mkdir(parents=True, exist_ok=True)
    (ready / f'rank{rank}').write_text(str(time.monotonic_ns()))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if len(list(ready.glob('rank[0-7]'))) == 8:
            return
        time.sleep(0.05)
    raise TimeoutError(f'Run577 all8 {phase} barrier timeout')


def benchmark(cohort_index):
    global BENCHMARKED
    root = os.getenv(ENV)
    if not root or cohort_index != 8:
        return
    if BENCHMARKED:
        raise RuntimeError('Run577 terminal benchmark repeated')
    BENCHMARKED = True
    import torch
    import torch_npu
    from vllm.distributed import get_tp_group
    rank = int(get_tp_group().rank_in_group)
    out = Path(root)
    out.mkdir(parents=True, exist_ok=True)
    if set(WEIGHTS) != set(range(43)):
        raise RuntimeError(f'Run577 missing production Target weights: {sorted(set(range(43))-set(WEIGHTS))}')
    census_bytes = CENSUS.read_bytes()
    if hashlib.sha256(census_bytes).hexdigest() != CENSUS_SHA:
        raise RuntimeError('Run577 pinned Run576 census drift')
    census = json.loads(census_bytes)
    fixture_rel = f'evidence/20260928_loop080_bound/run576/live/b/capture/rank{rank}_cohort5.json'
    fixture_path = ROOT / fixture_rel
    fixture_bytes = fixture_path.read_bytes()
    fixture_sha = hashlib.sha256(fixture_bytes).hexdigest()
    if fixture_sha != census['provenance'].get(fixture_rel):
        raise RuntimeError('Run577 pinned Run576 route fixture drift')
    fixture = json.loads(fixture_bytes)
    rows = fixture['records']['170']['target']
    if fixture['run_tag'] != 'LOOP080-RUN576-B' or len(rows) != 43:
        raise RuntimeError('Run577 Run576 route fixture identity drift')
    assert [r['ordinal'] for r in rows] == list(range(43))
    torch.npu.synchronize()
    _wait_all(out, rank, 'terminal')
    free, total = torch.npu.mem_get_info()
    if free < 4 * 1024**3:
        raise RuntimeError(f'Run577 scratch memory guard: free={free}')

    activation_seed = 577000 + rank
    torch.manual_seed(activation_seed)
    cases = []
    for row in rows:
        ordinal = row['ordinal']
        item = WEIGHTS[ordinal]
        for key in ('w1', 'w2', 'w1_scale', 'w2_scale', 'w1_scale_bias', 'w2_scale_bias'):
            actual = _meta(item[key][0])
            prior = row['weights'][key][0]
            if (actual['shape'] != prior['shape'] or
                actual['dtype'] != prior['dtype'] or
                actual['format'] != prior['npu_format'] or
                actual['storage_nbytes'] != prior['storage_nbytes']):
                raise RuntimeError(f'Run577 production operand drift rank{rank} layer{ordinal} {key}')
        counts = row['group']
        if len(counts) != 32 or not 0 < sum(counts) <= 576:
            raise RuntimeError('Run577 group fixture invalid')
        # Every writable operand is private to this terminal benchmark.
        g = torch.tensor(counts, device=f'npu:{rank}', dtype=torch.int64)
        x = torch.randint(-3, 4, (576, 4096), device=f'npu:{rank}', dtype=torch.int8)
        xs = torch.ones((576,), device=f'npu:{rank}', dtype=torch.float32)
        cases.append((ordinal, item, g, x, xs, sum(counts)))

    def one(case):
        _, item, g, x, xs, _ = case
        h, s = torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
            x=x, weight=item['w1'], weight_scale=item['w1_scale'],
            x_scale=xs, group_list=g, weight_assist_matrix=item['w1_scale_bias'],
            dequant_mode=0, group_list_type=1,
            swiglu_limit=item['swiglu_limit'])
        y = torch_npu.npu_grouped_matmul(
            x=[h], weight=item['w2'], scale=item['w2_scale'],
            bias=item['w2_scale_bias'], per_token_scale=[s],
            split_item=2, group_list_type=1, group_type=0,
            group_list=g, output_dtype=torch.bfloat16)[0]
        return y

    # Eager and Graph must agree on defined local routed rows. Padded rows are
    # intentionally excluded; native kernels need not initialize them.
    eager = [one(case) for case in cases]
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_outputs = [one(case) for case in cases]
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    for case, reference, actual in zip(cases, eager, graph_outputs):
        n = case[-1]
        if tuple(actual.shape) != (576, 4096) or not bool(torch.isfinite(actual[:n]).all().item()):
            raise RuntimeError(f'Run577 invalid Graph output layer{case[0]}')
        torch.testing.assert_close(actual[:n], reference[:n], rtol=0.01, atol=0.05)
    del eager
    _wait_all(out, rank, 'verified')
    for _ in range(3):
        graph.replay()
    torch.npu.synchronize()
    _wait_all(out, rank, 'timed')
    events = []
    host_start = time.perf_counter_ns()
    for _ in range(20):
        a = torch.npu.Event(enable_timing=True)
        b = torch.npu.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        events.append((a, b))
    torch.npu.synchronize()
    host_end = time.perf_counter_ns()
    durations_ms = [a.elapsed_time(b) for a, b in events]
    free_after, _ = torch.npu.mem_get_info()
    if not all(x > 0 for x in durations_ms):
        raise RuntimeError('Run577 invalid elapsed event')
    result = dict(status='isolated_real_weight_synthetic_activation_service',
                  run_tag=os.getenv('RUN_TS'), rank=rank, cohort=cohort_index,
                  fixture='Run576 cohort5 cycle170', target_layers=43,
                  fixture_sha256=fixture_sha, census_sha256=CENSUS_SHA,
                  synthetic_activation_seed=activation_seed,
                  group_counts_per_layer=[row['group'] for row in rows],
                  operator_contract=dict(gmm1='grouped_matmul_swiglu_quant_v2',
                      gmm1_dequant_mode=0, group_list_type=1,
                      gmm2='npu_grouped_matmul', gmm2_split_item=2,
                      gmm2_group_type=0, gmm2_output_dtype='torch.bfloat16',
                      swiglu_limits=[case[1]['swiglu_limit'] for case in cases]),
                  route_incidence=sum(c[-1] for c in cases),
                  actual_weight_storage_ptrs=[{key: item[key][0].data_ptr()
                                               for key in ('w1','w2')}
                                              for _,item,_,_,_,_ in cases],
                  free_hbm_before_bytes=free, total_hbm_bytes=total,
                  free_hbm_after_bytes=free_after,
                  max_memory_allocated_bytes=torch.npu.max_memory_allocated(),
                  max_memory_reserved_bytes=torch.npu.max_memory_reserved(),
                  eager_graph_valid_rows_close=True,
                  samples_ms=durations_ms,
                  median_ms=statistics.median(durations_ms),
                  host_start_ns=host_start, host_end_ns=host_end,
                  limits=['real production W4A8 weights and format, synthetic nonzero activation/scale',
                          'isolated 43-layer GMM service with no HCCL, Target/Draft or KV contention',
                          'attained Engineering service only; not compulsory work, strict C+ or Product TPS'])
    (out / f'rank{rank}.json').write_text(json.dumps(result, indent=2)+'\n')
    _wait_all(out, rank, 'complete')
