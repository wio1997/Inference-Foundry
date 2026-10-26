"""Reproduce L22 representative trace comparisons offline; no service/model calls."""
import collections
import hashlib
import json
import statistics
from decimal import Decimal, getcontext
from pathlib import Path

getcontext().prec = 50
ROOT = Path(__file__).resolve().parents[1]
CORE_TYPES = {'AI_CORE', 'AI_VECTOR_CORE', 'MIX_AIC', 'MIX_AIV', 'AI_CPU'}
SOURCES = [
    ('vllm', 43, 4, 'runs/20261019T-L22-vllm-representative-offline-export-v1/node-43-trace.json', 'b08aee635de61c034c5d8870cec0e554d6cc876c13167cd868786ef92b095931'),
    ('vllm', 45, 10, 'runs/20261019T-L22-vllm-representative-offline-export-v1/node-45-trace.json', '70c0e28f7ab0681e155d4d7ce06c9e4b9a5267aa0cd13eb741d2571ef992e557'),
    ('sglang', 36, 4, 'runs/20261019T-L22-native-evidence-collect-sglang-v3/trace-36-trace_view.json', '2a42ebb274c19c7c4a1ea7da6e99a77870abc360821a9ac7a15048c9715827b1'),
    ('sglang', 41, 10, 'runs/20261019T-L22-native-evidence-collect-sglang-v3/trace-41-trace_view.json', 'd2b3ceb46ad207efc6d6c0b9f69364589dcbd2bfbe459760bb5788adafecf3a1'),
]


def dec(value):
    return Decimal(str(value))


def stats(events):
    values = [dec(e['dur']) for e in events]
    return {'count': len(values), 'sum_us_nonadditive': str(sum(values)),
            'median_us': str(statistics.median(values)) if values else None,
            'max_us': str(max(values)) if values else None}


def analyze(framework, node, dp, relative, sha):
    raw = (ROOT / relative).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == sha, relative
    document = json.loads(raw, parse_float=Decimal)
    events = document if isinstance(document, list) else document['traceEvents']
    processes = {e['pid']: e['args']['name'] for e in events if e.get('name') == 'process_name'}
    hardware_pids = {p for p, n in processes.items() if n == 'Ascend Hardware'}
    cann_pids = {p for p, n in processes.items() if n == 'CANN'}
    assert len(hardware_pids) == 1 and len(cann_pids) == 1
    hardware = [e for e in events if e.get('ph') == 'X' and e.get('pid') in hardware_pids]
    core = [e for e in hardware if e.get('args', {}).get('Task Type') in CORE_TYPES]
    graph = sorted([e for e in core if e['args'].get('Model Id') != 4294967295], key=lambda e: dec(e['ts']))
    assert graph and {e['args']['Model Id'] for e in graph} == {46}
    first = graph[0]
    def key(e):
        return (e['args'].get('Physic Stream Id'), e['args'].get('Task Id'), e['name'])
    markers = [e for e in graph if key(e) == key(first)]
    expected = 8 if framework == 'vllm' else 2
    assert len(markers) == expected, (framework, node, len(markers))
    spans, periods, gaps = [], [], []
    for i, marker in enumerate(markers):
        start = dec(marker['ts'])
        end_limit = dec(markers[i + 1]['ts']) if i + 1 < len(markers) else Decimal('Infinity')
        members = [e for e in graph if start <= dec(e['ts']) < end_limit]
        end = max(dec(e['ts']) + dec(e['dur']) for e in members)
        spans.append((end - start) / 1000)
        if i + 1 < len(markers):
            periods.append((end_limit - start) / 1000)
            gaps.append((end_limit - end) / 1000)
    eager = [e for e in core if e['args'].get('Model Id') == 4294967295]
    def dispatch(es):
        return [e for e in es if 'MoeDistributeDispatchV2' in e.get('name', '')]
    cann = [e for e in events if e.get('ph') == 'X' and e.get('pid') in cann_pids]
    api_threads = {}
    for name in ['aclnnInnerMoeDistributeDispatchV2', 'aclnnInnerMoeDistributeDispatchV2GetWorkspaceSize', 'aclrtMemcpyAsync']:
        selected = [e for e in cann if e.get('args', {}).get('id') == name]
        assert selected, (framework, node, name)
        api_threads[name] = stats(selected) | {'thread_counts': dict(collections.Counter(str(e['tid']) for e in selected))}
    queue = collections.Counter(e.get('cat', '') for e in events)
    steps = collections.Counter(e['name'] for e in events if e.get('ph') == 'X' and e['name'].startswith('step['))
    return {
        'framework': framework, 'node': node, 'native_DP': dp, 'native_TP': 0,
        'source': relative, 'bytes': len(raw), 'sha256': sha,
        'segmentation': 'First graph core-task stream/task/name repeats; model 46; original hardware process only. Connection IDs are reused and are not iteration keys.',
        'first_graph_task': list(key(first)), 'target_graph_replays': len(markers),
        'target_graph_core_spans_ms': [str(x) for x in spans],
        'target_graph_start_periods_ms': [str(x) for x in periods],
        'inter_target_graph_gaps_ms': [str(x) for x in gaps],
        'eager_MoeDistributeDispatchV2': stats(dispatch(eager)),
        'target_MoeDistributeDispatchV2': stats(dispatch(graph)),
        'target_MoE_dispatch_per_replay': len(dispatch(graph)) / len(markers),
        'eager_MoE_dispatch_per_target_replay': len(dispatch(eager)) / len(markers),
        'target_MlaPrologV3_count': sum('MlaPrologV3' in e['name'] for e in graph),
        'async_task_queue_flow_events': queue.get('async_task_queue', 0),
        'CANN_API_threads_and_nonadditive_durations': api_threads,
        'native_step_labels': dict(steps),
        'limitations': ['No cross-host clock alignment.', 'Durations are Decimal differences before conversion; no float subtraction of epoch timestamps.',
                        'Nested operator/API sums are not added to graph/period spans.',
                        'Native request batch/context is not proven equal for every replay; SG has VERIFY bs=1 labels, VL actual running/graph shape still needs a separate audit.',
                        'SGLang has only one complete inter-target period per representative; profiling overhead differs. No whole-machine performance or causal attribution.'],
    }


def main():
    destination = ROOT / 'results/C32-L22-paired-trace-primary-analysis-v2.json'
    assert not destination.exists()
    rows = [analyze(*source) for source in SOURCES]
    comparisons = []
    for dp in [4, 10]:
        vl = next(r for r in rows if r['framework'] == 'vllm' and r['native_DP'] == dp)
        sg = next(r for r in rows if r['framework'] == 'sglang' and r['native_DP'] == dp)
        delta = {name: str(dec(sg[field][0]) - dec(vl[field][0])) for name, field in [
            ('period_delta_ms', 'target_graph_start_periods_ms'),
            ('graph_core_span_delta_ms', 'target_graph_core_spans_ms'),
            ('inter_graph_gap_delta_ms', 'inter_target_graph_gaps_ms')]}
        assert dec(delta['period_delta_ms']) == dec(delta['graph_core_span_delta_ms']) + dec(delta['inter_graph_gap_delta_ms'])
        comparisons.append({'native_DP': dp, 'basis': 'First observed period in each native profile; no cross-host timestamp subtraction.', **delta})
    result = {
        'project_id': 'glm53', 'scope': 'L22 representative traces only', 'model_requests': 0,
        'remote_operations': 0, 'new_profiles': 0, 'performance_certificate_allowed': False,
        'representatives': rows, 'paired_first_period_decomposition': comparisons,
        'findings': [
            'Both frameworks show 75 target MoE dispatches per replay and five eager draft MoE dispatches per target replay.',
            'The largest observed representative period difference lies outside the target graph (approximately 69.5-69.6 ms); target core-span difference is approximately 8.9-9.4 ms.',
            'SGLang eager dispatch device spans include waiting/arrival skew; target-graph dispatch medians are similar. This is not a computation-efficiency ratio.',
            'vLLM records asynchronous task-queue flows and separate CANN execution threads; SGLang does not in these traces. A task-queue candidate remains unexecuted and unproven.',
            'vLLM has MlaPrologV3 in its target graph; SGLang uses a split path. Its contribution is not isolated.'],
        'supersedes': 'v1 CANN API filters missed the AscendCL@ name prefix; all v1 graph/device results reproduce unchanged, but v1 empty API tables must not be interpreted as absent APIs.',
        'user_pause': 'No subsequent candidate, model run or profiling is authorized to auto-start while paused.',
    }
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'path': str(destination.relative_to(ROOT)), 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(), 'comparisons': comparisons}))


if __name__ == '__main__':
    main()
