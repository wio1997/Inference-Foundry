#!/usr/bin/env python3
"""Join frozen client, native consumer and SDK evidence; summarize sampled resources."""
import argparse
import json
from pathlib import Path
import re
import tarfile


def metrics(raw):
    values = {}
    for line in raw.splitlines():
        if line.startswith('#') or not line.strip():
            continue
        key, value = line.rsplit(' ', 1)
        key = key.split('{', 1)[0]
        values[key] = values.get(key, 0)+float(value)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive32', type=Path, required=True)
    parser.add_argument('--archive33', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--audit-run-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert re.fullmatch(r'[a-z0-9-]+', args.run_id)
    with tarfile.open(args.archive32) as a, tarfile.open(args.archive33) as b:
        def read(archive, path):
            return archive.extractfile(path).read().decode()

        root = 'tests/'+args.run_id+'/'
        run = json.loads(read(a, root+'result.json'))
        audit = json.loads(read(b, 'consumer-audits/'+args.audit_run_id+'/result.json'))
        client = [json.loads(s) for s in read(a, root+'client-events.jsonl').splitlines()]
        rows = []
        for event in run['proxy_records']:
            rid = event.get('consumer_id', '')
            found = [r for r in audit['requests'] if rid and (r['request_id'] == rid or r['request_id'].startswith(rid+'-'))]
            assert len(found) == 1, ('Consumer audit not unique', rid, len(found))
            row = found[0]
            expected = (event['decode_prompt_tokens']-1)//128*128
            assert row['pool_cached_tokens'] == row['need_load_tokens'] == expected > 0, (event, row)
            rows.append({'proxy_id': event['proxy_id'], 'consumer_id': rid,
                         'mode': event['mode'], 'input_tokens': row['input_tokens'],
                         'external_cached_tokens': row['pool_cached_tokens'],
                         'actually_loaded_tokens': row['need_load_tokens'],
                         'expected_rank_key_gets': expected//128*16})
        before = metrics(read(a, root+'node33-metrics-before.txt'))
        after_path = root+'node33-metrics-after.txt'
        after_available = after_path in a.getnames()
        counter = None
        if after_available:
            after = metrics(read(a, after_path))
            expected = sum(r['expected_rank_key_gets'] for r in rows)
            observed = after['vllm:ascend_store_load_get_keys_total']-before['vllm:ascend_store_load_get_keys_total']
            assert expected == observed, ('SDK gets disagree', expected, observed)
            count = after['vllm:ascend_store_load_get_duration_seconds_count']-before['vllm:ascend_store_load_get_duration_seconds_count']
            duration = after['vllm:ascend_store_load_get_duration_seconds_sum']-before['vllm:ascend_store_load_get_duration_seconds_sum']
            counter = {'expected_get_keys': expected, 'observed_get_keys': observed,
                       'rank_get_operations': count, 'mean_rank_get_ms': duration/count*1000 if count else None}
        resources = []
        for node, archive in ((32, a), (33, b)):
            data = [json.loads(s) for s in read(archive, 'resource-samples/baseline-and-handoff-v1/samples.jsonl').splitlines()]
            # These are sampled ranges over both runs, not integration or a
            # cross-host time-aligned measurement of individual requests.
            cpu, memory, hbm, ai = [], [], [], []
            names = set()
            for sample in data:
                stats = sample['docker_stats']
                if stats['rc'] == 0:
                    value = json.loads(stats['stdout'])
                    cpu.append(float(value['CPUPerc'].rstrip('%')))
                    memory.append(value['MemUsage'])
                    names.add(value['Name'])
                for line in sample['npu_smi']['stdout'].splitlines():
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) == 5 and re.fullmatch(r'\d+\s+\d+', parts[1]) and ':' in parts[2]:
                        usage = re.search(r'^(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)$', parts[3])
                        if usage:
                            ai.append(int(usage[1])); hbm.append(int(usage[2]))
            resources.append({'node': node, 'samples': len(data), 'container_names': sorted(names),
                              'first_host_utc': data[0]['host_utc'], 'last_host_utc': data[-1]['host_utc'],
                              'node32_samples_cover_client_window': (
                                  min(d['monotonic_ns'] for d in data) <= min(c['start_ns'] for c in client)
                                  and max(d['monotonic_ns'] for d in data) >= max(c['closed_ns'] for c in client)
                              ) if node == 32 else None,
                              'docker_cpu_percent_min_max': [min(cpu), max(cpu)] if cpu else None,
                              'docker_memory_usage_observed': sorted(set(memory)),
                              'per_die_hbm_MB_min_max': [min(hbm), max(hbm)] if hbm else None,
                              'per_die_aicore_percent_min_max': [min(ai), max(ai)] if ai else None,
                              'scope': '5s queries over the stated finite interval, including gaps; coverage checked with same-node monotonic time only. Docker CPU percent may exceed100 across cores. HBM includes runtime. No isolated contention attribution.'})
        report = {'run_id': args.run_id, 'benchmark_passed': run['passed'],
                  'strict_client_successes': sum(x['success'] for x in client), 'client_requests': len(client),
                  'per_request_external_KV_verified': True, 'sdk_counter_audit': counter,
                  'consumer_native_error_lines': audit['native_error_lines'],
                  'consumer_data_op_types': sorted(set(audit['hybm_data_op_types'])),
                  'requests': rows, 'summaries': run.get('summaries', []), 'resources': resources,
                  'accepted_baseline': bool(run['passed'] and counter and not audit['native_error_lines']),
                  'failure': run.get('failure'),
                  'limitations': ['C1 temporary workloads only; C4 strict output gate failed.',
                                  'No business SLO or production accuracy acceptance.',
                                  'Existing quantized cold/full and native-prefix paths can differ numerically.',
                                  'No isolated compute/communication/KV contention measurement or global optimum claim.']}
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('requests', 'summaries')}, indent=2))


if __name__ == '__main__':
    main()
