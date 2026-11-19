"""Reconcile actual G1317 rank-zero pool gauges to startup token capacities."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re


METRICS_SHA = 'c47ad43b322ad85127b94f641056ccd1bab4416692a9040d7c5d069fb0d7116f'
P_SOURCE_SHA = '3e2c2902ca5dde7297b39f88f1726b85081468b98cbade59f02ef2309b10a136'
D_SOURCE_SHA = 'd25af01bc0b737397394e3f6bf3711ab9a25d1f2aa8592360ef9d9637047add2'
DATASET_MANIFEST_SHA = '2491233b84c38ed476d885d8379d464af728b95ea810a4428e2ad65ca076513a'
FORMAL_DATASET_SHA = '2484713fafb09065e6426717d1f07fa26cd3e7de4d771e734c60ec2511b94ffe'
BASELINE_SHA = '3639fe8a522b719d9e36fa0145d38ce762cfad6b3c7b01a6f1f21d64d4e2bf87'
STARTUP_CAPACITY = re.compile(r'DSV4 pool sizes: full=(\d+), swa=(\d+),')
POOL_STATS = 'managers/scheduler_components/pool_stats_observer.py'


def pinned(path, digest):
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == digest
    return json.loads(raw)


def receipt(path, digest, marker):
    packet = pinned(path, digest)
    assert packet['exitstatus'] == 0 and packet['outcome'] == 'finished'
    line, = (x.removeprefix(marker+' ') for x in packet['output'].splitlines()
             if x.startswith(marker+' '))
    data = json.loads(line)
    assert data['epoch'] == 1317
    return data


def metric(scrape, name, role, label=None):
    prefix = f'sglang:{name}{{'
    scope = 'cache_type="UnifiedRadixCache"' if name == 'evicted_tokens_total' else f'engine_type="{role}"'
    value, = (number for key, number in scrape.items() if key.startswith(prefix)
              and 'tp_rank="0"' in key and scope in key
              and (label is None or f'mode="{label}"' in key))
    assert math.isfinite(value) and value >= 0
    return value


def pool(scrape, role, metric_prefix, capacity, usage_name):
    order = ('available', 'evictable', 'used')
    values = {}
    for field in order:
        point = metric(scrape, f'{metric_prefix}_{field}_tokens', role)
        assert int(point) == point
        values[field] = int(point)
    assert sum(values.values()) == capacity
    active_fraction = metric(scrape, usage_name, role)
    assert math.isclose(active_fraction, values['used'] / capacity, rel_tol=1e-10)
    return {**values, 'active_usage': active_fraction,
            'active_plus_evictable_fraction': (values['used'] + values['evictable']) / capacity}


def audit(measurement, sources, manifest):
    assert measurement['current_phase'] == 'cold' and measurement['runner_PID'] == 3090471
    assert measurement['run_id'] == 'sg_pd_g1317_aisbench_32k_g1324'
    assert measurement['baseline_sha256'] == BASELINE_SHA
    assert 'unique first128token page per phase/request/length' in manifest['construction']
    dataset, = (row for row in manifest['profiles'] if row['sha256'] == FORMAL_DATASET_SHA)
    assert (dataset['phase'], dataset['input_tokens'], dataset['requests']) == ('formal', 32768, 32)

    result = {}
    for role, source in sources.items():
        assert source['role'] == role
        capacity, = {tuple(map(int, m.groups())) for line in source['startup_quantity_rows']
                     if (m := STARTUP_CAPACITY.search(line))}
        full, swa = capacity
        assert full % 128 == swa % 128 == 0
        collector = source['source_summary'][POOL_STATS]
        assert any('full_token_usage = full_num_used / full_capacity' in line
                   for _, line in collector['selected'])
        samples = measurement['roles'][role]
        pools = {}
        for phase, scrape in samples.items():
            pools[phase] = {'full': pool(scrape, role, 'kv', full, 'full_token_usage'),
                            'SWA': pool(scrape, role, 'swa', swa, 'swa_token_usage')}
        evicted = [metric(samples[phase], 'evicted_tokens_total', role)
                   for phase in ('previous', 'current')]
        assert evicted[1] >= evicted[0]
        result[role] = {
            'rank': 0, 'full_pool_capacity': full, 'SWA_pool_capacity': swa,
            'pool_gauges': pools,
            'eviction_counter_delta': int(evicted[1] - evicted[0]),
            'pool_observer_source_sha256': collector['sha256'],
            'startup_log_prefix_sha256': source['startup_log_sha256'],
            'max_independent_complete_32768_token_prefixes_if_all_full_slots_available': full // 32768,
        }

    p = measurement['roles']['prefill']
    phases = ('previous', 'current')
    p_effective = {}
    for mode in ('input', 'device_hit', 'host_hit', 'storage_hit'):
        old, new = (metric(p[phase], 'prefill_effective_tokens_total', 'prefill', mode) for phase in phases)
        assert new >= old
        p_effective[mode] = int(new - old)
    assert p_effective == {'input': 278528, 'device_hit': 0, 'host_hit': 0, 'storage_hit': 0}
    return {
        'epoch': 1317, 'cold_phase_snapshot_at_UTC': measurement['at_UTC'],
        'rank_zero_pools': result, 'prefill_effective_tokens_delta': p_effective,
        'formal_dataset_sha256': FORMAL_DATASET_SHA,
        'formal_cohort_tokens_in_independent_complete_prefixes': 32 * 32768,
        'cold_or_replay_score_verified': False, 'decode_per_request_local_replay_hits_verified': False,
        'scope': ('Rank-zero startup capacities and live gauge snapshots. Each capacity bound assumes an '
                  'entire full pool is available and first 128-token pages are independent. '
                  'It is a simultaneous resident full-prefix bound. Eviction deltas are slot counters.'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('metrics', 'prefill_source', 'decode_source', 'dataset_manifest'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    metrics = receipt(args.metrics, METRICS_SHA, 'G1471_COLD_CACHE_METRICS')
    source = {role: receipt(path, digest, 'G1471_CACHE_CAPACITY_SOURCES')
              for role, path, digest in (('prefill', args.prefill_source, P_SOURCE_SHA),
                                         ('decode', args.decode_source, D_SOURCE_SHA))}
    result = audit(metrics, source, pinned(args.dataset_manifest, DATASET_MANIFEST_SHA))
    result['input_artifact_sha256'] = {'metrics': METRICS_SHA, 'prefill_source': P_SOURCE_SHA,
                                      'decode_source': D_SOURCE_SHA, 'dataset_manifest': DATASET_MANIFEST_SHA}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
