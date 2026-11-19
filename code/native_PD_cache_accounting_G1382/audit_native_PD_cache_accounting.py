"""Recompute local and connector prefix accounting for the original formal runs."""

import argparse
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import re


FORMAL_RUNS = {
    32768: 'native32_first_G691',
    65536: 'native64_first_G708',
    131072: 'native128_first_G748',
    262144: 'native256_first_G765',
}
COUNTERS = (
    'prefix_cache_queries_total', 'prefix_cache_hits_total',
    'external_prefix_cache_queries_total', 'external_prefix_cache_hits_total',
    'prompt_tokens_total', 'generation_tokens_total',
)
SERIES = re.compile(r'^vllm:(\w+)\{([^\n]*)\}\s+(\S+)$', re.M)
LABELS = {'engine': '0', 'model_name': 'ds41flash'}


def read(path):
    raw = path.read_bytes()
    return json.loads(raw), sha256(raw).hexdigest()


def counters(snapshot, role):
    metrics = snapshot[role]['metrics']
    assert metrics['status'] == 200
    values = {}
    for name, encoded_labels, value in SERIES.findall(metrics['body']):
        if name not in COUNTERS:
            continue
        labels = dict(re.findall(r'(\w+)="([^"]*)"', encoded_labels))
        if labels != LABELS:
            continue
        assert name not in values, (role, name, 'Duplicate metric series')
        num = Decimal(value)
        assert num.is_finite() and num >= 0 and num == num.to_integral_value()
        values[name] = int(num)
    assert values.keys() == set(COUNTERS), (role, 'Missing metric series')
    return values


def audit_phase(root, folder, phase, length, pin):
    run = root/'evidence'/folder/'run'
    command, command_sha = read(run/f'{phase}_{length}_command.json')
    count = command['requests']
    assert count == 32 and command['dataset_phase'] == 'formal' and command['dataset_sha256'] == pin
    phase_audit, audit_sha = read(run/f'{phase}_{length}_audit.json')
    assert phase_audit['fully_validated'] and phase_audit['requests'] == phase_audit['successful_requests'] == count
    assert (phase_audit['input_tokens'],phase_audit['output_tokens'],phase_audit['observed_max_concurrency']) == (length,1024,16)
    before, before_sha = read(run/f'before_{phase}_{length}_server_snapshot.json')
    after, after_sha = read(run/f'after_{phase}_{length}_server_snapshot.json')
    cache_audit, cache_sha = read(run/f'{phase}_{length}_cache_audit.json')
    roles = {}
    for role in ('prefill','decode'):
        baseline = counters(before,role)
        final = counters(after,role)
        change = {name:final[name]-baseline[name] for name in COUNTERS}
        assert min(change.values()) >= 0
        local = change['prefix_cache_hits_total']
        external = change['external_prefix_cache_hits_total']
        queries = change['prefix_cache_queries_total']
        assert change['external_prefix_cache_queries_total'] == queries-local
        assert queries == change['prompt_tokens_total'] and local == cache_audit[role]['delta']
        assert cache_audit[role]['verified']
        residual = queries-local-external
        assert residual >= 0
        roles[role] = {'phase_counter_deltas':change, 'logical_query_tokens_outside_recorded_cache_hits':residual}
    p = roles['prefill']['phase_counter_deltas']
    d = roles['decode']['phase_counter_deltas']
    assert (p['prompt_tokens_total'],p['generation_tokens_total']) == (count*(length-1),count)
    assert (d['prompt_tokens_total'],d['generation_tokens_total']) == (count*length,count*1024)
    assert p['external_prefix_cache_hits_total'] == 0
    assert roles['decode']['logical_query_tokens_outside_recorded_cache_hits'] == count
    if phase == 'cold':
        assert p['prefix_cache_hits_total'] == d['prefix_cache_hits_total'] == 0
    return {'requests':count,'dataset_sha256':pin,'roles':roles,
            'source_file_sha256':{'command':command_sha,'before':before_sha,'after':after_sha,
                                  'raw_count_audit':audit_sha,'local_cache_audit':cache_sha}}


def audit(root):
    results = {}
    for length, folder in FORMAL_RUNS.items():
        manifest, manifest_sha = read(root/'evidence'/folder/'run_manifest.json')
        assert manifest['lengths'] == [length]
        pin = manifest['formal_dataset_sha256'][str(length)]
        cold = audit_phase(root,folder,'cold',length,pin)
        replay = audit_phase(root,folder,'hot',length,pin)
        ext = 'external_prefix_cache_hits_total'
        external = [phase['roles']['decode']['phase_counter_deltas'][ext] for phase in (cold,replay)]
        results[str(length)] = {'native_run_evidence':folder,'manifest_sha256':manifest_sha,
                                'cold':cold,'same_input_replay':replay,
                                'D_replay_connector_logical_tokens_reduced':external[0]-external[1]}
    return {'native_epoch':630, 'engine_selector':LABELS, 'results':results,
            'scope':'Original vLLM 0.32 formal phases. Counters represent logical tokens. Connector bytes and eviction traces are outside these measurements.'}


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    cli.add_argument('--output',type=Path,required=True)
    args = cli.parse_args()
    data = audit(args.root.resolve(strict=True))
    with args.output.open('x') as stream:
        json.dump(data,stream,indent=2)
        stream.write('\n')
