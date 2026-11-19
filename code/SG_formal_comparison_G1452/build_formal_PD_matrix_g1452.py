"""Pair terminal AISBench phase reports with frozen G630 native evidence."""

import argparse
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path


NATIVE = {
    32768: ('native32_first_G691', '07e9ad3d8b07830d10e59bf86065b26fc5f0f4e9471fde687cad58aa735fd549'),
    65536: ('native64_first_G708', 'a6f4abcac980c419b95c4a5c9b9ee9619edbcd616375057bef6a4d10007e6050'),
    131072: ('native128_first_G748', '224b57cb56864f2883a021fea1e56c825cda294a736ac15988ca835d47a37ab9'),
    262144: ('native256_first_G765', '7edef91c053e8704ff6265d753e10f5758f21c0234a6d7034cb1183b630278a9'),
}
NATIVE_CACHE = ('native_cache_G1382/G630_PD_native_formal_prefix_accounting.json',
                'df75ab06043e8623dc00c9d142aef7522cd38f61f29cc214f8bdcd7e8917849a')
STAGES = {'cold': 'cold', 'same_input_replay': 'hot_replay'}
PREFILL_ONLY_COUNTER = 'sglang:prefill_effective_tokens_total'
METRICS = ('duration_s', 'ttft_mean_ms', 'ttft_p90_ms', 'ttft_p99_ms', 'tpot_mean_ms', 'tpot_p90_ms',
           'tpot_p99_ms', 'input_tokens_per_s', 'output_tokens_per_s', 'total_tokens_per_s')


def record(path, expected=None):
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError('Authoritative input digest mismatch')
    return json.loads(raw), digest


def validated_request_shape(phase, length, requests):
    return (phase['fully_validated'] is True and phase['server_usage_provenance']['verified'] is True
            and phase['requests'] == phase['successful_requests'] == requests
            and phase['input_tokens'] == length and phase['output_tokens'] == 1024
            and phase['configured_concurrency'] == phase['observed_max_concurrency'] == 16)


def validate_report(report, engine, length):
    if report['engine'] != engine or report['all_required_raw_metrics_and_usage_validated'] is not True:
        raise ValueError('The campaign has no accepted terminal raw audit')
    row, = [row for row in report['profiles'] if row['input_tokens'] == length]
    if row['status'] != 'cold_hot_raw_metrics_validated':
        raise ValueError('The target length has no fully validated pair')
    warmups = row['unscored_warmup_passes']
    if len(warmups) != 5 or not all(validated_request_shape(s, length, 16) for s in warmups):
        raise ValueError('Expected five valid independent warmups')
    for key in STAGES.values():
        phase = row[key]
        if not validated_request_shape(phase, length, 32) or len(phase['evidence']) != 3:
            raise ValueError(f'{key} is missing an accepted 32-request C16 phase or its raw evidence')
        values = {name: phase[name] for name in METRICS}
        if not all(isfinite(value) and value > 0 for value in values.values()):
            raise ValueError(f'{key} metrics require positive finite values')
        duration = values['duration_s']
        for key, tokens in (('input_tokens_per_s', length), ('output_tokens_per_s', 1024),
                            ('total_tokens_per_s', length + 1024)):
            if not isclose(values[key], 32 * tokens / duration, rel_tol=1e-5, abs_tol=0.01):
                raise ValueError(f'{key} does not match the actual tokens and duration')
    return row


def readable_metrics(profile, stage):
    phase = profile[STAGES[stage]]
    return {'metrics': {name: phase[name] for name in METRICS},
            'phase_raw_evidence_sha256': sorted(phase['evidence'].values()),
            'server_usage_contract_sha256': phase['server_usage_provenance']['contract_sha256']}


def native_metrics(profile, cache, stage):
    reading = readable_metrics(profile, stage)
    reference = cache[stage]
    assert reference['dataset_sha256'] == profile['dataset_sha256'] and reference['requests'] == 32
    reading['prefix_counters'] = {}
    for role, status in reference['roles'].items():
        delta = status['phase_counter_deltas']
        original = profile['cache_counter_evidence']['cold' if stage == 'cold' else 'hot'][role]
        assert original['verified'] is True and original['delta'] == delta['prefix_cache_hits_total']
        reading['prefix_counters'][role] = {
            'local_hit_tokens': delta['prefix_cache_hits_total'],
            'connector_hit_tokens': delta['external_prefix_cache_hits_total'],
            'local_scope': 'Native vLLM engine 0 local prefix hits',
            'connector_scope': 'Native vLLM engine 0 external logical prefix hits',
        }
    return reading


def sglang_metrics(profile, stage):
    reading = readable_metrics(profile, stage)
    scope = profile['cache_counter_evidence']['cold' if stage == 'cold' else 'hot']
    reading['prefix_counters'] = {}
    for role in ('prefill', 'decode'):
        observation = scope.get(role, {})
        source = observation.get('counter')
        direct = (observation.get('verified') is True and source
                  and not (role == 'decode' and source == PREFILL_ONLY_COUNTER))
        if direct and (not isfinite(observation['delta']) or observation['delta'] < 0):
            raise ValueError('Cache hit delta must be finite and nonnegative')
        reading['prefix_counters'][role] = {
            'local_hit_tokens': observation['delta'] if direct else None,
            'local_scope': source if direct else 'No independent local hit counter in this audit',
            'connector_hit_tokens': None,
            'connector_scope': 'No token-equivalent connector counter in this audit',
        }
    prefill = reading['prefix_counters']['prefill']['local_hit_tokens']
    if stage == 'cold' and prefill != 0:
        raise ValueError('Cold evidence requires the verified zero-hit Prefill counter')
    return reading


def build(native_root, sglang_files):
    cache_report, cache_sha = record(native_root / NATIVE_CACHE[0], NATIVE_CACHE[1])
    assert cache_report['native_epoch'] == 630
    tiers = {}
    for length, (directory, pin) in NATIVE.items():
        native, _ = record(native_root / directory / 'audit_results.json', pin)
        reference = validate_report(native, 'vllm', length)
        cache = cache_report['results'][str(length)]
        assert cache['native_run_evidence'] == directory
        phases = {phase: {'vLLM': native_metrics(reference, cache, phase), 'SGLang': None,
                          'SGLang_divided_by_vLLM': None} for phase in STAGES}
        tier = {'input_tokens': length, 'cohort_sha256': reference['dataset_sha256'],
                'vLLM_run_id': native['run_id'], 'vLLM_audit_sha256': pin, 'phases': phases,
                'SGLang_evidence_status': 'awaiting_accepted_terminal_audit'}
        if length in sglang_files:
            try:
                if not sglang_files[length].is_file():
                    raise ValueError('The supplied SGLang audit file is missing')
                sg, digest = record(sglang_files[length])
                tier['SGLang_supplied_audit_sha256'] = digest
                profile = validate_report(sg, 'sglang', length)
                if profile['dataset_sha256'] != reference['dataset_sha256']:
                    raise ValueError('The two formal input manifests have different dataset digests')
                actual = {phase: sglang_metrics(profile, phase) for phase in STAGES}
            except (KeyError, TypeError, ValueError) as failure:
                tier.update(SGLang_evidence_status='supplied_audit_rejected',
                            SGLang_rejection_reason=str(failure))
            else:
                for phase, data in phases.items():
                    data['SGLang'] = actual[phase]
                    data['SGLang_divided_by_vLLM'] = {
                        name: actual[phase]['metrics'][name] / data['vLLM']['metrics'][name]
                        for name in METRICS}
                    hit = tuple(data[framework]['prefix_counters'][role]['local_hit_tokens']
                                for framework in ('SGLang', 'vLLM') for role in ('prefill', 'decode'))
                    data['reported_P_and_D_local_hit_counts_equal'] = (
                        None if None in hit else hit[:2] == hit[2:])
                tier.update(SGLang_evidence_status='cold_and_replay_results_reportable',
                            SGLang_run_id=sg['run_id'])
        tiers[str(length)] = tier
    return {'requests_per_scored_phase': 32, 'peak_concurrency': 16, 'output_tokens_per_request': 1024,
            'vLLM_cache_evidence_sha256': cache_sha, 'tiers': tiers,
            'scope': 'Ratios describe one validated pair at the same input digest and C16/O1024. '
                     'Cache counters, runtime settings, quality and replay repeatability require their own evidence.'}


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--vllm-evidence', type=Path, required=True)
    cli.add_argument('--sglang-audit', nargs=2, action='append', metavar=('INPUT_TOKENS', 'FILE'), default=[])
    args = cli.parse_args()
    inputs = {int(width): Path(file) for width, file in args.sglang_audit}
    if len(inputs) != len(args.sglang_audit) or set(inputs) - set(NATIVE):
        cli.error('Provide at most one audit per supported input length')
    print(json.dumps(build(args.vllm_evidence, inputs), indent=2, sort_keys=True))
