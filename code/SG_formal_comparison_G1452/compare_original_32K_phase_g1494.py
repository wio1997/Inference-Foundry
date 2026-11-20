"""Compare one independently preserved G1317 phase against the accepted native phase."""

import argparse
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path

from build_formal_PD_matrix_g1452 import (
    METRICS, NATIVE, NATIVE_CACHE, native_metrics, record, validate_report, validated_request_shape,
)


RUN_ID = 'sg_pd_g1317_aisbench_32k_g1324'
DATA_SHA = '2484713fafb09065e6426717d1f07fa26cd3e7de4d771e734c60ec2511b94ffe'
GATE_SHA = '81b675d4f4ae9929564b68c45586868da26d0ce8883488249771e1b2db653d5d'
USAGE_SHA = 'df9026cdbb4228219051d97694eb95164009856cebc87858022ebc8dedf60fc4'
INPUT_TOKENS = 32768
REQUESTS = 32
PHASE_NAMES = {'cold': 'cold', 'hot': 'same_input_replay'}


def digest(path):
    h = sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def phase_metrics(audit):
    if not validated_request_shape(audit, INPUT_TOKENS, REQUESTS):
        raise ValueError('The separate phase requires verified server usage and all 32 requests at C16/O1024')
    stats = {name: audit[name] for name in METRICS}
    if not all(isfinite(v) and v > 0 for v in stats.values()):
        raise ValueError('Phase latency, throughput and duration values must be positive and finite')
    for metric, tokens in (('input_tokens_per_s', INPUT_TOKENS), ('output_tokens_per_s', 1024),
                           ('total_tokens_per_s', INPUT_TOKENS + 1024)):
        if not isclose(stats[metric], REQUESTS * tokens / stats['duration_s'], rel_tol=1e-5, abs_tol=0.01):
            raise ValueError(f'{metric} disagrees with exact phase tokens and elapsed time')
    return stats


def original_phase(folder, phase):
    receipt_path = folder / f'{phase}_preservation_verified.json'
    receipt, receipt_digest = record(receipt_path)
    assert (receipt['run_id'], receipt['epoch'], receipt['phase']) == (RUN_ID, 1317, phase)
    if receipt['raw_phase_verified'] is not True or receipt['dataset_sha256'] != DATA_SHA:
        raise ValueError('The original phase preservation verifier has no accepted result for this cohort')
    durability = receipt['durable_phase_event']
    assert durability['event'] == 'durable_phase' and durability['finished_phase'] == phase
    assert durability['run_id'] == RUN_ID
    assert durability['durable_files'][f'{phase}_32768_audit.json']
    assert durability['durable_files'][f'{phase}_32768_compile_audit.json']
    archive = folder / f'{phase}_artifacts.tar'
    assert 0 < archive.stat().st_size <= 49 * 1024 * 1024
    assert digest(archive) == receipt['artifact_tar_sha256']
    exact = receipt['per_request']
    assert exact['exact_frozen_prompts'] == exact['sqlite_reference_rows'] == REQUESTS
    assert receipt['prior_warmups_reaudited'] == 5 and receipt['D_persistent_compile_cache_check'] is True

    root = folder / f'{phase}_verified'
    inv, _ = record(root / 'preservation_inventory.json')
    assert (inv['run_id'], inv['phase'], inv['gate_sha256'], inv['dataset_sha256']) == (
        RUN_ID, phase, GATE_SHA, DATA_SHA
    )
    assert digest(root / 'formal_32768.jsonl') == DATA_SHA
    assert digest(root / 'formal_gate.json') == GATE_SHA
    assert digest(root / 'usage' / 'usage_contract.json') == USAGE_SHA
    audit, audit_digest = record(root / 'audit.json')
    assert audit['server_usage_provenance']['contract_sha256'] == USAGE_SHA
    metrics = phase_metrics(audit)
    assert metrics == receipt['metrics']
    files = audit['evidence']
    assert len(files) == 3
    details, = [name for name in files if name.endswith('_details.jsonl')]
    summaries = {Path(details).with_name(Path(details).name.removesuffix('_details.jsonl') + suffix).as_posix()
                 for suffix in ('.json', '.csv')}
    assert set(files) == summaries | {details}
    raw_root = (root / 'raw').resolve(strict=True)
    for name, checksum in files.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        actual = (raw_root / name).resolve(strict=True)
        assert actual.is_relative_to(raw_root) and digest(actual) == checksum
    databases = list((raw_root / details).parent.joinpath('db_data').glob('*.db'))
    database, = databases
    assert not any(database.with_name(database.name + ext).exists() for ext in ('-wal', '-shm', '-journal'))
    assert digest(database) == exact['timing_database_sha256']

    cache, _ = record(root / 'cache_audit.json')
    p = cache['prefill']
    assert p['verified'] is True and p['delta'] == receipt['P_local_hit_token_delta'] >= 0
    if phase == 'cold':
        assert p['delta'] == 0
    else:
        command, _ = record(root / 'command.json')
        prior, _ = record(root / 'cold_command.json')
        assert command['dataset_sha256'] == prior['dataset_sha256'] == DATA_SHA
        assert command['config_sha256'] == prior['config_sha256']
    return {'metrics': metrics, 'P_local_device_hit_tokens': p['delta'],
            'D_independent_local_hit_tokens': None, 'phase_receipt_sha256': receipt_digest,
            'phase_audit_sha256': audit_digest, 'artifact_tar_sha256': receipt['artifact_tar_sha256'],
            'phase_raw_evidence_sha256': sorted(files.values()),
            'request_timing_database_sha256': exact['timing_database_sha256'],
            'server_usage_contract_sha256': USAGE_SHA}


def native_phase(vllm_evidence, phase):
    directory, pin = NATIVE[INPUT_TOKENS]
    run, _ = record(vllm_evidence / directory / 'audit_results.json', pin)
    profile = validate_report(run, 'vllm', INPUT_TOKENS)
    assert profile['dataset_sha256'] == DATA_SHA
    cache, checksum = record(vllm_evidence / NATIVE_CACHE[0], NATIVE_CACHE[1])
    counters = cache['results'][str(INPUT_TOKENS)]
    assert cache['native_epoch'] == 630 and counters['native_run_evidence'] == directory
    return {**native_metrics(profile, counters, PHASE_NAMES[phase]), 'run_id': run['run_id'],
            'run_audit_sha256': pin, 'effective_cache_audit_sha256': checksum}


def compare(folder, phase, vllm_evidence):
    native = native_phase(vllm_evidence, phase)
    sglang = original_phase(folder, phase)
    return {'phase': PHASE_NAMES[phase], 'cohort_sha256': DATA_SHA, 'input_tokens': INPUT_TOKENS,
            'output_tokens': 1024, 'concurrency': 16, 'requests': REQUESTS, 'SGLang_run_id': RUN_ID,
            'SGLang': sglang, 'vLLM': native,
            'SGLang_divided_by_vLLM': {k: sglang['metrics'][k] / native['metrics'][k] for k in METRICS},
            'scope': 'One first formal phase, with its own accepted raw request and server usage audit. '
                     'Paired cold and replay acceptance, GSM8K and matched device profiling are tracked separately.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preservation-dir', type=Path, required=True)
    parser.add_argument('--vllm-evidence', type=Path, required=True)
    parser.add_argument('--phase', choices=PHASE_NAMES, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.preservation_dir, args.phase, args.vllm_evidence), indent=2, sort_keys=True))
