"""Bind the completed 32K diagnostic to both sets of retained raw rank files."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from inventory_SG_raw_profile_g1423 import EXPECTED_RANKS, inventory


ROLES = ('prefill', 'decode')
PREFIX = 'matched_32k_g1429'
COHORT = '5cf5dbc8174409a627c8a3439f4b07e0ca99d53ac19c187ed4988eff354d0c00'
CAPTURE_CODE_SHA = 'ab831dabbff37bd6c4337737807c074a3bc78d0407350f927f4335209c336635'
INPUT_MANIFEST_SHA = '308d46f5ae05d7d30b8af9b4711b24582fe4b8a0c6fdc76f32473720f7acf4f4'


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    value = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def diagnostic_identity(control, capture, inputs):
    from bind_decode_prefix_profile_g1391 import verified_requests

    control, capture = (folder.resolve(strict=True) for folder in (control, capture))
    gate_path = control / 'gate.json'
    gate = read(gate_path)
    manifest = read(capture / 'manifest.json')
    handle = read(control / 'actual_driver_handle.json')
    state = read(capture / 'state.json')
    assert state['phase'] == 'profile_workload_complete_pending_allrank_review'
    assert state['pid'] == handle['driver_PID'] and state['actual_requests'] == 16
    assert Path(handle['capture_dir']).resolve(strict=True) == capture and handle['driver_live_at_launch']
    assert manifest['gate'] == gate
    assert digest(gate_path) == manifest['gate_sha256'] == handle['gate_SHA256']
    assert gate['profile_epoch'] == 1317 and gate['engine'] == 'sglang' and gate['ready'] is True
    assert gate['profile_cohort_distinct_from_formal_and_warmup'] is True
    assert gate['CPU_client_container_id'] == manifest['CPU_client_id']
    assert manifest['dataset_sha256'] == COHORT and manifest['source_sha256'] == CAPTURE_CODE_SHA
    assert manifest['inputs_manifest_sha256'] == INPUT_MANIFEST_SHA

    workload = read(capture / 'workload_raw_audit.json')
    assert workload['fully_validated'] is True and workload['server_usage_provenance']['verified'] is True
    assert workload['requests'] == workload['successful_requests'] == 16
    assert workload['input_tokens'] == 32768 and workload['output_tokens'] == 1024
    assert workload['configured_concurrency'] == workload['observed_max_concurrency'] == 16
    raw_root = (capture / 'aisbench').resolve(strict=True)
    assert raw_root.is_relative_to(capture)
    assert len(workload['evidence']) == 3
    for name, expected in workload['evidence'].items():
        path = (raw_root / name).resolve(strict=True)
        assert path.is_relative_to(raw_root) and digest(path) == expected
    detail, = (name for name in workload['evidence'] if name.endswith('_details.jsonl'))
    frozen_raw = inputs.read_bytes()
    assert sha256(frozen_raw).hexdigest() == INPUT_MANIFEST_SHA
    frozen = json.loads(frozen_raw)
    assert frozen['dataset_sha256'] == COHORT
    details = [json.loads(line) for line in (raw_root / detail).read_text().splitlines() if line.strip()]
    bound = verified_requests(frozen, details, 32768, 1024)
    assert len(bound) == 16

    record_files = ['manifest.json', 'state.json', 'workload_raw_audit.json']
    for role in ROLES:
        name = f'{role}_start_profile_body.json'
        body = read(capture / name)
        assert Path(body['output_dir']).parts[-2:] == (PREFIX, role)
        assert body['profile_id'] == body['profile_prefix'] == f'{PREFIX}-{role}'
        assert body['activities'] == ['CPU', 'GPU'] and body['detailed_annotations'] is True
        assert body['record_shapes'] is True and body['merge_profiles'] is False
        record_files.append(name)
        for command in ('start_profile', 'stop_profile'):
            name = f'{role}_{command}.json'
            record = read(capture / name)
            assert record['role'] == role and record['action'] == command and record['ok'] is True
            record_files.append(name)
    return {
        'epoch': 1317, 'dataset_sha256': COHORT, 'capture_prefix': PREFIX,
        'workload': {'requests': 16, 'concurrency': 16, 'input_tokens': 32768, 'output_tokens': 1024},
        'bound_input_records': len(bound),
        'input_manifest_sha256': INPUT_MANIFEST_SHA,
        'control_record_sha256': {'gate.json': digest(gate_path),
                                  'actual_driver_handle.json': digest(control / 'actual_driver_handle.json')},
        'capture_record_sha256': {name: digest(capture / name) for name in record_files},
        'raw_request_files_sha256': workload['evidence'],
    }


def identity_hash(info):
    assert info['epoch'] == 1317 and info['dataset_sha256'] == COHORT and info['capture_prefix'] == PREFIX
    packed = json.dumps(info, sort_keys=True, separators=(',', ':')).encode()
    return sha256(packed).hexdigest()


def role_receipt(profile_parent, role, info):
    parent = profile_parent.resolve(strict=True)
    path = parent / info['capture_prefix'] / role
    source = path.resolve()
    assert path == source and source.is_relative_to(parent)
    return {'role': role, 'capture_identity_sha256': identity_hash(info), **inventory(source)}


def pair_receipt(info, reports):
    scope = identity_hash(info)
    assert len(reports) == 2 and {row['role'] for row in reports} == set(ROLES)
    all_raw, all_parsed = True, True
    for report in reports:
        assert report['capture_identity_sha256'] == scope
        claimed = set(report['raw_cpu_and_device_ranks'])
        assert len(claimed) == len(report['raw_cpu_and_device_ranks'])
        rank_rows = report['ranks']
        assert len({row['rank'] for row in rank_rows}) == len(rank_rows)
        assert claimed == {row['rank'] for row in rank_rows if row['raw_cpu_and_device_files_present']}
        assert (claimed == EXPECTED_RANKS) == report['all8_raw_file_presence_verified']
        parsed = set(report['parsed_db_file_ranks'])
        assert parsed == {row['rank'] for row in rank_rows if row['parsed_DB_file_present']}
        all_raw &= claimed == EXPECTED_RANKS
        all_parsed &= parsed == EXPECTED_RANKS
    return {
        'capture_identity_sha256': scope, 'paired_8_plus_8_raw_files_present': all_raw,
        'paired_8_plus_8_parsed_DB_files_present': all_parsed,
        'role_status': {row['role']: row for row in reports},
        'scope': ('Trace-file presence and frozen workload identity. Parsed event coverage, same-work '
                  'matching, and trace metric acceptance require their own review.'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    client = commands.add_parser('capture')
    client.add_argument('--control', type=Path, required=True)
    client.add_argument('--capture', type=Path, required=True)
    client.add_argument('--inputs', type=Path, required=True)
    role = commands.add_parser('role')
    role.add_argument('--parent', type=Path, required=True)
    role.add_argument('--role', choices=ROLES, required=True)
    role.add_argument('--identity', type=Path, required=True)
    pair = commands.add_parser('pair')
    pair.add_argument('--identity', type=Path, required=True)
    pair.add_argument('--reports', type=Path, nargs=2, required=True)
    args = parser.parse_args()
    if args.command == 'capture':
        report = diagnostic_identity(args.control, args.capture, args.inputs)
    elif args.command == 'role':
        report = role_receipt(args.parent, args.role, read(args.identity))
    else:
        report = pair_receipt(read(args.identity), [read(path) for path in args.reports])
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
