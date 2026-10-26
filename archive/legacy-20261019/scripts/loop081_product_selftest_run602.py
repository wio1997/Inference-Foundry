#!/usr/bin/env python3
"""CPU synthetic admission and corruption negatives for Run602 validator."""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
OLD = ROOT / 'evidence/20260928_loop081_bound/run597/live/b'
VALIDATOR = ROOT / 'scripts/loop081_product_validate_run602.py'
RUN_TS = 'LOOP081-RUN602-B'
TIME_NS = 'time:[selftest]'


def load(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, separators=(',', ':')) + '\n')


def output_from_basis(basis, slot):
    out = []
    for tokens, counts in zip(basis['token_history'], basis['count_history']):
        out.extend(tokens[slot][:counts[slot]][:1024-len(out)])
        if len(out) == 1024:
            break
    assert len(out) == 1024
    return out


def run_validator(arm):
    return subprocess.run(
        [sys.executable, str(VALIDATOR), '--arm-dir', str(arm),
         '--run-ts', RUN_TS, '--output', str(arm / 'product_admission.json')],
        cwd=ROOT, text=True, capture_output=True)


def main():
    results = {}
    with tempfile.TemporaryDirectory(prefix='run602_product_selftest_') as temp:
        arm = Path(temp)
        for name in ('warmup_client', 'measured_client'):
            (arm / name).symlink_to(OLD / name, target_is_directory=True)
        shutil.copytree(OLD / 'runtime', arm / 'runtime')
        shutil.copytree(OLD / 'basis', arm / 'basis')
        for path in (arm / 'basis').glob('*.json'):
            row = load(path)
            row['run_ts'] = RUN_TS
            write(path, row)
        shutil.copy2(OLD / 'client_admission.json', arm / 'client_admission.json')
        admission = load(OLD / 'basis_admission.json')
        admission['run_ts'] = RUN_TS
        source_paths = [arm / 'client_admission.json'] + list((arm / 'runtime').glob('*.json')) + list((arm / 'basis').glob('*.json'))
        admission['source_sha256'] = {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        }
        write(arm / 'basis_admission.json', admission)
        product = arm / 'product'
        product.mkdir()
        (product / 'arm').touch()
        client_admission = load(OLD / 'client_admission.json')
        by_response = {x['response_id']: x for x in client_admission['request_index']
                       if x['phase'] == 'measured'}
        client_raw = {}
        for path in (OLD / 'measured_client').glob('request_*.json'):
            item = load(path)
            client_raw[item['row']['response_id']] = item
        assert len(by_response) == len(client_raw) == 48
        bulk_index = 0
        for cohort in range(5, 9):
            basis = load(OLD / 'basis' / f'rank0_cohort{cohort}.json')
            ids = basis['req_ids']
            assert len(ids) == 12
            for rank in range(8):
                cached = [{'req_id': rid} for rid in ids]
                marks = [
                    {'kind': 'execute_entry', 'rank': rank, 't_ns': 1000,
                     'total_scheduled_tokens': 996, 'new': cached, 'cached': []},
                    {'kind': 'execute_entry', 'rank': rank, 't_ns': 2000,
                     'total_scheduled_tokens': 96, 'new': [], 'cached': cached},
                    {'kind': 'runtime_built_host', 'rank': rank, 't_ns': 2100, 'req_ids': ids},
                    {'kind': 'serve_start_host', 'rank': rank, 't_ns': 2200, 'req_ids': ids},
                    {'kind': 'serve_synced_host', 'rank': rank, 't_ns': 3000, 'req_ids': ids},
                ]
                calls = [
                    {'kind': 'target_forward', 'execute_mark_count': 1,
                     'host_start_ns': 1100, 'host_end_ns': 1200,
                     'current_stream_elapsed_ms': 0.1},
                    {'kind': 'propose_and_optional_copy', 'execute_mark_count': 1,
                     'host_start_ns': 1300, 'host_end_ns': 1400,
                     'current_stream_elapsed_ms': 0.1, 'req_ids': ids},
                ]
                write(product / f'rank{rank}_cohort{cohort}.json',
                      {'rank': rank, 'cohort': cohort, 'run_ts': RUN_TS,
                       'time_namespace': TIME_NS, 'marks': marks, 'calls': calls})
            bulk_index += 1
            bulk_rows = []
            for slot, rid in enumerate(ids):
                response = next(x for x in by_response if rid.startswith(x + '-'))
                tokens = output_from_basis(basis, slot)
                bulk_rows.append({'req_id': rid, 'pre_ids': [], 'pre_placeholders': 0,
                                  'incoming_bulk_ids': tokens, 'accepted_bulk_ids': tokens,
                                  'post_ids': tokens, 'post_placeholders': 0,
                                  't_ns': 2300, 'post_ns': 2400})
                first_event = next(x for x in client_raw[response]['events'] if not x.get('done'))
                api = {'kind': 'api_stream', 'run_ts': RUN_TS, 'time_namespace': TIME_NS,
                       'request_id': response, 'previous_num_tokens': [1024],
                       'events': [{'choice': 0, 'engine_seen_ns': 2400,
                                   'token_ids': tokens, 'cum_output_tokens': 1024,
                                   'sse_yield_ns': 2600,
                                   'sse_payload_sha256': first_event['payload_sha256']}],
                       'before_done_ns': 3500}
                name = hashlib.sha256(response.encode()).hexdigest()[:24]
                write(product / f'api_{name}.json', api)
            write(product / f'scheduler_bulk_{bulk_index}.json',
                  {'kind': 'bulk_step', 'run_ts': RUN_TS,
                   'time_namespace': TIME_NS, 'write_begin_ns': 2500,
                   'rows': bulk_rows})
        positive = run_validator(arm)
        if positive.returncode != 0:
            raise AssertionError(f'positive failed: {positive.stderr[-2000:]}')
        results['positive'] = 'pass'
        mutations = [
            ('wrong_scheduler_prefix', product / 'scheduler_bulk_1.json',
             lambda x: x['rows'][0]['pre_ids'].append(1)),
            ('wrong_api_token', next(product.glob('api_*.json')),
             lambda x: x['events'][0]['token_ids'].__setitem__(0, -1)),
            ('wrong_api_hash', next(product.glob('api_*.json')),
             lambda x: x['events'][0].__setitem__('sse_payload_sha256', '0'*64)),
            ('missing_proposal', product / 'rank0_cohort5.json',
             lambda x: x['calls'].pop()),
            ('wrong_time_namespace', product / 'rank0_cohort5.json',
             lambda x: x.__setitem__('time_namespace', 'time:[other]')),
        ]
        for name, path, mutate in mutations:
            raw = path.read_bytes()
            modified = json.loads(raw)
            mutate(modified)
            write(path, modified)
            trial = run_validator(arm)
            if trial.returncode == 0:
                raise AssertionError(f'{name} was accepted')
            results[name] = 'rejected'
            path.write_bytes(raw)
        # Change all downstream lists consistently so only the basis join catches it.
        bulk_path = product / 'scheduler_bulk_1.json'
        bulk_raw = bulk_path.read_bytes()
        bulk = json.loads(bulk_raw)
        target_rid = bulk['rows'][0]['req_id']
        response = next(x for x in by_response if target_rid.startswith(x + '-'))
        api_name = hashlib.sha256(response.encode()).hexdigest()[:24]
        api_path = product / f'api_{api_name}.json'
        api_raw = api_path.read_bytes()
        api = json.loads(api_raw)
        for key in ('incoming_bulk_ids', 'accepted_bulk_ids', 'post_ids'):
            bulk['rows'][0][key][0] = -1
        api['events'][0]['token_ids'][0] = -1
        write(bulk_path, bulk)
        write(api_path, api)
        trial = run_validator(arm)
        if trial.returncode == 0 or 'Runtime basis vs scheduler incoming bulk IDs' not in trial.stderr:
            raise AssertionError('consistent downstream wrong_basis_token did not fail the basis join')
        results['wrong_basis_token'] = 'rejected'
        bulk_path.write_bytes(bulk_raw)
        api_path.write_bytes(api_raw)
        # A synthetic admission with one false current-arm source digest must fail.
        local_admission = arm / 'basis_admission.json'
        bad = load(local_admission)
        first = next(iter(bad['source_sha256']))
        bad['source_sha256'][first] = '0'*64
        write(local_admission, bad)
        trial = run_validator(arm)
        if trial.returncode == 0:
            raise AssertionError('wrong source hash was accepted')
        results['wrong_source_hash'] = 'rejected'
    output = ROOT / 'evidence/20260928_loop081_bound/run602/preflight/selftest.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'status': 'pass', 'results': results,
                                  'validator_sha256': hashlib.sha256(VALIDATOR.read_bytes()).hexdigest(),
                                  'selftest_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + '\n')
    print(json.dumps({'status': 'pass', 'negative_rejected': len(results)-1}))


if __name__ == '__main__':
    main()
