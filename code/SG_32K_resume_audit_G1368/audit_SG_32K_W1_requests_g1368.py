"""Re-audit the original 32K AISBench warmup after its runner and worker exit."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys


RUN_ID = 'sg_pd_g1317_aisbench_32k_g1324'
INPUT = 32768
OUTPUT = 1024
COUNT = 16


def load(path):
    return json.loads(path.read_bytes())


def summarize_requests(rows):
    per_request = [{key: row[key] for key in ('id', 'success', 'input_tokens', 'output_tokens')} for row in rows]
    counts = Counter(row['id'] for row in per_request if type(row['id']) is int)
    exact = sum(row['success'] is True and row['input_tokens'] == INPUT and row['output_tokens'] == OUTPUT
                for row in per_request)
    expected = set(range(COUNT))
    return {'requests':per_request, 'request_IDs_exactly_once':len(rows) == COUNT and counts == Counter(expected),
            'exact_shape_successes':exact, 'missing_request_IDs':sorted(expected-counts.keys()),
            'repeated_request_IDs':sorted(key for key,count in counts.items() if count > 1)}


def still_running(pid, *expected):
    try:
        cmdline = Path('/proc', str(pid), 'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return False
    return all(str(item).encode() in cmdline for item in expected)


def audit_warmup(client):
    run = client/'runs'/'sglang'/RUN_ID
    pipeline = client/'benchmark_pipeline_SG_g1183'
    auditor = pipeline/'audit_aisbench.py'
    manifest_path = run/'run_manifest.json'
    manifest = load(manifest_path)
    state_path = run/'state.json'
    state = load(state_path)
    command = load(run/'warmup1_32768_command.json')
    assert manifest['run_id'] == state['run_id'] == RUN_ID and manifest['lengths'] == [INPUT]
    assert manifest['host_runner_pid'] == 3019382 and manifest['warmup_passes'] == 5
    assert command['requests'] == COUNT and command['dataset_phase'] == 'warmup1'
    assert command['argv'][command['argv'].index('--num-prompts')+1] == str(COUNT)
    assert not any(path.name.startswith(('cold_', 'hot_', 'before_cold_', 'before_hot_')) for path in run.iterdir())
    container = manifest['client_container_id']
    rows = subprocess.check_output(['docker', 'top', container, '-eo', 'pid,args'], text=True, timeout=8).splitlines()[1:]
    workers = [int(row.split()[0]) for row in rows if 'ais_bench' in row and '/sglang_warmup1_32768.py' in row]
    outer = still_running(manifest['host_runner_pid'], pipeline/'run_aisbench_cold_hot_g1294.py', RUN_ID)
    prefix = {'at':datetime.now(timezone.utc).isoformat(),'run_id':RUN_ID, 'phase':state['phase'],
              'original_runner_live':outer,'warmup_worker_PIDs':workers,'request_count':COUNT}
    if outer or workers:
        return {**prefix, 'classification':'in_flight'}
    details = list((run/'warmup1_32768').rglob('*_details.jsonl'))
    if len(details) != 1:
        return {**prefix, 'classification':'terminal_requires_review', 'details_file_count':len(details)}
    detail = details[0]
    raw = detail.read_bytes()
    request_report = summarize_requests([json.loads(line) for line in raw.splitlines() if line.strip()])
    result = {**prefix, **request_report, 'raw_details_SHA256':sha256(raw).hexdigest()}
    if (state['phase'] != 'warmup_paused_for_P_snapshot' or state.get('completed_warmups') != 1
            or not request_report['request_IDs_exactly_once'] or request_report['exact_shape_successes'] != COUNT):
        return {**result, 'classification':'terminal_requires_review'}
    cfg = pipeline/'configs'/'sglang_warmup1_32768.py'
    config_hash = sha256(cfg.read_bytes()).hexdigest()
    inspect = json.loads(subprocess.check_output(['docker', 'inspect', container], timeout=8))[0]
    mount, = [row for row in inspect['Mounts'] if row['Destination'] == '/task/client_ds41_g1122/datasets']
    assert mount['RW'] is False
    dataset = Path(mount['Source'])/'warmup1_32768.jsonl'
    dataset_hash = sha256(dataset.read_bytes()).hexdigest()
    assert config_hash == command['config_sha256'] and dataset_hash == command['dataset_sha256']
    assert dataset_hash == manifest['warmup_dataset_sha256'][str(INPUT)]['warmup1']
    assert (run/'before_warmup1_32768_server_snapshot.json').is_file()
    assert (run/'after_warmup1_32768_server_snapshot.json').is_file()

    sys.path.insert(0, str(pipeline))
    from audit_aisbench import audit_phase

    audit = audit_phase(run/'warmup1_32768', INPUT, requests=COUNT, usage_contract=manifest['usage_contract'])
    assert audit['fully_validated'] and audit == load(run/'warmup1_32768_audit.json')
    assert raw == detail.read_bytes() and state == load(state_path)
    fields = {'duration_s','input_tokens_per_s','output_tokens_per_s','total_tokens_per_s',
              'ttft_mean_ms','ttft_p90_ms','ttft_p99_ms','tpot_mean_ms','tpot_p90_ms','tpot_p99_ms'}
    return {**result, 'classification':'terminal_verified', 'profile_epoch':manifest['gate']['profile_epoch'],
            'versioned_auditor_SHA256':sha256(auditor.read_bytes()).hexdigest(), 'dataset_SHA256':dataset_hash,
            'config_SHA256':config_hash,'metrics':{key:value for key,value in audit.items() if key in fields},
            'standard_raw_artifact_SHA256':audit['evidence']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--client',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(audit_warmup(args.client.resolve(strict=True))))
