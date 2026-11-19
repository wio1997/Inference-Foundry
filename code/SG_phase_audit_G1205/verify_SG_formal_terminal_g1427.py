"""Wait for the exact first 32K runner and verify its retained cold/replay outputs."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import select

RUN = 'sg_pd_g1317_aisbench_32k_g1324'
FORMAL_DATA_SHA = '2484713fafb09065e6426717d1f07fa26cd3e7de4d771e734c60ec2511b94ffe'


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    value = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def wait_exact(handle, deadline):
    pid = handle['outer_PID']
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    with os.fdopen(fd, 'rb') as event:
        poller = select.poll()
        poller.register(event, select.POLLIN)
        if poller.poll(0):
            return
        try:
            cmd = Path('/proc', str(pid), 'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        except FileNotFoundError:
            return
        if cmd != [part.encode() for part in handle['argv']]:
            if poller.poll(0):
                return
            raise ValueError('Active PID does not match the recorded formal runner')
        while True:
            remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
            if remaining <= 0:
                raise TimeoutError('The quality launch deadline elapsed while formal was active')
            if poller.poll(min(60000, max(1, int(remaining * 1000)))):
                return


def verify_outputs(client, expected, audit_root=None):
    run = audit_root or client / 'runs/sglang' / RUN
    state = read(run / 'state.json')
    audit_path = run / 'audit_results.json'
    assert state['phase'] == 'all_phases_terminal'
    assert state['run_id'] == RUN
    assert datetime.fromisoformat(state['at']) >= datetime.fromisoformat(expected['at'])
    audit = read(audit_path)
    assert audit['engine'] == 'sglang' and audit['run_id'] == RUN
    assert audit['all_required_raw_metrics_and_usage_validated']
    profile, = audit['profiles']
    assert profile['input_tokens'] == 32768 and profile['status'] == 'cold_hot_raw_metrics_validated'
    assert profile['dataset_sha256'] == FORMAL_DATA_SHA
    assert len(profile['unscored_warmup_passes']) == 5
    assert all(row['fully_validated'] and row['requests'] == row['successful_requests'] == 16
               for row in profile['unscored_warmup_passes'])
    cold = profile['cache_counter_evidence']['cold']['prefill']
    assert cold['verified'] and cold['delta'] == 0
    stage_sha = {}
    for phase, key in (('cold', 'cold'), ('hot', 'hot_replay')):
        report = profile[key]
        assert report['configured_concurrency'] == report['observed_max_concurrency'] == 16
        assert report['input_tokens'] == 32768 and report['output_tokens'] == 1024
        assert report['fully_validated'] and report['requests'] == report['successful_requests'] == 32
        folder = (run / f'{phase}_32768').resolve(strict=True)
        assert len(report['evidence']) == 3
        for rel, checksum in report['evidence'].items():
            path = (folder / rel).resolve(strict=True)
            assert path.is_relative_to(folder) and digest(path) == checksum
        stage_sha[phase] = report['evidence']
    return {'runner_PID': expected['outer_PID'], 'run_id': RUN, 'epoch': 1317,
            'accepted_formal_audit_SHA256': digest(audit_path),
            'formal_stage_raw_SHA256': stage_sha,
            'formal_state_SHA256': digest(run / 'state.json'),
            'accepted_cold_and_replay_requests': [32, 32],
            'prefill_cold_local_hit_delta': cold['delta']}



def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run-root', type=Path, required=True)
    cli.add_argument('--launch-handle', type=Path, required=True)
    cli.add_argument('--wait-until')
    args = cli.parse_args()
    run = args.run_root.resolve(strict=True)
    handle = read(args.launch_handle)
    assert run.name == RUN and handle['run_id'] == RUN and handle['step'] == 'formal'
    assert handle['stage'] == 'formal' and handle['formal_submissions_authorized_by_gate']
    if args.wait_until:
        wait_exact(handle, datetime.fromisoformat(args.wait_until))
    print(json.dumps(verify_outputs(run, handle, audit_root=run), indent=2))


if __name__ == '__main__':
    main()
