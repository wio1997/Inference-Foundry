"""Mechanical CPU checks and handoff to the existing frozen unique controller."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main(path):
    job = json.loads(Path(path).read_text())
    dest = Path(job['result']['path']).parent
    run = Path(job['run_directory'])
    evidence = []
    result = dict(schema_version=1, job_id=job['job_id'], status='failed', summary='',
                  execution=dict(inner_exit_code=1, acceptance='failed', processes=[]),
                  findings=[], evidence=evidence, unknowns=[], decision_request=None, next_check_at=None)

    def save(name, raw, locator):
        p = dest / name
        with p.open('xb') as f:
            f.write(raw)
        evidence.append(dict(id=name, path=str(p), sha256=hashlib.sha256(raw).hexdigest(),
                             bytes=len(raw), locator=locator))

    try:
        for source in job['inputs']:
            assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256']
        start = subprocess.run(['/usr/bin/python3', str(run / 'runtime_bundle/controller.py'),
                                'start', str(run / 'controller_spec.json')], capture_output=True, timeout=20)
        save('controller_start.raw', start.stdout + start.stderr, 'real controller.start exit/output')
        assert start.returncode == 0, 'controller start exit%d' % start.returncode
        state = json.loads((run / 'state.json').read_text())
        owner = state['owner']
        parts = Path('/proc/%d/stat' % owner['pid']).read_text().rsplit(')', 1)[1].split()
        assert owner['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        assert owner['start_ticks'] == parts[19] and parts[0] != 'Z'
        assert state['status'] == 'running', state
        save('controller_state_snapshot.json', json.dumps(state, indent=2).encode() + b'\n',
             'actual running controller identity;readiness unknown')
        result.update(status='running', summary='Fresh unique controller owns Run252 one profiling-OFF CPU-clock attribution; no model reload/source edits.')
        result['execution'].update(inner_exit_code=start.returncode, acceptance='passed',
                                  processes=[dict(host='166', role='controller', pid=owner['pid'],
                                                  readiness='unknown', evidence_ids=['controller_state_snapshot.json'])])
        result['findings'] = [dict(kind='fact', text='Actual controller running;existing resident PD is accepted, this CPU-clock diagnostic is not yet complete.',
                                  scope=dict(run_id='GLM-RUN-0252', performance_gain=None),
                                  evidence_ids=['controller_start.raw', 'controller_state_snapshot.json'])]
        result['unknowns'] = ['CPU-clock attribution and before/after ownership await this controller phase;full API contract and SLA remain unaccepted.']
        result['next_check_at'] = 'Read actual state/readiness/final_status;never replay spec or infer ready from this handoff.'
        code = 0
    except Exception as error:
        result['summary'] = type(error).__name__ + ': ' + str(error)
        result['decision_request'] = 'Read actual controller state and preserved raw;do not replay this Job or blindly clean processes.'
        code = 1
    tmp = dest / 'result.tmp'
    tmp.write_text(json.dumps(result, indent=2) + '\n')
    os.replace(tmp, job['result']['path'])
    return code


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
