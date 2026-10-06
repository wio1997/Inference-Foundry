"""Verified Git FF plus bounded two-host config retargeting through Zcode."""
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

J = Path(__file__).parent
job = json.loads(Path(sys.argv[1]).read_text())
R = Path('/data/tiankuan/wio/Inference-Foundry')
target = job['target_commit']
pins = json.loads((J/'deploy_pins.json').read_text())
result = dict(schema_version=1, job_id=job['job_id'], status='failed', summary='',
              execution=dict(inner_exit_code=1, acceptance='failed', processes=[]),
              findings=[], evidence=[], unknowns=[], decision_request=None, next_check_at=None)

def sha(b):
    return hashlib.sha256(b).hexdigest()

def git(*args):
    return subprocess.check_output(['git', '-C', str(R), *args], timeout=60)

def artifact(name, value):
    p = J/name
    with p.open('x') as f:
        json.dump(value, f, indent=2); f.write('\n')
    raw = p.read_bytes()
    return dict(id=name, path=str(p), sha256=sha(raw), bytes=len(raw), locator='configuration operation evidence')

exit_code = 1
try:
    for item in job['inputs']:
        assert sha(Path(item['path']).read_bytes()) == item['sha256']
    assert git('branch', '--show-current').decode().strip() == 'glm5-3-autonomous-20261001'
    before = git('rev-parse', 'HEAD').decode().strip()
    dirty = set(filter(None, git('diff', '--name-only', '-z', 'HEAD').decode().split('\0')))
    tracked = {n: sha((R/n).read_bytes()) if (R/n).is_file() else None for n in dirty}
    worktree, index = sha(git('diff', '--binary')), sha(git('diff', '--cached', '--binary'))
    untracked = set(filter(None, git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')))
    git('fetch', str(J/'research.bundle'), 'glm5-3-autonomous-20261001')
    assert git('rev-parse', 'FETCH_HEAD').decode().strip() == target
    git('merge-base', '--is-ancestor', before, target)
    incoming = set(filter(None, git('diff', '--name-only', '-z', before, target).decode().split('\0')))
    assert not incoming.intersection(dirty), 'Incoming commit overlaps dirty work'
    tracked_paths = set(filter(None, git('ls-files', '-z').decode().split('\0')))
    # Git may overwrite ignored files during FF; verify and preserve these too.
    collisions = {n for n in incoming if n not in tracked_paths and (R/n).exists()}
    allowed = set(job['identical_untracked_adoption'])
    assert collisions.issubset(allowed), f'Unexpected untracked collision: {sorted(collisions)}'
    for name in sorted(collisions):
        blob = git('show', target+':'+name)
        assert (R/name).read_bytes() == blob, f'Untracked content differs: {name}'
    adopted = []
    for name in sorted(collisions):
        saved = J/'git_adoption'/name
        saved.parent.mkdir(parents=True, exist_ok=True)
        assert not saved.exists()
        os.replace(R/name, saved)
        adopted.append(dict(path=name, sha256=sha(saved.read_bytes()), backup=str(saved)))
    git('merge', '--ff-only', '--no-edit', target)
    assert git('rev-parse', 'HEAD').decode().strip() == target
    assert worktree == sha(git('diff', '--binary')) and index == sha(git('diff', '--cached', '--binary'))
    assert all((sha((R/n).read_bytes()) if (R/n).is_file() else None) == h for n, h in tracked.items())
    remaining = set(filter(None, git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')))
    assert (untracked-collisions).issubset(remaining)
    assert all((R/a['path']).read_bytes() == Path(a['backup']).read_bytes() for a in adopted)
    result['evidence'].append(artifact('git_sync.json', dict(before=before, after=target,
        dirty_sha256=tracked, dirty_preserved=True, index_preserved=True,
        identical_untracked_adoption=adopted, other_untracked_preserved=True,
        force_used=False, service_operations=0)))
    node_script = (J/'node_config.py').read_text()
    rows = []
    for host in ['166', '167']:
        cfg = dict(host=host, pins=pins[host], allowed_paths=job['deploy_paths'])
        if host == '166':
            cmd = ['/usr/bin/python3', '-c', node_script, json.dumps(cfg)]
        else:
            command = 'python3 -c '+shlex.quote(node_script)+' '+shlex.quote(json.dumps(cfg))
            cmd = ['ssh', '-o', 'BatchMode=yes', 'root@172.16.10.167', command]
        proc = subprocess.run(cmd, capture_output=True, timeout=90)
        with (J/(host+'.stdout')).open('xb') as f:
            f.write(proc.stdout)
        with (J/(host+'.stderr')).open('xb') as f:
            f.write(proc.stderr)
        proc.check_returncode()
        rows.append(json.loads(proc.stdout))
    result['evidence'].append(artifact('deploy_config_changes.json', dict(hosts=rows,
        weight_upload_touched=False, weight_payload_reads=0, service_operations=0,
        inference_requests=0, historical_Run_raw_changed=False)))
    result.update(status='completed', summary='GLM-5.3 target/defaults/docs synced; both hosts retargeted with backups; no weights loaded or service operations.')
    result['execution'].update(inner_exit_code=0, acceptance='passed')
    result['findings'] = [dict(kind='fact', text='New model path and alias are GLM-5.3-w8a8 / glm-53; dirty work and historical evidence preserved.',
        scope=dict(hosts=['166','167'], configuration_only=True, weights_loaded=False,
                   service_operations=0, inference_requests=0), evidence_ids=['git_sync.json','deploy_config_changes.json'])]
    result['unknowns'] = ['Upload completion, config/index/shard readiness and GLM-5.3 model compatibility/correctness/performance remain unverified.']
    exit_code = 0
except Exception as exc:
    result['summary'] = f'{type(exc).__name__}: {exc}'
    result['decision_request'] = 'Inspect preserved evidence; do not repeat, overwrite backups, reset/clean/force Git, or start services.'
finally:
    with Path(job['result']['path']).open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
sys.exit(exit_code)
