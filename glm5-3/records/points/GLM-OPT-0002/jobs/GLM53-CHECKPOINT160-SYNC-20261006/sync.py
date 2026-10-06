"""Locally committed checkpoint bundle fetch/FF and local Agent workspace verification through Zcode."""
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
    os.environ['GIT_TERMINAL_PROMPT'] = '0'
    git('fetch', str(J/'research.bundle'), 'glm5-3-autonomous-20261001')
    remote = git('rev-parse', 'FETCH_HEAD').decode().strip()
    assert remote == target, 'Bundle target mismatch'
    target = remote
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
    names = job['verify_files']
    files = []
    for name in names:
        raw = (R/name).read_bytes()
        assert raw == git('show', target+':'+name), f'Local file differs: {name}'
        files.append(dict(path=str(R/name), sha256=sha(raw), bytes=len(raw)))
    assert b'GLM-5.3' in (R/'glm5-3/AGENTS.md').read_bytes()
    assert b'GLM-5.3' in (R/'glm5-3/MISSION.md').read_bytes()
    result['evidence'].append(artifact('local_workspace.json', dict(
        host='166', repository=str(R), workspace=str(R/'glm5-3'),
        branch='glm5-3-autonomous-20261001', head=target,
        minimum_requested_head=job['target_commit'], files=files,
        local_files_match_git=True, service_operations=0, weight_operations=0)))
    result.update(status='completed', summary='166 local Agent workspace synchronized from locally committed checkpoint bundle; diagnostic/evidence/rule files verified against Git; dirty work preserved. GitHub publication pending explicit approval.')
    result['execution'].update(inner_exit_code=0, acceptance='passed')
    result['findings'] = [dict(kind='fact', text='Use /data/tiankuan/wio/Inference-Foundry/glm5-3 local files for rules, mission, plan and recovery; Local checkpoint is the current synchronization source; publication to the public GitHub repository is pending explicit approval.',
        scope=dict(hosts=['166'], configuration_only=True, weights_loaded=False,
                   service_operations=0, inference_requests=0), evidence_ids=['git_sync.json','local_workspace.json'])]
    result['unknowns'] = ['Model upload completion and GLM-5.3 loading/correctness/performance not probed by this synchronization Job.']
    exit_code = 0
except Exception as exc:
    result['summary'] = f'{type(exc).__name__}: {exc}'
    result['decision_request'] = 'Inspect preserved evidence; do not repeat, overwrite backups, reset/clean/force Git, or start services.'
finally:
    with Path(job['result']['path']).open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
sys.exit(exit_code)
