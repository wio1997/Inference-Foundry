"""Fast-forward only the research checkout; preserve all dirty work and services."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

J = Path(__file__).parent
job = json.loads(Path(sys.argv[1]).read_text())
R = Path('/data/tiankuan/wio/Inference-Foundry')
target = '5bbe8429212438378c326cd2caa4c1f23e971015'

def git(*args):
    return subprocess.check_output(['git', '-C', str(R), *args], timeout=60)

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

result = dict(schema_version=1, job_id=job['job_id'], status='failed', summary='',
              execution=dict(inner_exit_code=1, acceptance='failed', processes=[]),
              findings=[], evidence=[], unknowns=[], decision_request=None, next_check_at=None)
code = 1
try:
    assert git('branch', '--show-current').decode().strip() == 'glm5-3-autonomous-20261001'
    before = git('rev-parse', 'HEAD').decode().strip()
    dirty = set(filter(None, git('diff', '--name-only', '-z', 'HEAD').decode().split('\0')))
    untracked = set(filter(None, git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')))
    prior = {name: digest((R/name).read_bytes()) if (R/name).is_file() else None for name in dirty}
    staged = digest(git('diff', '--cached', '--binary'))
    worktree = digest(git('diff', '--binary'))
    bundle = J/'research.bundle'
    assert digest(bundle.read_bytes()) == job['inputs'][0]['sha256']
    git('fetch', str(bundle), 'glm5-3-autonomous-20261001')
    assert git('rev-parse', 'FETCH_HEAD').decode().strip() == target
    git('merge-base', '--is-ancestor', before, target)
    incoming = set(filter(None, git('diff', '--name-only', '-z', before, target).decode().split('\0')))
    assert not incoming.intersection(dirty), 'incoming change overlaps tracked dirty work'
    assert not incoming.intersection(untracked), 'incoming change overlaps untracked work'
    git('merge', '--ff-only', '--no-edit', target)
    assert git('rev-parse', 'HEAD').decode().strip() == target
    assert staged == digest(git('diff', '--cached', '--binary'))
    assert worktree == digest(git('diff', '--binary'))
    assert all((digest((R/n).read_bytes()) if (R/n).is_file() else None) == h for n, h in prior.items())
    assert untracked.issubset(set(filter(None, git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0'))))
    evidence = dict(before=before, after=target, dirty_sha256=prior,
                    tracked_diff_preserved=True, index_preserved=True,
                    prior_untracked_preserved=True, service_operations=0,
                    inference_requests=0, git_method='verified bundle; ff-only; no force')
    p = J/'sync_evidence.json'
    with p.open('x') as f:
        json.dump(evidence, f, indent=2); f.write('\n')
    raw = p.read_bytes()
    result.update(status='completed', summary='Strict PD scope and source review fast-forwarded; dirty work preserved; no service operations.')
    result['execution'].update(inner_exit_code=0, acceptance='passed')
    result['evidence'] = [dict(id='sync', path=str(p), sha256=digest(raw), bytes=len(raw), locator='before/after HEAD and dirty preservation')]
    result['findings'] = [dict(kind='fact', text='Research checkout now contains strict PD product rules.', scope='166 research Git checkout only', evidence_ids=['sync'])]
    code = 0
except Exception as exc:
    result['summary'] = f'{type(exc).__name__}: {exc}'
    result['decision_request'] = 'Inspect failure; do not reset, clean, overwrite or force Git.'
finally:
    with Path(job['result']['path']).open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
sys.exit(code)
