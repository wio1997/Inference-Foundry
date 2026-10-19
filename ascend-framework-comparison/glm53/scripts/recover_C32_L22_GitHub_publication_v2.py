"""Resume the already committed checkpoint after HTTP/1.1 fetch failure, using HTTP/2."""
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
OLD = ROOT / 'runs/20261019T-L22-github-publication-v1'
RUN = ROOT / 'runs/20261019T-L22-github-publication-recovery-v2'
PREFIX = 'ascend-framework-comparison/glm53/'
MANIFEST_SHA = '494f7962643d1595ee19bd143df28be0d06bef1f084cd845ea945d3e944c1991'
CHECKPOINT = 'c4ef2884d8333b8e5401f6804cf851b3af499d9a'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def names(raw):
    return {s for s in raw.decode().split('\0') if s}


def main():
    assert sha(OLD / 'publication-manifest.json') == MANIFEST_SHA
    manifest = json.loads((OLD / 'publication-manifest.json').read_bytes())
    previous = json.loads((OLD / 'publication-receipt.json').read_bytes())
    assert previous['committed'] and not previous['pushed'] and previous['checkpoint_commit'] == CHECKPOINT
    RUN.mkdir(exist_ok=False)
    env = os.environ | {'GIT_TERMINAL_PROMPT': '0', 'GIT_EDITOR': 'true', 'GIT_SEQUENCE_EDITOR': 'true'}
    receipt = {'project_id': 'glm53', 'recovery_launcher_sha256': sha(Path(__file__)), 'original_manifest_sha256': MANIFEST_SHA,
        'checkpoint_commit': CHECKPOINT, 'prior_failure': 'HTTP/1.1 git fetch empty reply; original launcher not replayed.',
        'transport': 'Git HTTPS HTTP/2; independent ls-remote succeeded before this review.',
        'model_requests': 0, 'profile_actions': 0, 'resource_actions': 0, 'force_push': False, 'events': [], 'pushed': False}
    sequence = 0
    def git(args, timeout=90, allow_failure=False):
        nonlocal sequence
        sequence += 1
        result = subprocess.run(['git', '-c', 'core.hooksPath=/dev/null', '-c', 'http.version=HTTP/2', '-c', 'http.lowSpeedLimit=1', '-c', 'http.lowSpeedTime=30', *args], cwd=REPO, env=env, capture_output=True, timeout=timeout)
        (RUN / f'git-{sequence:02d}.stdout').write_bytes(result.stdout)
        (RUN / f'git-{sequence:02d}.stderr').write_bytes(result.stderr)
        receipt['events'].append({'sequence': sequence, 'args': args, 'rc': result.returncode})
        write(RUN / 'receipt.json', receipt)
        if not allow_failure:
            assert result.returncode == 0, ('Git failed; inspect bounded log', sequence, result.returncode)
        return result
    def contains(a, b):
        r = git(['merge-base', '--is-ancestor', a, b], allow_failure=True)
        assert r.returncode in [0, 1]
        return r.returncode == 0
    def fetch(branch):
        git(['fetch', '--no-tags', 'origin', 'refs/heads/' + branch])
        return git(['rev-parse', 'FETCH_HEAD']).stdout.decode().strip()
    def verify_tree():
        entries = git(['ls-tree', '-r', '-z', 'HEAD', '--', PREFIX]).stdout.split(b'\0')
        blobs = {}
        for entry in entries:
            if entry:
                meta, name = entry.split(b'\t', 1)
                mode, kind, digest = meta.decode().split()
                assert kind == 'blob'
                blobs[name.decode()] = digest
        for path, info in manifest['files'].items():
            p = ROOT / path
            raw = p.read_bytes()
            assert len(raw) == info['bytes'] and hashlib.sha256(raw).hexdigest() == info['sha256'], path
            blob_sha = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
            assert blobs[PREFIX + path] == blob_sha, path
    try:
        assert git(['rev-parse', '--show-toplevel']).stdout.decode().strip() == str(REPO)
        assert git(['rev-parse', 'HEAD']).stdout.decode().strip() == CHECKPOINT
        branch = manifest['preferred_branch']
        assert git(['branch', '--show-current']).stdout.decode().strip() == branch
        assert not names(git(['diff', '--name-only', '-z']).stdout) and not names(git(['diff', '--cached', '--name-only', '-z']).stdout)
        assert json.loads((ROOT / 'STATE.json').read_bytes())['status'] == 'PAUSED_BY_USER_AFTER_L22_COMPLETE'
        verify_tree()
        tip = fetch(branch)
        selected = {PREFIX + p for p in manifest['files']}
        if not contains(tip, 'HEAD'):
            common = git(['merge-base', 'HEAD', tip]).stdout.decode().strip()
            collisions = names(git(['diff', '--name-only', '-z', common, tip]).stdout) & selected
            if collisions:
                branch = manifest['fallback_branch_if_remote_selected_path_collision']
                assert not git(['ls-remote', '--heads', 'origin', 'refs/heads/' + branch]).stdout.strip()
                git(['switch', '-c', branch])
                receipt['remote_colliding_paths_preserved'] = sorted(collisions)
            else:
                git(['merge', '--no-edit', tip])
        verify_tree()
        receipt['branch'] = branch
        receipt['checkpoint_published_head'] = git(['rev-parse', 'HEAD']).stdout.decode().strip()
        git(['push', 'origin', 'HEAD:refs/heads/' + branch])
        receipt['pushed'] = True
        observed = fetch(branch)
        assert contains(receipt['checkpoint_published_head'], observed)
        receipt['checkpoint_remote_contains_verified'] = True
        proof = {'project_id': 'glm53', 'published_checkpoint_commit': receipt['checkpoint_published_head'],
            'branch': branch, 'url': 'https://github.com/wio1997/Inference-Foundry/commit/' + receipt['checkpoint_published_head'],
            'remote_contains_checkpoint_verified': True, 'manifest_sha256': MANIFEST_SHA,
            'manifest_file_count': manifest['file_count'], 'source_bytes': manifest['total_bytes'],
            'HTTP1_fetch_failure_preserved': True, 'recovery_transport': 'HTTP/2',
            'recovery_source_sha256': sha(Path(__file__)), 'raw_payloads_providers_credentials_binaries_excluded': True,
            'user_pause_preserved': True, 'new_model_requests_or_profiles': 0,
            'note': 'Receipt certifies main checkpoint; the subsequent metadata-only commit also records this recovery source.'}
        proof_path = ROOT / 'results/C32-L22-GitHub-publication-v1.json'
        assert not proof_path.exists()
        write(proof_path, proof)
        for name in ['STATE.json', 'STATUS_BRIEF.json']:
            value = json.loads((ROOT / name).read_bytes())
            assert value['status'] == 'PAUSED_BY_USER_AFTER_L22_COMPLETE'
            value['GitHub_publication_receipt_for_current_snapshot'] = {'path': str(proof_path.relative_to(ROOT)), 'sha256': sha(proof_path),
                'checkpoint_commit': receipt['checkpoint_published_head'], 'branch': branch}
            value['next_checkpoint'] = 'Remain paused until the user explicitly resumes; GitHub checkpoint publication verified.'
            write(ROOT / name, value)
        metadata = [PREFIX + p for p in ['results/C32-L22-GitHub-publication-v1.json', 'STATE.json', 'STATUS_BRIEF.json', str(Path(__file__).relative_to(ROOT))]]
        assert names(git(['diff', '--name-only', '-z']).stdout) <= set(metadata)
        git(['add', '--', *metadata])
        assert names(git(['diff', '--cached', '--name-only', '-z']).stdout) == set(metadata)
        git(['commit', '-m', 'Record verified GLM53 L22 GitHub publication and HTTP2 recovery'])
        tip = fetch(branch)
        if not contains(tip, 'HEAD'):
            common = git(['merge-base', 'HEAD', tip]).stdout.decode().strip()
            assert not names(git(['diff', '--name-only', '-z', common, tip]).stdout) & set(metadata)
            git(['merge', '--no-edit', tip])
        git(['push', 'origin', 'HEAD:refs/heads/' + branch])
        receipt['final_published_commit'] = git(['rev-parse', 'HEAD']).stdout.decode().strip()
        observed = fetch(branch)
        assert contains(receipt['final_published_commit'], observed)
        receipt['remote_contains_final_verified'] = True
        assert not names(git(['diff', '--name-only', '-z']).stdout) and not names(git(['diff', '--cached', '--name-only', '-z']).stdout)
        receipt['tracked_worktree_clean'] = True
        print(json.dumps({k: v for k, v in receipt.items() if k not in ['events', 'remote_colliding_paths_preserved']}), flush=True)
    except BaseException as error:
        receipt['error'] = repr(error)
        raise
    finally:
        receipt['as_of'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(RUN / 'receipt.json', receipt)


if __name__ == '__main__':
    main()
