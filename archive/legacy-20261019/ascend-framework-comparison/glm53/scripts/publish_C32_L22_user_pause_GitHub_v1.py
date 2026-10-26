"""Publish exact reviewed GLM53 source/config/evidence manifest without force push."""
import datetime
import hashlib
import json
import os
import subprocess
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RUN = ROOT / 'runs/20261019T-L22-github-publication-v1'
PREFIX = 'ascend-framework-comparison/glm53/'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def names(raw):
    return {s for s in raw.decode().split('\0') if s}


def main():
    assert len(sys.argv) == 2
    manifest_path = RUN / 'publication-manifest.json'
    assert digest(manifest_path) == sys.argv[1]
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest['project_id'] == 'glm53' and manifest['user_pause_preserved']
    assert manifest['raw_capture_paths_excluded'] and manifest['provider_paths_excluded'] and not manifest['force_push_allowed']
    with (RUN / 'publication-once.lock').open('x') as f:
        f.write(str(os.getpid()))
    env = os.environ | {'GIT_TERMINAL_PROMPT': '0', 'GIT_EDITOR': 'true', 'GIT_SEQUENCE_EDITOR': 'true'}
    receipt = {'project_id': 'glm53', 'manifest_sha256': sys.argv[1], 'launcher_sha256': digest(Path(__file__)),
        'model_requests': 0, 'remote_resource_actions': 0, 'profile_actions': 0, 'force_push': False,
        'events': [], 'committed': False, 'pushed': False}
    sequence = 0
    def git(args, timeout=60, allow_failure=False):
        nonlocal sequence
        sequence += 1
        result = subprocess.run(['git', '-c', 'core.hooksPath=/dev/null', '-c', 'http.version=HTTP/1.1', *args], cwd=REPO, env=env, capture_output=True, timeout=timeout)
        (RUN / f'git-{sequence:02d}.stdout').write_bytes(result.stdout)
        (RUN / f'git-{sequence:02d}.stderr').write_bytes(result.stderr)
        receipt['events'].append({'sequence': sequence, 'git_args': args, 'rc': result.returncode,
            'stdout_sha256': hashlib.sha256(result.stdout).hexdigest(), 'stderr_sha256': hashlib.sha256(result.stderr).hexdigest()})
        write(RUN / 'publication-receipt.json', receipt)
        if not allow_failure:
            assert result.returncode == 0, ('Git command failed; inspect bounded log', sequence, args[:3], result.returncode)
        return result
    def verify_files():
        for relative, info in manifest['files'].items():
            path = ROOT / relative
            assert path.resolve().is_relative_to(ROOT) and not path.is_symlink() and not relative.startswith('runs/')
            assert path.stat().st_size == info['bytes'] and digest(path) == info['sha256'], relative
    def remote_tip(branch):
        git(['fetch', '--no-tags', 'origin', 'refs/heads/' + branch], 120)
        return git(['rev-parse', 'FETCH_HEAD']).stdout.decode().strip()
    def contains(ancestor, descendant):
        result = git(['merge-base', '--is-ancestor', ancestor, descendant], allow_failure=True)
        assert result.returncode in [0, 1]
        return result.returncode == 0
    try:
        assert git(['rev-parse', '--show-toplevel']).stdout.decode().strip() == str(REPO)
        branch = manifest['preferred_branch']
        assert git(['branch', '--show-current']).stdout.decode().strip() == branch
        url = urllib.parse.urlsplit(git(['remote', 'get-url', 'origin']).stdout.decode().strip())
        assert url.scheme == 'https' and url.hostname == 'github.com' and not url.username and not url.password and url.path.rstrip('/') == '/wio1997/Inference-Foundry.git'
        assert not names(git(['diff', '--cached', '--name-only', '-z']).stdout)
        selected = {PREFIX + p for p in manifest['files']}
        assert names(git(['diff', '--name-only', '-z']).stdout) <= selected
        assert json.loads((ROOT / 'STATE.json').read_bytes())['user_pause']['auto_resume_allowed'] is False
        verify_files()
        receipt['head_before'] = git(['rev-parse', 'HEAD']).stdout.decode().strip()
        git(['add', '--pathspec-from-file=' + str(RUN / 'pathspec.nul'), '--pathspec-file-nul'])
        staged = names(git(['diff', '--cached', '--name-only', '-z']).stdout)
        assert staged and staged <= selected and all(p.startswith(PREFIX) for p in staged)
        receipt['own_changed_paths'] = len(staged)
        # Historical patches contain intentional whitespace; report rather than alter accepted source.
        receipt['whitespace_check_rc'] = git(['diff', '--cached', '--check', '--', PREFIX, ':(exclude)*.patch'], allow_failure=True).returncode
        git(['commit', '-m', 'Summarize GLM53 C32 L22 diagnostics and pause experiments', '-m',
             'Record completed paired count/cache diagnostics and native profile evidence, preserving SGLang stop HTTP500. Separate representative graph and inter-graph spans without causal or performance promotion. Retain L17 formal baseline and independent request ledgers. Publish only GLM53 source/config/aggregate evidence and hash indices; exclude raw reports, traces, request text, provider files, credentials and binaries. Honor the human pause; no new candidate or model run.'], 120)
        receipt['committed'] = True
        receipt['checkpoint_commit'] = git(['rev-parse', 'HEAD']).stdout.decode().strip()
        tip = remote_tip(branch)
        if not contains(tip, 'HEAD'):
            common = git(['merge-base', 'HEAD', tip]).stdout.decode().strip()
            changed_remote = names(git(['diff', '--name-only', '-z', common, tip]).stdout)
            collision = sorted(changed_remote & selected)
            if collision:
                branch = manifest['fallback_branch_if_remote_selected_path_collision']
                assert not git(['ls-remote', '--heads', 'origin', 'refs/heads/' + branch]).stdout.strip()
                git(['switch', '-c', branch])
                receipt['remote_selected_path_collision_preserved'] = collision
            else:
                git(['merge', '--no-edit', tip], 120)
        verify_files()
        # Commit tree bytes must match the manifest before any publication metadata update.
        for relative, info in manifest['files'].items():
            blob = git(['show', 'HEAD:' + PREFIX + relative]).stdout
            assert hashlib.sha256(blob).hexdigest() == info['sha256'], relative
        receipt['publication_branch'] = branch
        receipt['checkpoint_published_head'] = git(['rev-parse', 'HEAD']).stdout.decode().strip()
        git(['push', 'origin', 'HEAD:refs/heads/' + branch], 120)
        receipt['pushed'] = True
        observed = remote_tip(branch)
        assert contains(receipt['checkpoint_published_head'], observed)
        receipt['checkpoint_remote_contains_verified'] = True
        public = {'project_id': 'glm53', 'published_checkpoint_commit': receipt['checkpoint_published_head'],
            'branch': branch, 'url': 'https://github.com/wio1997/Inference-Foundry/commit/' + receipt['checkpoint_published_head'],
            'remote_contains_checkpoint_verified': True, 'manifest_sha256': sys.argv[1],
            'manifest_file_count': manifest['file_count'], 'changed_paths': len(staged), 'source_bytes': manifest['total_bytes'],
            'raw_reports_traces_payloads_providers_credentials_binaries_excluded': True,
            'user_pause_preserved': True, 'new_model_requests_or_profiles': 0,
            'note': 'This receipt certifies the main checkpoint commit. Its subsequent metadata-only commit is not self-referential.'}
        proof_path = ROOT / 'results/C32-L22-GitHub-publication-v1.json'
        assert not proof_path.exists()
        write(proof_path, public)
        for name in ['STATE.json', 'STATUS_BRIEF.json']:
            value = json.loads((ROOT / name).read_bytes())
            assert value['status'] == 'PAUSED_BY_USER_AFTER_L22_COMPLETE'
            value['GitHub_publication_receipt_for_current_snapshot'] = {'path': str(proof_path.relative_to(ROOT)), 'sha256': digest(proof_path),
                'checkpoint_commit': receipt['checkpoint_published_head'], 'branch': branch}
            value['next_checkpoint'] = 'Remain paused until the user explicitly resumes; GitHub checkpoint publication verified.'
            write(ROOT / name, value)
        metadata_paths = [PREFIX + p for p in ['results/C32-L22-GitHub-publication-v1.json', 'STATE.json', 'STATUS_BRIEF.json']]
        assert names(git(['diff', '--name-only', '-z']).stdout) <= set(metadata_paths)
        git(['add', '--', *metadata_paths])
        assert names(git(['diff', '--cached', '--name-only', '-z']).stdout) == set(metadata_paths)
        git(['commit', '-m', 'Record verified GLM53 L22 GitHub checkpoint receipt'], 120)
        # A shared branch may advance with unrelated projects; merge only disjoint metadata changes.
        tip = remote_tip(branch)
        if not contains(tip, 'HEAD'):
            common = git(['merge-base', 'HEAD', tip]).stdout.decode().strip()
            assert not names(git(['diff', '--name-only', '-z', common, tip]).stdout) & set(metadata_paths)
            git(['merge', '--no-edit', tip], 120)
        git(['push', 'origin', 'HEAD:refs/heads/' + branch], 120)
        receipt['final_published_commit'] = git(['rev-parse', 'HEAD']).stdout.decode().strip()
        observed = remote_tip(branch)
        assert contains(receipt['final_published_commit'], observed)
        receipt['remote_contains_final_verified'] = True
        assert not names(git(['diff', '--name-only', '-z']).stdout) and not names(git(['diff', '--cached', '--name-only', '-z']).stdout)
        receipt['final_tracked_worktree_clean'] = True
        print(json.dumps({k: v for k, v in receipt.items() if k != 'events'}), flush=True)
    except BaseException as error:
        receipt['error'] = repr(error)
        raise
    finally:
        receipt['as_of'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(RUN / 'publication-receipt.json', receipt)


if __name__ == '__main__':
    main()
