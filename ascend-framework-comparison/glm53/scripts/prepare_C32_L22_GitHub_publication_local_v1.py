"""Build a scoped source/config/aggregate-evidence publication manifest; no Git mutation."""
import ast
import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/20261019T-L22-github-publication-v1'
RESULTS = [
    'C32-L22-GitHub-publication-primary-review-v1.json',
    'C32-L22-terminal-primary-acceptance-v1.json', 'C32-L22-paired-trace-primary-analysis-v2.json',
    'model-request-ledger-through-L22-diagnostic-v1.json', 'model-request-ledger-vllm-L20-L21-L22-diagnostics-v1.json',
    'C32-L22-vllm-native-raw-capture-primary-acceptance-v1.json', 'C32-L22-vllm-offline-export-review-v1.json',
    'C32-L22-paired-diagnostic-actual-review-v1.json', 'C32-L22-native-evidence-collect-review-v3.json',
    'C32-L22-representative-admission-pure-review-v1.json', 'C32-L22-seventh-input-preparation-review-v1.json',
    'C32-L21-vllm-independent-diagnostic-review-v1.json', 'C32-L20-vllm-v2-independent-diagnostic-review-v1.json',
    'C32-L20-current-prefix-metadata-primary-acceptance-v2.json', 'C32-L18-existing-trace-primary-analysis-v1.json',
    'C32-L17-native-formal-decision-v1.json', 'C32-L17-formal-primary-independent-review-v1.json',
    'sglang-C32-L17-native-formal-measurement-v2.json', 'C32-L17-current-comparison-v1.json',
    'C32-L17-closeout-primary-acceptance-v1.json', 'C32-L12D-native-HiCache-acceptance-decision-v1.json',
    'sglang-C32-L12D-native-HiCache-v1-measurement-annotated-v2.json',
    'C32-L17-native-functional-decision-v1.json', 'C32-L17-post-functional-independent-native-review-v1.json',
    'C32-L15-timeline-source-decision-v1.json',
]
SECRET = re.compile(r'(?i)(huawei@\d+|BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,})')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    assert json.loads((ROOT / 'STATE.json').read_bytes())['status'] == 'PAUSED_BY_USER_AFTER_L22_COMPLETE'
    RUN.mkdir(exist_ok=False)
    index_path = ROOT / 'results/C32-L22-evidence-index-v1.json'
    assert not index_path.exists()
    indexed = [p for p in sorted((ROOT / 'results').glob('*.json')) if p != index_path]
    acceptance = json.loads((ROOT / 'results/C32-L22-terminal-primary-acceptance-v1.json').read_bytes())
    indexed.extend(ROOT / e['path'] for e in acceptance['evidence'])
    entries = {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in indexed}
    write(index_path, {'project_id': 'glm53', 'scope': 'Evidence content hashes, not upload of raw referenced files.',
        'raw_reports_and_traces_retained_locally': True, 'user_pause': True, 'entries': entries})
    selected = set()
    for folder, extensions in [('scripts', {'.py', '.sh'}), ('config', {'.py', '.json', '.conf', '.patch', '.md'}), ('diagnostics', {'.json', '.md'})]:
        for path in (ROOT / folder).rglob('*'):
            relative = str(path.relative_to(ROOT))
            if path.is_file() and path.suffix in extensions and 'provider' not in relative.lower():
                selected.add(relative)
    for name in ['STATE.json', 'STATUS_BRIEF.json', 'HANDOFF.md', 'PLAN.md', 'README.md', 'RECOVERY_LATEST.md', 'PERFORMANCE_CHECKPOINT.md', 'EXECUTION_CONTRACT.md']:
        selected.add(name)
    selected.update('results/' + name for name in RESULTS)
    selected.add(str(index_path.relative_to(ROOT)))
    files = {}
    for relative in sorted(selected):
        path = ROOT / relative
        assert path.resolve().is_relative_to(ROOT) and path.is_file() and not path.is_symlink()
        assert not relative.startswith('runs/') and 'provider' not in relative.lower()
        raw = path.read_bytes()
        assert len(raw) < 1536 * 1024, relative
        text = raw.decode('utf-8')
        assert not SECRET.search(text), ('Secret-shaped content blocked; path only', relative)
        if path.suffix == '.py':
            ast.parse(text)
        elif path.suffix == '.json':
            json.loads(text)
        files[relative] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    repo = ROOT.parents[1]
    prefix = 'ascend-framework-comparison/glm53/'
    staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z'], cwd=repo).decode().split('\0')
    assert not any(staged), 'Pre-existing staged files; do not mix.'
    changed = {p for p in subprocess.check_output(['git', 'diff', '--name-only', '-z'], cwd=repo).decode().split('\0') if p}
    assert changed <= {prefix + p for p in selected}, ('Changed files outside manifest', sorted(changed - {prefix + p for p in selected}))
    manifest = {'project_id': 'glm53', 'as_of': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'repository': 'wio1997/Inference-Foundry', 'preferred_branch': 'codex/ascend-sglang-vllm-comparison',
        'fallback_branch_if_remote_selected_path_collision': 'codex/glm53-l22-pause-20261019',
        'raw_capture_paths_excluded': True, 'provider_paths_excluded': True, 'force_push_allowed': False,
        'source_config_historical_drafts_are_not_current_launch_authority': True,
        'user_pause_preserved': True, 'files': files, 'file_count': len(files),
        'total_bytes': sum(v['bytes'] for v in files.values()), 'no_remote_model_or_profile_actions': True}
    write(RUN / 'publication-manifest.json', manifest)
    (RUN / 'pathspec.nul').write_bytes(b''.join((prefix + p).encode() + b'\0' for p in sorted(files)))
    print(json.dumps({'manifest_sha256': digest(RUN / 'publication-manifest.json'), 'files': len(files), 'total_bytes': manifest['total_bytes'], 'evidence_index_sha256': digest(index_path)}))


if __name__ == '__main__':
    main()
