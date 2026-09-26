#!/usr/bin/env python3
"""Small, source-pinned retrieval and validation for performance priors.

Search reads Round TASK/REPORT/RESULT only; raw evidence is opened on demand.
No historical verdict is promoted to an Extreme verdict by this tool.
"""
import argparse
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / 'performance_knowledge'
SOURCE = json.loads((CATALOG / 'sources.json').read_text())['historical_dsv4f']
DEFAULT_REPO = Path(os.environ.get('PERFORMANCE_HISTORY_REPO', SOURCE['cache_path']))
REQUIRED = {'id', 'topic', 'mechanism', 'source', 'environment', 'observed',
            'failure_or_limit', 'revalidate_when', 'extreme_relation', 'status'}


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def check_repo(repo):
    if not (repo / '.git').exists():
        raise SystemExit(f'history checkout missing: {repo}; run sync or set PERFORMANCE_HISTORY_REPO')
    head = git(repo, 'rev-parse', 'HEAD')
    if head != SOURCE['commit']:
        raise SystemExit(f'history checkout {head} differs from pinned {SOURCE["commit"]}; run sync')
    return head


def sync(repo):
    repo.parent.mkdir(parents=True, exist_ok=True)
    if not (repo / '.git').exists():
        subprocess.run(['git', 'clone', '--depth', '1', '--filter=blob:none',
                        SOURCE['url'], str(repo)], check=True)
    if git(repo, 'rev-parse', 'HEAD') != SOURCE['commit']:
        subprocess.run(['git', '-C', str(repo), 'fetch', 'origin', SOURCE['commit']], check=True)
        subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', SOURCE['commit']], check=True)
    print(json.dumps({'repo': str(repo), 'commit': check_repo(repo)}, ensure_ascii=False))


def tokens(query):
    return [t.lower() for t in re.findall(r'[\w\u4e00-\u9fff]+', query) if len(t) > 1]


def search(repo, query, limit):
    check_repo(repo)
    terms = tokens(query)
    if not terms:
        raise SystemExit('query needs at least one searchable term')
    results = []
    for folder in sorted((repo / 'rounds').glob('R*')):
        parts = []
        for name in ('TASK.md', 'REPORT.md', 'RESULT.md'):
            path = folder / name
            if path.is_file():
                parts.append((path, path.read_text(errors='replace')))
        if not parts:
            continue
        haystack = '\n'.join(t for _, t in parts).lower()
        name = folder.name.lower()
        distinct = sum(t in haystack or t in name for t in terms)
        if not distinct:
            continue
        # Saturate long reports: a repeated common term is less useful than a
        # matching mechanism in the Round name or an explicit Round ID.
        hits = sum(min(haystack.count(t), 3) + 30 * (t in name) for t in terms)
        exact_round = any(re.fullmatch(r'r[0-9]+', t) and
                          name.startswith(t + '_') for t in terms)
        if exact_round:
            hits += 1000
        snippets = []
        for path, body in parts:
            lines = body.splitlines()
            matches = [(i, line.strip()) for i, line in enumerate(lines, 1)
                       if any(t in line.lower() for t in terms)]
            for number, line in matches[:3]:
                snippets.append({'path': str(path.relative_to(repo)), 'line': number,
                                 'text': line[:240]})
        results.append({'round': folder.name, 'score': hits + 10 * distinct,
                        'matched_terms': distinct, 'snippets': snippets[:6],
                        'report_url': f'{SOURCE["url"]}/blob/{SOURCE["commit"]}/rounds/{folder.name}/REPORT.md'})
    results.sort(key=lambda x: (-x['matched_terms'], -x['score'], x['round']))
    print(json.dumps({'query': query, 'source_commit': SOURCE['commit'],
                      'results': results[:limit]}, ensure_ascii=False, indent=2))


def validate():
    path = CATALOG / 'entries.jsonl'
    ids = set()
    count = 0
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        missing = REQUIRED - set(obj)
        if missing:
            raise SystemExit(f'entry line {number} missing {sorted(missing)}')
        if obj['id'] in ids:
            raise SystemExit(f'duplicate knowledge id {obj["id"]}')
        ids.add(obj['id'])
        src = obj['source']
        if not isinstance(src, list) or not src or any(not item.get('path') or not item.get('ref') for item in src):
            raise SystemExit(f'entry line {number} needs source path and ref')
        if obj['status'] not in ('prior', 'current_diagnostic', 'current_formal', 'rejected', 'conditional'):
            raise SystemExit(f'entry line {number} invalid status')
        count += 1
    print(json.dumps({'entries': count, 'valid': True}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('sync', help='materialize pinned historical checkout outside Foundry')
    s.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    s = sub.add_parser('search', help='find relevant historical Round summaries')
    s.add_argument('query')
    s.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    s.add_argument('--limit', type=int, default=5)
    sub.add_parser('validate', help='validate curated knowledge entries')
    a = p.parse_args()
    if a.command == 'sync':
        sync(a.repo)
    elif a.command == 'search':
        search(a.repo, a.query, a.limit)
    else:
        validate()


if __name__ == '__main__':
    main()
