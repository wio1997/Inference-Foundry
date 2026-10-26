#!/usr/bin/env python3
"""Compare deterministic same-prompt output content across original and owner paths."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    base = json.loads(a.baseline.read_text())
    test = json.loads(a.candidate.read_text())
    if base['summary']['success'] != 12 or test['summary']['success'] != 12:
        raise RuntimeError('one side lacks 12 complete client outputs')
    b = {r['i']: r for r in base['requests']}
    t = {r['i']: r for r in test['requests']}
    if set(b) != set(range(12)) or set(t) != set(range(12)):
        raise RuntimeError('different request inventory')
    mismatches = []
    for i in range(12):
        for key in ('input_tokens', 'output_tokens', 'content_sha256', 'reasoning_sha256'):
            if b[i][key] != t[i][key]:
                mismatches.append({'request': i, 'field': key,
                                   'baseline': b[i][key], 'candidate': t[i][key]})
    result = {'status': 'same_prompt_content_exact' if not mismatches else 'content_mismatch',
              'requests': 12, 'mismatch_count': len(mismatches),
              'mismatches': mismatches[:30],
              'scope': 'same prompts/seed temperature0, separate service runs; output text equality, not hidden-state proof'}
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'mismatch_count': len(mismatches)}))
    if mismatches:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
