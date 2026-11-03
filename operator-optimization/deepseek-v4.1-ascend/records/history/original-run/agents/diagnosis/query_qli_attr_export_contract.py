#!/usr/bin/env python3
"""Read-only keyword query on known installed CANN profiler analysis text sources."""
import hashlib, json, os, re
from pathlib import Path
root = Path('/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis')
patterns = re.compile(r'attrInfo|attr_info|attr_\{|attr_\%|attr_\d', re.I)
results, count, skipped, total = [], 0, [], 0
if not root.is_dir():
    raise SystemExit('missing fixed CANN profiler analysis root')
for base, dirs, files in os.walk(root, followlinks=False):
    dirs[:] = sorted(d for d in dirs if d not in ('__pycache__', '.git') and not (Path(base) / d).is_symlink())
    for name in sorted(files):
        if not name.endswith(('.py', '.sql')):
            continue
        p = Path(base) / name
        try:
            if p.is_symlink():
                skipped.append(str(p) + ' (symlink)')
                continue
            size = p.stat().st_size
            if size > 2_000_000:
                skipped.append(str(p) + ' (size limit)')
                continue
            if count >= 3000 or total + size > 80_000_000:
                raise RuntimeError('source bound exceeded')
            raw = p.read_bytes()
            total += len(raw)
            content = raw.decode('utf-8')
            count += 1
        except (OSError, UnicodeError):
            skipped.append(str(p) + ' (read/decode error)')
            continue
        hits = [{'line': i, 'text': line.strip()[:240]} for i, line in enumerate(content.splitlines(), 1) if patterns.search(line)]
        if hits:
            results.append({'path': str(p), 'sha256': hashlib.sha256(raw).hexdigest(), 'matches': hits[:20], 'total_matches': len(hits)})
if len(results) > 50:
    raise SystemExit('too many results; narrow query')
print(json.dumps({'root': str(root), 'files_checked': count, 'skipped': skipped, 'hits': results, 'limitations': ['Python/SQL analysis sources only; no native binaries or other paths.', 'Matching names alone do not prove attribute ordering or historical binary provenance.', 'No device libraries imported or device work.']}, ensure_ascii=False, indent=2))
