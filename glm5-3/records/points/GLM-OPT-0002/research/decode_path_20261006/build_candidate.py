"""Build the exact reviewed candidate from a pinned stock file; no live edit.

Usage: python build_candidate.py STOCK_FILE NEW_OUTPUT_FILE
Preserves the installed CRLF bytes. Refuses unknown source or existing output.
"""
import hashlib
import json
import sys
from pathlib import Path


def build(stock, output):
    root = Path(__file__).resolve().parent
    identity = json.loads((root/'patch_identity.json').read_text())
    raw = stock.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == identity['installed_original_sha256']
    assert stock.resolve() != output.resolve() and not output.exists()
    diff = (root/'moe_gather.patch').read_text().splitlines(keepends=True)
    assert sum(line.startswith('@@') for line in diff) == 1
    hunk = diff[next(i for i,line in enumerate(diff) if line.startswith('@@'))+1:]
    assert all(line[0] in ' +-' for line in hunk)
    before = ''.join(line[1:] for line in hunk if line[0] in ' -')
    after = ''.join(line[1:] for line in hunk if line[0] in ' +')
    text = raw.decode().replace('\r\n', '\n')
    assert text.count(before) == 1
    result = text.replace(before, after, 1).encode()
    if b'\r\n' in raw:
        result = result.replace(b'\n', b'\r\n')
    assert hashlib.sha256(result).hexdigest() == identity['patched_sha256']
    with output.open('xb') as f:
        f.write(result)
    return identity['patched_sha256']


if __name__ == '__main__':
    print(build(Path(sys.argv[1]), Path(sys.argv[2])))
