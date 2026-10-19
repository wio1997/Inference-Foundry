#!/usr/bin/env python3
"""Summarize retained HCCL SDMA link evidence separately from timing results."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    backup = json.loads((a.run/'artifacts-backup.json').read_text())
    archive = Path(backup['local_archive'])
    h = hashlib.sha256()
    with archive.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    assert h.hexdigest() == backup['sha256']
    metrics = json.loads((a.run/'metrics.json').read_text())
    assert metrics['ok'], 'Failed payload test is not path correctness evidence'
    counts = dict(sdma_link_success=0, rdma_used_zero=0, rdma_used_one=0,
                  inter_rdma_one=0, force_rdma_one=0)
    tags, paths = set(), set()
    patterns = {'rdma_used_zero': r'isUsedRdma\[0\]', 'rdma_used_one': r'isUsedRdma\[1\]',
                'inter_rdma_one': r'isInterRdma\[1\]', 'force_rdma_one': r'forceRdma\[1\]'}
    with tarfile.open(archive, 'r:gz') as t:
        for member in t:
            if not member.isfile() or not member.name.endswith('.log'):
                continue
            for raw in t.extractfile(member):
                line = raw.decode(errors='replace')
                if 'GetSdmaLinksByRankAndTag' in line and 'success!' in line:
                    counts['sdma_link_success'] += 1
                    paths.add(member.name)
                    tag = re.search(r'newTag\[([^]]+)\]', line)
                    if tag:
                        tags.add(tag.group(1))
                for name, pattern in patterns.items():
                    if re.search(pattern, line):
                        counts[name] += 1
    ok = counts['sdma_link_success'] > 0 and counts['rdma_used_zero'] > 0 and not any(
        counts[k] for k in ('rdma_used_one', 'inter_rdma_one', 'force_rdma_one'))
    result = {'basis': 'Complete retained HCCL log archive plus successful payload checks',
        'archive_sha256': backup['sha256'], 'counts': counts,
        'runtime_sdma_link_evidence': ok, 'sdma_success_paths': sorted(paths),
        'sdma_success_tags': sorted(tags), 'rank_mapping': metrics['rank_mapping'],
        'actual_acl_mapping': [r.get('acl_mapping') for r in metrics['rank_results']],
        'scope': 'Only the tested rank group, messages, operators and backend; no PD inference',
        'timing_use': 'INFO-logging diagnosis; excluded from timing map',
        'inference_limit': 'HCCL runtime link evidence; no device-profiler packet accounting'}
    (a.run/'sdma-analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'run': a.run.name, 'runtime_sdma_link_evidence': ok, 'counts': counts}))
    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
