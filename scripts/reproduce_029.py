#!/usr/bin/env python3
"""Reproduce document 029 from this archive only; do not emit request text.

Stdlib only.  Output is aggregate JSON on stdout; no file, service or device writes.
--reference accepts an optional independently generated local results JSON and checks
all input hashes and derived request durations in memory; it never outputs the input.
"""
import argparse
import hashlib
import json
import lzma
import math
import statistics
from pathlib import Path

from timing_witness import phase_witness

ROOT = Path(__file__).resolve().parents[1]
PAIRS = {
    (32768, 'cold'): ('g228_cold_32768', 'g47_cold_32768', 'sg_formal_cold', 'v_formal_cold'),
    (32768, 'hot'): ('g228_hot_32768', 'g47_hot_32768', 'sg_formal_hot', 'v_formal_hot'),
    (65536, 'cold'): ('g273_cold_65536', 'g48_cold_65536', 'sg64_formal_cold', 'v64_formal_cold'),
    (65536, 'hot'): ('g273_hot_65536', 'g48_hot_65536', 'sg64_formal_hot', 'v64_formal_hot'),
}
RANK_DIRS = {
    ('sg', 'p'): 'evidence/matched_SG_G181/profile_G240/CPU_p/review_g252',
    ('sg', 'd'): 'evidence/matched_SG_G181/profile_G240/CPU_d/review_g252',
    ('vllm', 'p'): 'evidence/G4_profile_G138/CPU_p/review',
    ('vllm', 'd'): 'evidence/G4_profile_G138/CPU_d/review',
}
P_KDA = {'sg': 'chunk_kda_fwd', 'vllm': 'aclnnChunkKdaFwd_KdaChunkForward_ChunkKdaFwd'}
INDEXER = '_glm5_next_lightning_indexer_score_kernel'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def source(path, hashes):
    path = Path(path)
    name = path.as_posix()
    raw = (ROOT / path).read_bytes()
    if name not in hashes or sha(raw) != hashes[name]:
        raise ValueError('source hash mismatch: ' + name)
    return raw


def rank_event(obj, framework, name):
    if framework == 'sg':
        rows = [x for x in obj['top50_hardware'] if x['name'] == name]
        return None if not rows else (rows[0]['count'], rows[0]['sum_us'])
    rows = [x for x in obj['analysis']['operators'] if x['key'] == name]
    return None if not rows else (rows[0]['events'], rows[0]['sum_duration_us'])


def summarize(values):
    return {'min': min(values), 'median': statistics.median(values), 'max': max(values)}


def compare_rows(sg, vllm):
    """Same id+input hash pairs; output contains no id, hashes, or input text."""
    a = {r['id']: r for r in sg}
    b = {r['id']: r for r in vllm}
    if len(a) != len(sg) or len(b) != len(vllm) or set(a) != set(b):
        raise ValueError('duplicate or differing request IDs')
    if any(a[i]['input_sha256'] != b[i]['input_sha256'] for i in a):
        raise ValueError('input hash mismatch for a request ID')
    fields = ('ttft_ms', 'post_first_ms', 'e2e_ms')
    result = {'matched': len(a)}
    for key in fields:
        differences = [a[i][key] - b[i][key] for i in sorted(a)]
        result[key + '_sg_minus_vllm'] = {
            'positive': sum(v > 0 for v in differences), **summarize(differences)}
    return result


def analyze(reference=None):
    sums = {}
    for line in (ROOT / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        if name in sums:
            raise ValueError('duplicate archive checksum')
        sums[name] = digest
    metrics = json.loads(source('records/metrics.json', sums))
    prov = {x['artifact']: x for x in json.loads(source('provenance/artifacts.json', sums))}
    if reference is not None:
        reference = json.loads(Path(reference).read_text())['request_witness']
    comparisons = {}
    for (size, phase), (sp, vp, sr, vr) in PAIRS.items():
        collected = []
        for target, refkey in ((sp, sr), (vp, vr)):
            matches = [x for x in metrics if x['phase'] == target and x['scored']]
            if len(matches) != 1:
                raise ValueError('ambiguous phase ' + target)
            record = matches[0]
            if (record['input_tokens'], record['output_tokens'], record['requests'],
                record['successful_requests'], record['configured_concurrency']) != (size, 1024, 32, 32, 16):
                raise ValueError('unexpected workload')
            manifest = json.loads(source(record['manifest'], sums))
            evidence = manifest['raw_evidence']
            raw = lzma.decompress(source(evidence['details'], sums))
            if sha(raw) != prov[evidence['details']]['uncompressed_sha256']:
                raise ValueError('uncompressed hash mismatch')
            for db in evidence['timing_databases']:
                source(db, sums)
            entries = [json.loads(line) for line in raw.splitlines() if line.strip()]
            witness, _ = phase_witness(ROOT, manifest)
            if len(entries) != 32 or len(witness) != 32:
                raise ValueError('unexpected request count')
            byid = {x['data_id']: x for x in witness}
            if len(byid) != 32:
                raise ValueError('duplicate witness data ID')
            rows = []
            for entry in entries:
                i = entry['data_id']
                if entry['id'] != i or not entry['success'] or (entry['input_tokens'], entry['output_tokens']) != (size, 1024):
                    raise ValueError('request identity, success or usage mismatch')
                w = byid[i]
                item = {'id': i, 'input_sha256': sha(entry['input'].encode()),
                        'ttft_ms': w['TTFT'], 'e2e_ms': w['E2EL'],
                        'post_first_ms': w['E2EL'] - w['TTFT']}
                rows.append(item)
            if reference is not None:
                old = {x['id']: x for x in reference[refkey]}
                if set(old) != {r['id'] for r in rows}:
                    raise ValueError('reference identity mismatch')
                for row in rows:
                    oldrow = old[row['id']]
                    if row['input_sha256'] != oldrow['input_sha256']:
                        raise ValueError('published input hash differs from original witness')
                    for k in ('ttft_ms', 'post_first_ms', 'e2e_ms'):
                        if not math.isclose(row[k], oldrow[k], abs_tol=1e-5):
                            raise ValueError('published duration differs from original witness: ' + k)
            collected.append(rows)
        comparisons[f'{size}_{phase}'] = compare_rows(*collected)
    input_hashes = []
    for name in ('evidence/G4_profile_G138/client_profile/assets/profile_32768.jsonl.xz',
                 'evidence/matched_SG_G181/profile_G240/control/dataset.jsonl.xz'):
        raw = lzma.decompress(source(name, sums))
        input_hashes.append((sha(raw), len(raw.splitlines())))
    sgmeta = json.loads(source('evidence/matched_SG_G181/profile_G240/control/assets_manifest.json', sums))
    vmeta = json.loads(source('evidence/G4_profile_G138/client_profile/assets/assets.json', sums))
    if input_hashes[0] != input_hashes[1] or input_hashes[0][1] != 16 or input_hashes[0][0] != sgmeta['dataset_sha256'] or input_hashes[0][0] != vmeta['profile_sha256']:
        raise ValueError('diagnostic dataset hashes or counts differ')
    events = {}
    for (framework, role), directory in RANK_DIRS.items():
        values = {'indexer': [], 'kda': []}
        for rank in range(8):
            name = f'{directory}/rank_{rank}.json'
            obj = json.loads(source(name, sums))
            if obj['rank'] != rank:
                raise ValueError('rank mismatch')
            for label, op in [('indexer', INDEXER), ('kda', P_KDA[framework])]:
                if role != 'p' and label == 'kda':
                    continue
                hit = rank_event(obj, framework, op)
                if hit is None:
                    raise ValueError('required event absent from compact: ' + label)
                count, sum_us = hit
                values[label].append({'count': count, 'sum_ms': sum_us / 1000,
                                      'mean_ms_per_event': sum_us / count / 1000})
        events[f'{framework}_{role}'] = {label: {
            'ranks': len(v), 'counts': sorted(set(x['count'] for x in v)),
            'sum_ms_across_rank': summarize([x['sum_ms'] for x in v]),
            'mean_ms_per_event_across_rank': summarize([x['mean_ms_per_event'] for x in v])}
            for label, v in values.items() if v}
    return {'scope': '32K diagnostic compact and separate 32K/64K first cold/hot; not matched device steps or causal attribution',
            'formal': comparisons, 'profile_32k_rank_events': events,
            'diagnostic_decompressed_sha256': input_hashes[0][0], 'diagnostic_rows': input_hashes[0][1],
            'reference_verified': reference is not None,
            'limitations': ['each framework/phase one trial; shared capture ranks are not repeats',
                'TTFT and post-first are client durations not device components',
                'events are aggregate and may overlap; shape/step not matched; missing compact entries not zero']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', help='optional separate source analysis JSON; verified in memory only')
    args = parser.parse_args()
    print(json.dumps(analyze(args.reference), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
