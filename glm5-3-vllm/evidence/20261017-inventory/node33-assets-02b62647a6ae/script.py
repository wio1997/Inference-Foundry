#!/usr/bin/env python3
"""Validate existing transfer receipts against known relative paths; no payload rehash."""
from collections import Counter
import hashlib
import json
from pathlib import Path


def main():
    root = Path('/data/wio/ascend-framework-comparison/assets/w8a8c8-node32-33-20261017')
    weights = Path('/data/wio/GLM53/GLM-5.3-w8a8c8')
    receipts = json.loads((root / 'COPIED_FILES.json').read_text())
    ready = json.loads((root / 'WEIGHTS_READY.json').read_text())
    # Exactly the canonical encoding used by the transfer/finalizer, not raw JSON bytes.
    identity = hashlib.sha256(json.dumps(receipts, sort_keys=True).encode()).hexdigest()
    files = {}
    for name, expected in receipts.items():
        p = weights / name
        assert p.resolve().is_relative_to(weights.resolve()) if hasattr(p, 'is_relative_to') else '..' not in Path(name).parts
        files[name] = {'exists': p.is_file()}
        if p.is_file():
            s = p.stat()
            files[name].update(bytes=s.st_size, mtime_ns=s.st_mtime_ns,
                               expected_bytes=expected['bytes'], payload_sha256_receipt=expected['sha256'])
    missing = [n for n, v in files.items() if not v['exists']]
    bad_size = [n for n, v in files.items() if v.get('bytes') != receipts[n]['bytes']]
    index = json.loads((weights / 'quant_model_weights.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    quant_path = weights / 'quant_model_description.json'
    quant = json.loads(quant_path.read_text())
    counter = Counter(v for v in quant.values() if isinstance(v, str))
    c8 = {k: v for k, v in quant.items() if any(x in k.lower() for x in ('kv_cache', 'k_cache', 'v_cache', 'indexer', 'attention'))}
    result = {'scope': 'receipt identity plus current file metadata; no independent payload rehash',
        'weights_path': str(weights), 'canonical_alias': str(Path('/data/wio/models/GLM-5.3-w8a8c8').resolve()),
        'receipt': ready, 'canonical_manifest_sha256': identity,
        'manifest_matches_receipt': identity == ready['manifest_sha256'],
        'file_count': len(files), 'total_bytes': sum(v.get('bytes', 0) for v in files.values()),
        'missing': missing, 'size_mismatch': bad_size, 'files': files,
        'index_shards': shards, 'index_shards_missing_from_receipts': sorted(set(shards)-set(receipts)),
        'index_tensor_count': len(index['weight_map']),
        'quant_description_sha256': hashlib.sha256(quant_path.read_bytes()).hexdigest(),
        'quant_string_counts': dict(counter), 'quant_attention_indexer_entries': c8,
        'transfer_root_entries': sorted(p.name for p in root.iterdir())}
    print(json.dumps(result, indent=2, sort_keys=True))
    if missing or bad_size or identity != ready['manifest_sha256']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
