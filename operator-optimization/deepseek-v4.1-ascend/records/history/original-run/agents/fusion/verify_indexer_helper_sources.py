#!/usr/bin/env python3
"""Check saved helper file hashes against the two-container provenance; CPU only."""
import hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parents[2]
provenance = json.loads((root / 'agents/fusion/indexer-helper-provenance.json').read_text())
assert len(provenance['containers']) == 2
assert provenance['containers'][0]['image_id'] == provenance['containers'][1]['image_id']
for c in provenance['containers']:
    assert c['workspace_source_overmount'] is False
assert len(provenance['source_files']) == 2
for record in provenance['source_files']:
    b=(root / record['snapshot_path']).read_bytes()
    assert len(b)==record['size'] and hashlib.sha256(b).hexdigest()==record['sha256']
    print(record['snapshot_path'],record['sha256'])
