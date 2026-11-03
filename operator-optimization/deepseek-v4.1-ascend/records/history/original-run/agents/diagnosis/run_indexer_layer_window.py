#!/usr/bin/env python3
from pathlib import Path
import hashlib,subprocess
T=Path(__file__).resolve().parents[2]; source=T/'agents/diagnosis/summarize_indexer_layer_window.py'; out=T/'agents/diagnosis/indexer-layer-window-occupancy.json'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='bb717cfae2e9524d6b5afa8aad728f34528f096a4d186cd82a7195239915e638'
if out.exists() or out.is_symlink(): raise RuntimeError('output already exists; refusing overwrite')
r=subprocess.run(['python3',str(source)],capture_output=True,check=True,timeout=90)
with out.open('xb') as f:f.write(r.stdout)
print('RESULT_SHA256',hashlib.sha256(r.stdout).hexdigest())
