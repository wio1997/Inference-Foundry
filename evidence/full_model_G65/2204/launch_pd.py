from pathlib import Path
import hashlib
import json
import os
import runpy
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
candidate=json.loads((ROOT/'candidate.json').read_text())
for key,value in candidate['environment'].items():os.environ[key]=value
os.chdir(ROOT)
with (ROOT/'server_help.txt').open('x') as f:
    p=subprocess.run([sys.executable,'-m','sglang.launch_server','--help'],stdout=f,stderr=subprocess.STDOUT,timeout=120)
if p.returncode:raise RuntimeError('Fixed-image native server help failed')
help_text=(ROOT/'server_help.txt').read_text()
for flag in candidate['argv']:
    if flag.startswith('--') and flag not in help_text:raise ValueError('Native CLI lacks '+flag)
from ds41_native_gate import EntryClass
manifest={'candidate':candidate,'candidate_sha256':hashlib.sha256((ROOT/'candidate.json').read_bytes()).hexdigest(),
          'native_source_sha256':json.loads((ROOT/'native_source_sha256.json').read_text()),
          'scope':'Native full TP8/EP8 DSPARK5 BF16KV candidate; full readiness/PD/graphs/performance unverified'}
(ROOT/'launch_manifest.json').write_text(json.dumps(manifest,indent=2))
sys.argv=candidate['argv']
runpy.run_module('sglang.launch_server',run_name='__main__')
