#!/usr/bin/env python3
"""Execute one controller-frozen command once and reject input changes."""
import datetime,hashlib,json,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
spec_path=(ROOT/sys.argv[1]).resolve();assert spec_path.is_relative_to(ROOT)
spec=json.loads(spec_path.read_text());out=ROOT/spec['receipt_dir'];out.mkdir(parents=True,exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in spec['inputs'].items():assert sha(ROOT/n)==h,n
start=datetime.datetime.now(datetime.timezone.utc).isoformat();code=125
with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
 try:code=subprocess.run(spec['command'],cwd=ROOT,stdout=stdout,stderr=stderr,timeout=spec['timeout_seconds']).returncode
 except subprocess.TimeoutExpired:code=124
after={n:sha(ROOT/n) for n in spec['inputs']};unchanged=after==spec['inputs']
if not unchanged:code=126
v={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'spec_sha256':sha(spec_path),'command':spec['command'],'exit_code':code,'inputs_unchanged':unchanged,'input_hashes_after':after,'artifacts':{n:sha(out/n) for n in ['stdout.log','stderr.log']}}
(out/'receipt.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({'exit_code':code,'inputs_unchanged':unchanged}));sys.exit(code)
