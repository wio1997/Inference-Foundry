from pathlib import Path
import hashlib,json,subprocess
P=Path(__file__).resolve().parent;capture=json.load(open(P.parent/'idle-attribution-171/scoped-capture/launch.json'))['remote']
source=(P/'extract_remote.py').read_text()
# Execution via stdin creates no remote files and imports only stdlib.
r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',source,capture],capture_output=True,timeout=240)
if r.returncode:
 (P/'read-failure.txt').write_bytes(r.stderr);print(r.stderr.decode()[-1200:]);raise SystemExit(r.returncode)
x=json.loads(r.stdout);(P/'measured-timeline.json').write_text(json.dumps(x,ensure_ascii=False,indent=2));print(json.dumps({'ranks':len(x['ranks']),'bytes':len(r.stdout),'trace_read_only':True,'new_npu_runs':0,'script_sha256':hashlib.sha256(source.encode()).hexdigest()}))
