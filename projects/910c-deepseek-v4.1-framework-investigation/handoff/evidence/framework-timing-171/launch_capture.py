from pathlib import Path
import datetime,hashlib,json,subprocess
HERE=Path(__file__).resolve().parent
ssh='/Users/wio/.local/bin/baai-ssh'
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
remote='/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/function-profile-'+stamp
files={'collect_remote.py':(HERE/'collect_remote.py').read_text(),'adapter.json':(HERE/'adapter.json').read_text()}
for name in ['analyze_scopes.py','export_when_restored.py']:
 files[name]=(HERE/name).read_text()
for name in ['sitecustomize.py','ft_hook.py']:
 files['hook/'+name]=(HERE.parents[1]/'tools/framework-timing'/name).read_text()
script=json.dumps(files).encode()
(HERE/'launch.json').write_text(json.dumps({'remote':remote,'utc_run_id':stamp,'script_sha256':hashlib.sha256(script).hexdigest()},indent=2))
source='''import pathlib,subprocess,sys,json,hashlib
p=pathlib.Path(sys.argv[1]);p.mkdir(exist_ok=False);s=sys.stdin.buffer.read()
for name,content in json.loads(s).items():
 dest=p/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(content)
f=open(p/'controller.log','wb');c=subprocess.Popen(['python3','-u',str(p/'collect_remote.py'),str(p)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
e=subprocess.Popen(['python3','-u',str(p/'export_when_restored.py'),str(p)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'exporter_pid':e.pid,'controller_pid':c.pid,'output':str(p),'sha256':hashlib.sha256(s).hexdigest()}))
'''
r=subprocess.run([ssh,'171','--','python3','-c',source,remote],input=script,capture_output=True,timeout=90)
(HERE/'launch-output.txt').write_bytes(r.stdout+r.stderr)
print(r.stdout.decode());r.check_returncode()
