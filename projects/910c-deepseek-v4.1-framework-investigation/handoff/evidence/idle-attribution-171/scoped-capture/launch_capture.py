from pathlib import Path
import datetime,hashlib,json,subprocess
HERE=Path(__file__).resolve().parent
ssh='/Users/wio/.local/bin/baai-ssh'
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
remote='/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/idle-scopes-profile-'+stamp
script=(HERE/'collect_remote.py').read_bytes()
(HERE/'launch.json').write_text(json.dumps({'remote':remote,'utc_run_id':stamp,'script_sha256':hashlib.sha256(script).hexdigest()},indent=2))
source='''import pathlib,subprocess,sys,json,hashlib
p=pathlib.Path(sys.argv[1]);p.mkdir(exist_ok=False);s=sys.stdin.buffer.read();(p/'collect_remote.py').write_bytes(s)
f=open(p/'controller.log','wb');c=subprocess.Popen(['python3','-u',str(p/'collect_remote.py'),str(p)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'controller_pid':c.pid,'output':str(p),'sha256':hashlib.sha256(s).hexdigest()}))
'''
r=subprocess.run([ssh,'171','--','python3','-c',source,remote],input=script,capture_output=True,timeout=90)
(HERE/'launch-output.txt').write_bytes(r.stdout+r.stderr)
print(r.stdout.decode());r.check_returncode()
