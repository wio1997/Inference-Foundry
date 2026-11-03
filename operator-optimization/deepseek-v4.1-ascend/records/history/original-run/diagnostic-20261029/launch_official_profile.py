"""Submit the reviewed detached 169 diagnostic. Do not run during preparation."""
import base64
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve().parent
SSH = '/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'

def main():
    if sys.argv[1:] != ['--execute']:
        raise SystemExit('Review first; to submit the 169 diagnostic, pass --execute.')
    run = 'official-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    controller = (HERE/'official_profile_controller.py').read_bytes()
    client = (HERE/'official_profile_client.py').read_bytes()
    compile(controller, 'official_profile_controller.py', 'exec')
    compile(client, 'official_profile_client.py', 'exec')
    encoded_controller = base64.b64encode(controller).decode('ascii')
    encoded_client = base64.b64encode(client).decode('ascii')
    code = 'import base64;exec(compile(base64.b64decode('+repr(encoded_controller)+'),"official_profile_controller.py","exec"))'
    bootstrap = ('import os,subprocess,json;env=dict(os.environ);'
                 'env.update(WIO_RUN_ID='+repr(run)+',WIO_CLIENT_B64='+repr(encoded_client)+');'
                 'p=subprocess.Popen(["python3","-c",'+repr(code)+'],env=env,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True);'
                 'print(json.dumps({"run_id":'+repr(run)+',"controller_pid":p.pid}))')
    command = shlex.join(['python3','-c',bootstrap])
    result = subprocess.run([SSH,command],text=True,capture_output=True,timeout=30)
    if result.returncode: raise RuntimeError('SSH submission failed: '+result.stderr[-2000:])
    receipt=json.loads(result.stdout)
    receipt.update(controller_sha256=hashlib.sha256(controller).hexdigest(),client_sha256=hashlib.sha256(client).hexdigest(),status_relative='diagnostic-20261029/'+run+'/controller-state.json',submission_only=True)
    (HERE/'official-launch-latest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__ == '__main__': main()
