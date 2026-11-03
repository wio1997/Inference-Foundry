#!/usr/bin/env python3
"""Launch exactly one reviewed, owned-container diagnostic controller."""
raise SystemExit('Diagnostic complete; re-review required before launch')
from pathlib import Path
import json,os,subprocess,sys,time
p=Path(__file__).resolve().parent
handle=p/'live-controller-handle.json'
fd=os.open(handle,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
try:
 with open(p/'live-controller.stdout','xb') as out, open(p/'live-controller.stderr','xb') as err:
  proc=subprocess.Popen([sys.executable,'-u',str(p/'controller.py'),'--execute'],stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
  data=(json.dumps({'pid':proc.pid,'launched_at_unix':time.time(),'controller':str(p/'controller.py')})+'\n').encode()
  os.write(fd,data);os.fsync(fd)
 print(json.dumps({'pid':proc.pid,'handle':str(handle)}))
finally:os.close(fd)
