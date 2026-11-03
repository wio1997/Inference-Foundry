"""Read the specific prepared/submitted handle; never submit or restart it."""
import sys,json
from finalize_origin_coverage_v9 import O,read
sys.path.insert(0,'/Users/wio/work/171-vllm-migration-20261102')
from control import invoke
prep=read(O/'shared-builtin-prepared-v164.json')
code='ROOT='+repr(prep['root'])+'\n'+r'''import json,os
from pathlib import Path
r=Path(ROOT);sub=r/'control/SUBMITTED.json';status=r/'STATUS.json'
out={'root':ROOT,'submitted':sub.exists(),'status_present':status.exists()}
if sub.exists():
 s=json.loads(sub.read_text());out['submission']=s;proc=Path('/proc',str(s['pid']))
 try:
  tick=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
  out['same_controller_handle_live']=tick==s['proc_start_ticks']
  out['controller_commandline']=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
 except FileNotFoundError:out['same_controller_handle_live']=False
if status.exists():out['status']=json.loads(status.read_text())
print(json.dumps(out))'''
print(json.dumps(invoke('target',code,20),ensure_ascii=False))
