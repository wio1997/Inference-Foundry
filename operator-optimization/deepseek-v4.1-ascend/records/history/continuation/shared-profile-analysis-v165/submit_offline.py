import json,sys
from pathlib import Path
sys.path.insert(0,'/Users/wio/work/171-vllm-migration-20261102')
from control import invoke
HERE=Path(__file__).resolve().parent
source=(HERE/'offline_worker.py').read_text()
code="SOURCE="+repr(source)+"\n"+r'''import json,subprocess,sys
from pathlib import Path
h=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/builtin-v164-official-db')
h.mkdir(parents=True,exist_ok=True)
if (h/'SUBMITTED.json').exists():
 print(json.dumps({'reused_existing_handle':True,**json.loads((h/'SUBMITTED.json').read_text())}));sys.exit(0)
p=h/'offline_worker.py';p.write_text(SOURCE)
log=(h/'controller.log').open('ab')
job=subprocess.Popen(['python3','-I','-S',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
handle={'pid':job.pid,'start_ticks':Path('/proc/'+str(job.pid)+'/stat').read_text().split()[21],'root':str(h)}
(h/'SUBMITTED.json').write_text(json.dumps(handle,indent=2)+'\n')
print(json.dumps(handle))
'''
d=invoke('target',code)
(HERE/'offline-submission.json').write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(d))
