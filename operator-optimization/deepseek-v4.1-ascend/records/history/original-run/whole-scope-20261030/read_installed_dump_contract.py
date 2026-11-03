#!/usr/bin/env python3
"""Read installed dump helper source by metadata; no target module import."""
import base64
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import urllib.request

CODE=r'''
import datetime,hashlib,importlib.metadata,json,pathlib
d=importlib.metadata.distribution('torch_npu')
candidates=[str(x) for x in (d.files or []) if str(x).endswith('.py') and str(x).startswith(('torch_npu/utils/','torch_npu/npu/'))]
candidates=sorted(set(candidates)|{'torch_npu/__init__.py'})
assert len(candidates)<=160
out={'scope':'read installed Python sources for one exact public dump symbol; metadata only, no torch/torch_npu import',
 'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'version':d.version,'searched_file_count':len(candidates),'files':{}}
total=0
for relative in candidates:
 p=pathlib.Path(d.locate_file(relative))
 if not p.is_file():continue
 b=p.read_bytes();total+=len(b);assert len(b)<600000 and total<12000000
 if b'save_npugraph_tensor' not in b:continue
 out['files'][relative]={'installed_path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode()}
out['total_source_bytes_read']=total
print(json.dumps(out,indent=2))
'''
W=Path(__file__).resolve().parent
payload=base64.b64encode(CODE.encode()).decode()
command=shlex.join(['docker','exec','wio-vllm-v41-repro169-20261028','python3','-c',"import base64;exec(base64.b64decode('"+payload+"'))"])
with (W/'installed-dump-contract.json').open('x') as out,(W/'installed-dump-contract.stderr').open('w') as err:
    result=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',command],stdout=out,stderr=err,timeout=40,check=False)
if result.returncode:sys.exit(result.returncode)
data=json.loads((W/'installed-dump-contract.json').read_text())
for relative,item in data['files'].items():
    p=W/'installed-dump-source'/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(item['text'])
ref=W/'references';ref.mkdir(exist_ok=True)
url='https://raw.githubusercontent.com/Ascend/torchair/master/docs/zh/npugraph_ex/dfx/data_dump.md'
record={'url':url,'repository':'Ascend/torchair','version':'master at retrieval; content SHA fixes snapshot','retrieved_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'supports':'official npugraph_ex data-dump guide describes torch_npu.save_npugraph_tensor for graph-internal targeted values; not a no-overhead or installed-support proof'}
try:
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'research-reference-snapshot/1.0'}),timeout=25) as response:
        b=response.read(300001);assert response.status==200 and len(b)<300000
    (ref/'torchair-npugraph-ex-data-dump.md').write_bytes(b)
    record.update(status='SNAPSHOT_SAVED',sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
except Exception as error:record.update(status='HTTP_SNAPSHOT_FAILED',error=repr(error),sha256=None)
(ref/'torchair-npugraph-ex-data-dump.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'returncode':result.returncode,'torch_npu_version':data['version'],'searched_source_files':data['searched_file_count'],
    'matched_source_files':list(data['files']),'reference_snapshot':record['status'],'target_modules_imported':False,'npu_or_inference_work':False}))
