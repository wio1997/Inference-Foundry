#!/usr/bin/env python3
"""Snapshot one public primary reference over direct HTTP, with provenance."""
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

root=Path(__file__).resolve().parent/'references'
root.mkdir(exist_ok=True)
url='https://docs.pytorch.org/docs/2.10/tensor_view.html'
record={'url':url,'version':'PyTorch 2.10','retrieved_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'fact_kind':'DOCUMENTED','section':'Tensor Views / Note and special behaviors',
    'supports':['basic <REDACTED_CREDENTIAL> returns views; advanced indexing returns copies','reshape may be a view or copy; contiguous copies only when needed'],
    'mathematical_use':'alias/materialization contract; not a hardware traffic or performance lower bound',
    'runtime_and_backend_applicability':'framework API contract; actual NPU implementation/copy identity separately open'}
try:
    request=urllib.request.Request(url,headers={'User-Agent':'research-reference-snapshot/1.0'})
    with urllib.request.urlopen(request,timeout=25) as response:
        data=response.read(2_000_001)
        assert len(data)<2_000_000 and response.status==200
        record.update(status='SNAPSHOT_SAVED',http_status=response.status,final_url=response.url,
            last_modified=response.headers.get('Last-Modified'),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
    (root/'pytorch-2.10-tensor-view.html').write_bytes(data)
except Exception as error:
    record.update(status='HTTP_SNAPSHOT_FAILED',error=repr(error),sha256=None)
(root/'pytorch-2.10-tensor-view.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['url','status','sha256']},ensure_ascii=False))
