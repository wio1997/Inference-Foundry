#!/usr/bin/env python3
"""Read fixed profiler parser files as text; no device imports or mutation."""
import hashlib, json
from pathlib import Path
root=Path('/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis')
files={
 'msparser/compact_info/node_attr_info_parser.py':'1c82d89d715926e6f8f2f2bd9f11f758e3b866e206575e9af08a0fe1c39ed6ed',
 'msparser/compact_info/node_attr_info_bean.py':'4c2bf15419c93d484f9b2a4182320ab93cc544c9f82e52a1b95a996dfe2ed46c',
 'profiling_bean/db_dto/node_attr_info_dto.py':'96cc57fc708f39e32bfc500295984819ddc20719a6d44b50ac73db6b0f87e38b',
 'msmodel/compact_info/node_attr_info_model.py':'054f8c928303cdc4faddd56b142131740b5c4793444f49cbd72ca25790c249f5',
}
out=[]
for rel,expected in files.items():
 p=root/rel; raw=p.read_bytes()
 if len(raw)>200000: raise SystemExit('file too large: '+str(p))
 sha=hashlib.sha256(raw).hexdigest()
 if sha!=expected: raise SystemExit('source hash changed: '+str(p))
 out.append({'path':str(p),'sha256':sha,'text':raw.decode('utf-8')})
print(json.dumps({'scope':'four fixed CANN profiler NodeAttrInfo source files','files':out,'limits':['Parser structure does not itself prove QLI attrInfo encoding or exporter binary used historically.','No device libraries imported, no device work.']},ensure_ascii=False,indent=2))
