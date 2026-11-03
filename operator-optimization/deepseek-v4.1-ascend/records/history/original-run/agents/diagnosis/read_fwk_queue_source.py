#!/usr/bin/env python3
"""Read a single previously indexed, hash-bound profiler queue parser."""
import hashlib,json
from pathlib import Path
p=Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis/prof_parse/_fwk_file_parser.py');raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest()
if h!='5368902817cf6344706f5faad6511d4c06aa297054779bf409a361333b198430':raise RuntimeError('source changed')
lines=raw.decode().splitlines()
print(json.dumps({'path':str(p),'sha256':h,'line_count':len(lines),'sections':[{'start':1,'end':min(180,len(lines)),'text':'\n'.join(lines[:180])}],'limitations':['Installed parser version must match export producer; queue timestamp conversion may be in an imported bean or native module.','No NPU work.']},ensure_ascii=False,indent=2))
