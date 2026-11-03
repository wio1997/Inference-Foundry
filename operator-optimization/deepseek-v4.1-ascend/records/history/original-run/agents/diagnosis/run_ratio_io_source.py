#!/usr/bin/env python3
"""Inspect fixed containers then run bounded source-only search; stdout summary only."""
import hashlib
import json
import shlex
import subprocess
from pathlib import Path, PurePosixPath

T=Path(__file__).resolve().parents[2]
SCRIPT=T/'agents/diagnosis/query_ratio_io_source.py'
OUT=T/'agents/diagnosis/ratio-io-source-excerpt.json'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
STAGE='wio-vllm-v41-repro169-stage-20261028'
DIAG='238c1fae630aff68ef3a7c77d4a4fdd26e55019b35cd2f8e2ae3a3c776d182f3'
STAGE_ID='42a5ed03e3a3014d64e47af56a0304a3e9724573bf8df0c98adc47866c587602'
IMAGE='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
INVENTORY=T/'agents/diagnosis/ratio-bean-source-excerpt.json'
PARSER=T/'agents/diagnosis/ratio-export-source-excerpt.json'
INVENTORY_SHA='157648763dd9cd52cffe883aaabf35759f3b745d9f1ac65fee0c6ec7fc3c488b'
PARSER_SHA='90d336a2aaf1904a92b15099d5bc84f08b489159d5fc36730f13b14c2a522195'
ROOTS=('/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis','/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis')

def remote(args):
    r=subprocess.run([SSH,shlex.join(args)],capture_output=True,text=True,timeout=120)
    if r.returncode: raise RuntimeError('container inspection or bounded scan failed')
    return r.stdout

def inspect(target,expected):
    raw=remote(['docker','inspect','--format','{{.Id}}|{{.Image}}|{{json .Mounts}}',target]).strip()
    cid,image,mounts=raw.split('|',2)
    if cid!=expected or image!=IMAGE: raise RuntimeError('container/image mismatch')
    mounts=json.loads(mounts) or []
    for mount in mounts:
        d=PurePosixPath(mount['Destination'])
        for root in map(PurePosixPath,ROOTS):
            if d==root or d in root.parents or root in d.parents:
                raise RuntimeError('profiler source root affected by mount')
    canonical=sorted(mounts,key=lambda m:json.dumps(m,sort_keys=True,separators=(',',':'),ensure_ascii=True))
    return (cid,image,canonical)

if OUT.exists(): raise RuntimeError('output already exists')
if hashlib.sha256(INVENTORY.read_bytes()).hexdigest()!=INVENTORY_SHA or hashlib.sha256(PARSER.read_bytes()).hexdigest()!=PARSER_SHA: raise RuntimeError('source anchor hash mismatch')
before_stage=inspect(STAGE,STAGE_ID)
before_diag=inspect(DIAG,DIAG)
x=json.loads(remote(['docker','exec',STAGE,'python3','-c',SCRIPT.read_text()]))
if inspect(STAGE,STAGE_ID)!=before_stage or inspect(DIAG,DIAG)!=before_diag:
    raise RuntimeError('container image or mounts changed during scan')
inventory=json.loads(INVENTORY.read_text())
parser=json.loads(PARSER.read_text())
if len(x['files'])!=2 or len(inventory['files'])!=2 or len(parser['files'])!=3: raise RuntimeError('source anchor mismatch')
x['bean_source_sha256']=INVENTORY_SHA
x['parser_source_sha256']=PARSER_SHA
x['script_sha256']=hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
x['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
x['fixed_image']=IMAGE
x['stage_id']=STAGE_ID
x['diagnostic_id']=DIAG
x['source_roots_not_overmounted_at_inspection']=True
x['limitations']=['These two source hashes were first observed during this read; inspections and same current image do not prove history file versions or loaded objects',
                  'Fixed line excerpts do not establish complete data flow or compiled exporter semantics']
OUT.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'OK','files':len(x['files']),'out':str(OUT)}))
