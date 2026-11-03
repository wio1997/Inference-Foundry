#!/usr/bin/env python3
"""Stage-container CPU streaming extraction; local output only; no service actions."""
import base64,hashlib,json,shlex,subprocess,zlib
from pathlib import Path,PurePosixPath
H=Path(__file__).resolve().parent
SOURCE=H/'extract_target_layer_mc2.py'
HIER=H/'sidecar-hierarchy-20261029.json'
SCOPE=H/'scope-summary-v2.json'
SCOPE_SHA='3cf4a030e5e665a485bb9dccd77d9d8c77308bf425cc1d01cb4e16ba78d7ec6b'
OUT=H/'target-layer-mc2-16rank.json'
HIER_SHA='e4ffaf3cacaeebddb6beb249d9f48d76a3800dc5f02528cdc2ae46e6c3a8a023'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
STAGE='wio-vllm-v41-repro169-stage-20261028'
STAGE_ID='42a5ed03e3a3014d64e47af56a0304a3e9724573bf8df0c98adc47866c587602'
IMAGE='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
MOUNT_SRC='/data/tiankuan/wio/main';MOUNT_DST='/work'
ROOT=PurePosixPath('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime')

def call(cmd,timeout=60):
    x=subprocess.run([SSH,shlex.join(cmd)],capture_output=True,text=True,timeout=timeout)
    if x.returncode:raise RuntimeError('remote command failed: '+x.stderr[-1000:])
    return x.stdout

def inspect():
    data=call(['docker','inspect','--format','{{.Id}}|{{.Image}}|{{json .Mounts}}',STAGE]).strip()
    cid,img,raw=data.split('|',2)
    if cid!=STAGE_ID or img!=IMAGE:raise RuntimeError('stage identity changed')
    mounts=json.loads(raw) or []
    required=[m for m in mounts if m.get('Destination')==MOUNT_DST and m.get('Source')==MOUNT_SRC and m.get('Type')=='bind']
    if len(required)!=1:raise RuntimeError('stage work mount mismatch')
    for m in mounts:
        d=PurePosixPath(m['Destination'])
        if d==PurePosixPath(MOUNT_DST):continue
        if d==ROOT or d in ROOT.parents or ROOT in d.parents:raise RuntimeError('runtime source overmounted')
    return (cid,img,sorted(mounts,key=lambda m:json.dumps(m,sort_keys=True,separators=(',',':'))))

if OUT.exists():raise RuntimeError('output already exists')
raw=HIER.read_bytes()
if hashlib.sha256(raw).hexdigest()!=HIER_SHA:raise RuntimeError('hierarchy hash mismatch')
scope_raw=SCOPE.read_bytes()
if hashlib.sha256(scope_raw).hexdigest()!=SCOPE_SHA:raise RuntimeError('scope summary hash mismatch')
scope=json.loads(scope_raw)
if len(scope['ranks'])!=16 or any(x.get('trace_status')!='read' or not x.get('trace_sha256') for x in scope['ranks']):raise RuntimeError('scope summary incomplete')
trace_expected={x['rank_directory']:{'sha256':x['trace_sha256'],'bytes':x['trace_bytes']} for x in scope['ranks']}
if len(trace_expected)!=16:raise RuntimeError('scope ranks duplicated')
j=json.loads(raw);samples=j['layer_samples']
keys=('ep','pid','role','layer_index','call_id','parent_digest','unknown_reasons')
trim=[{k:item.get(k) for k in keys} for item in samples]
payload=base64.b64encode(zlib.compress(json.dumps({'samples':trim,'traces':trace_expected},separators=(',',':')).encode())).decode()
source=SOURCE.read_text();compile(source,str(SOURCE),'exec')
prefix="import base64,zlib,json\n_payload=json.loads(zlib.decompress(base64.b64decode("+repr(payload)+")))\nHIER_SAMPLES=_payload['samples']\nTRACE_EXPECTED=_payload['traces']\nHIER_SHA="+repr(HIER_SHA)+"\n"
before=inspect()
output=call(['docker','exec',STAGE,'nice','-n','15','ionice','-c','3','python3','-c',prefix+source],timeout=1800)
if inspect()!=before:raise RuntimeError('stage identity or mounts changed')
result=json.loads(output)
result['extractor_sha256']=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
result['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
result['hierarchy_input_sha256']=HIER_SHA
result['scope_summary_sha256']=SCOPE_SHA
result['stage_id']=STAGE_ID;result['image_id']=IMAGE
result['scope_note']='Current image/mount inspection does not prove historical execution binary or cross-clock timing.'
with OUT.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'output':str(OUT),'completed_ranks':len(result['ranks']),'rank_errors':result['rank_errors'],
                  'layer_scopes':sum(len(r['layers']) for r in result['ranks']),
                  'matched_tasks':sum(r['counts'].get('target_tasks_uniquely_matched',0) for r in result['ranks'])}))
