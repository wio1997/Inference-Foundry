#!/usr/bin/env python3
"""Run bounded source-path discovery in the existing 169 stage container only."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent
QUERY=HERE/'metadata_discovery_inside.py'
OUT=HERE/'metadata-discovery.json'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
CONTAINER='wio-vllm-v41-repro169-stage-20261028'
IMAGE='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
ROOTS=('/vllm-workspace/vllm/vllm','/vllm-workspace/vllm-ascend/vllm_ascend')

def remote(argv):
    r=subprocess.run([SSH,shlex.join(argv)],capture_output=True,text=True,timeout=180)
    if r.returncode:
        raise RuntimeError('REMOTE_COMMAND_FAILED exit='+str(r.returncode))
    return r.stdout

if OUT.exists():
    print(json.dumps({'status':'STOP_OUTPUT_EXISTS'}));sys.exit(2)
try:
    parts=remote(['docker','inspect','--format','{{.Id}} {{.Image}} {{.State.Running}}',CONTAINER]).split()
    if len(parts)!=3 or parts[1]!=IMAGE or parts[2]!='true':
        raise RuntimeError('CONTAINER_IMAGE_OR_STATE_MISMATCH')
    mounts=json.loads(remote(['docker','inspect','--format','{{json .Mounts}}',CONTAINER]))
    covering=[]
    for m in mounts:
        dest=m.get('Destination','')
        if any(dest==root or root.startswith(dest+'/') or dest.startswith(root+'/') for root in ROOTS):
            covering.append({'destination':dest,'type':m.get('Type'),'rw':m.get('RW')})
    if covering:
        raise RuntimeError('SOURCE_ROOT_HAS_OVERLAPPING_MOUNT')
    output=remote(['docker','exec',CONTAINER,'python3','-c',QUERY.read_text()])
    if len(output.encode())>25000:
        raise RuntimeError('OUTPUT_LIMIT_EXCEEDED')
    result=json.loads(output)
    if any(result.get('root_status',{}).get(root)!='OK' for root in ROOTS):
        raise RuntimeError('SOURCE_ROOT_MISSING')
    result['stage_container_id']=parts[0]
    result['stage_image_id']=parts[1]
    result['source_roots_overlapping_mounts']=covering
    result['query_script_sha256']=hashlib.sha256(QUERY.read_bytes()).hexdigest()
    result['baseline_stage_target_file_identity']='NOT_CHECKED_BY_DISCOVERY'
    result['complete_search_proven']=(result['read_errors']==0 and result['files_too_large_skipped']==0 and result['omitted_candidates']==0)
    OUT.write_text(json.dumps(result,ensure_ascii=True,indent=2)+'\n')
    print(json.dumps({'status':'OK','output':str(OUT),'candidate_files_total':result['candidate_files_total'],
                      'omitted_candidates':result['omitted_candidates'],'complete_search_proven':result['complete_search_proven']}))
except Exception as exc:
    print(json.dumps({'status':'STOP','reason':type(exc).__name__+': '+str(exc)[:120]}));sys.exit(2)
