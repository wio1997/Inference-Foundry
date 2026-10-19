"""Fixed physical C4 diagnostic; stage four native P states before D cohort."""
from pathlib import Path
import concurrent.futures
import datetime
import json
import re
import shutil
import socket
import subprocess
import time
import urllib.request
import uuid

assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/mtp-moe-graph-live2-20261018/node33')
cid='084035e84399bd88c3788c80f184590f309fd6730f93cc308f460487c75af213'
plan=json.loads((root/'plan.json').read_text());d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['State']['Running'] and d['Id']==cid and d['Name']=='/'+plan['container']['name'] and d['Image']==plan['image']['id']
gate=json.loads((root/'mtp-moe-graph-c4-functional-v2/result.json').read_text());assert gate['completed'] and gate['numeric_all_pass']
assert not (root/'mtp-moe-graph-probe/disabled').exists()
marker=root/'mtp-moe-graph-probe/enable';assert not marker.exists()
assert shutil.disk_usage(root).free>24*1024**3
cases=json.loads((root/'prefill-padding-cases-v1.json').read_text())
out=root/'mtp-moe-graph-c4-profile-v1';out.mkdir(exist_ok=False)
report={'completed':False,'rows':[],'scope':'Physical diagnostic only:2serialnativeD8 arm, thennative4/candidate4 D8. Prepareall4Pstatesfirst, launchDcohorttogether. OriginalPremoteDecode/KV/target/K5; no profiletimeusedasservicegain.'}
(out/'cases.json').write_text(json.dumps(cases)+'\n')
before={p.name for p in (root/'profile').glob('*_ascend_pt')}
def post(node,path,body,rid=None,timeout=180):
    request=urllib.request.Request('http://172.16.12.%d:%d%s'%(node,29481 if node==32 else 29482,path),json.dumps(body).encode(),{'Content-Type':'application/json',**({'X-Request-Id':rid} if rid else {})})
    with urllib.request.urlopen(request,timeout=timeout) as f:b=f.read()
    return json.loads(b) if b else None
def idle():
    for attempt in range(121):
        values=[]
        for n,port in [(32,29481),(33,29482)]:
            with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(n,port),timeout=5) as f:s=f.read().decode()
            selected=[x for x in s.splitlines() if x.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))];assert len(selected)>=3;values+=selected
        if all(float(x.rsplit(' ',1)[1])==0 for x in values):return
        time.sleep(.25)
    raise AssertionError(values)
def prepare(case,mode,phase):
    rid='core-c4-profile-'+uuid.uuid4().hex
    body={'model':'glm5','prompt':case['prompt_token_ids'],'max_tokens':8,'min_tokens':1,'ignore_eos':True,'temperature':0,'top_p':1,'seed':1024,'return_token_ids':True,'cache_salt':rid,'stream':False}
    producer={**body,'max_tokens':1,'kv_transfer_params':{'do_remote_decode':True,'do_remote_prefill':False,'remote_engine_id':None,'remote_block_ids':None,'remote_host':None,'remote_port':None}}
    row={'case_id':case['case_id'],'mode':mode,'phase':phase,'profiled':phase!='preprofile-warmup','request_id':rid,'N':len(body['prompt']),'budget':8,'sent_monotonic_ns':time.monotonic_ns(),'sent_wall_ns':time.time_ns()}
    row['P_response']=post(32,'/v1/completions',producer,rid);params=row['P_response']['kv_transfer_params'];assert params['remote_num_prompt_tokens']==row['N']
    body['kv_transfer_params']=params;row['D_request']=body;report['rows'].append(row)
    return row,case
def consume(pair):
    row,case=pair;row['D_sent_monotonic_ns']=time.monotonic_ns();row['D_sent_wall_ns']=time.time_ns()
    result=post(33,'/v1/completions',row['D_request'],row['request_id']);row['D_response']=result;row['D_finished_monotonic_ns']=time.monotonic_ns();row['D_finished_wall_ns']=time.time_ns()
    match=re.search(r'[-+]?\d+(?:\.\d+)?',result['choices'][0]['text']);assert match and float(match.group())==case['expected']
    assert len(result['choices'][0]['token_ids'])==8 and result['usage']['completion_tokens']==8 and result['usage']['prompt_tokens']==row['N']
    row['complete_answer_pass']=True
attempted=False
try:
    idle();marker.touch()
    warmed=[prepare(case,'adaptive','preprofile-warmup') for case in cases['mixed']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(consume,warmed))
    idle();marker.unlink(missing_ok=True)
    attempted=True;report['start_profile']=post(33,'/start_profile',{},timeout=360)
    for case in cases['fixed']:
        idle();consume(prepare(case,'native','arm'))
    for mode in ['native','adaptive']:
        idle()
        if mode=='adaptive':marker.touch()
        else:marker.unlink(missing_ok=True)
        prepared=[prepare(case,mode,'C4') for case in cases['mixed']]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(consume,prepared))
        idle();(out/'progress.json').write_text(json.dumps(report)+'\n')
    report['completed']=True
except Exception as exc:report['failure']=type(exc).__name__+': '+str(exc)
finally:
    marker.unlink(missing_ok=True)
    if attempted:
        try:report['stop_profile']=post(33,'/stop_profile',{},timeout=360);report['profile_stopped']=True
        except Exception as exc:report['stop_failure']=str(exc)
dirs=sorted(p for p in (root/'profile').glob('*_ascend_pt') if p.name not in before)
report['new_profile_dirs']=[str(p) for p in dirs];report['ranks']=sorted(int(re.search(r'_rank(\d+)_',p.name)[1]) for p in dirs)
report['source_bytes']=sum(p.stat().st_size for directory in dirs for p in directory.rglob('*') if p.is_file())
(out/'result.json').write_text(json.dumps(report)+'\n')
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END')
assert report['completed'] and report.get('profile_stopped') and report['ranks']==list(range(16)) and report['source_bytes']<8*1024**3
