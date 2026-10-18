#!/usr/bin/env python3
"""Profile P's real opt-in draft D2H; no profile data enters timing."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import uuid


def main():
    p=argparse.ArgumentParser();p.add_argument('--client',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--producer-only',action='store_true');a=p.parse_args()
    assert a.output.is_relative_to('/data/wio/glm5-3-vllm-execution/20261018-state-tail/seed-live1-20261018/node32')
    a.output.mkdir(exist_ok=False)
    spec=importlib.util.spec_from_file_location('seed_client',a.client);client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)
    def post(path,obj):
        q=urllib.request.Request('http://172.16.12.32:29481'+path,json.dumps(obj).encode(),{'Content-Type':'application/json'})
        with urllib.request.urlopen(q,timeout=300) as response:raw=response.read()
        return json.loads(raw) if raw else None
    report={'completed':False,'profile_role':'P','scope':'Real opt-in proposal plus native P sampling/target; outside all performance runs. Need trace copy correlation before claiming new D2H SDMA.'}
    started=False
    try:
        report['start_profile']=post('/start_profile',{});started=True
        # Native schedule delays2 iterations: one native P arm, then the
        # control P of run_pairs is arm2; its seed P is the active iteration.
        report['arm']=post('/v1/completions',{'model':'glm5','prompt':client.CASES[0]['prompt_token_ids'],'max_tokens':1,'min_tokens':1,'temperature':0,'ignore_eos':True,'cache_salt':'profile-arm-'+uuid.uuid4().hex})
        if a.producer_only:
            report['arm2']=post('/v1/completions',{'model':'glm5','prompt':client.CASES[0]['prompt_token_ids'],'max_tokens':1,'min_tokens':1,'temperature':0,'ignore_eos':True,'cache_salt':'profile-arm-'+uuid.uuid4().hex})
            report['export_request']={'model':'glm5','prompt':client.CASES[0]['prompt_token_ids'],'max_tokens':1,'min_tokens':1,'temperature':0,'ignore_eos':True,'return_token_ids':True,'cache_salt':'profile-export-'+uuid.uuid4().hex,'kv_transfer_params':{'_glm53_export_p_draft':True,'do_remote_decode':False,'do_remote_prefill':False}}
            report['export_response']=post('/v1/completions',report['export_request']);assert report['export_response']['choices'][0]['token_ids']==[20]
            report['scope']='Actual P opt-in draft D2H on producer with native local completion; no P remote KV ownership created, D is reloading. Not a full PD profile or timing run.'
            report['completed']=True
        else:
            cmd=[sys.executable,str(a.client),'--execute','--output',str(a.output/'pairs'),'--max-tokens','2','--order','control,seed','--case-count','1']
            result=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
            (a.output/'client-stdout.txt').write_text(result.stdout);(a.output/'client-stderr.txt').write_text(result.stderr);report['client_rc']=result.returncode;assert result.returncode==0,result.stderr
            report['completed']=True
    except Exception as exc:report['failure']=type(exc).__name__+': '+str(exc)
    finally:
        if started:
            try:report['stop_profile']=post('/stop_profile',{});report['profile_stopped']=True
            except Exception as exc:report['stop_failure']=str(exc)
    (a.output/'profile-request.json').write_text(json.dumps(report,indent=2)+'\n')
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END')
    assert report['completed'] and report.get('profile_stopped'),report


if __name__=='__main__':main()
