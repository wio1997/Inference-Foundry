#!/usr/bin/env python3
"""Bounded six-node import of already copied artifacts; no model/service startup."""
from pathlib import Path
import json,subprocess,fcntl,os,time,concurrent.futures,shutil
Q=Path(__file__).resolve().parent;R=Q.parent;J=json.load(open(Q/'PINNED.json'));N=[34,35,36,41,46,47]
lock=os.open(Q/'.lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def save(phase,**kw):
 v={'phase':phase,'pid':os.getpid(),'epoch':time.time(),'models_started':False,'model_requests':0,**kw};t=Q/'PROGRESS.json.tmp';t.write_text(json.dumps(v,indent=2)+'\n');t.replace(Q/'PROGRESS.json');return v
def one(n):
 c=J['nodes'][str(n)];cmd=['kubectl','exec','-i','-n','default',c['pod'],'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-c',(Q/'node_import.py').read_text()]
 with (Q/(str(n)+'-stderr.log')).open('xb') as err:
  p=subprocess.run(cmd,input=json.dumps(J),stdout=subprocess.PIPE,stderr=err,text=True,timeout=2100)
 assert p.returncode==0,'Node'+str(n)+'failed; preserved stderr/progress'
 out=json.loads(p.stdout);assert out['node']==n and out['phase']=='ROLE_RESOURCES_READY_NO_MODEL_STARTED';(Q/(str(n)+'-receipt.json')).write_text(json.dumps(out,indent=2)+'\n');return out
try:
 old=json.load(open(R/'STATUS.json'));assert old['phase']=='PARTIAL_FAILURE_PRESERVED';assert not Path('/proc/53691').exists();(Q/'PREVIOUS_SOURCE_STATUS.json').write_text(json.dumps(old,indent=2)+'\n');save('IMPORTING_SIX_EXISTING_B103_ARCHIVES',nodes=N)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(one,N))
 allrows={}
 for n,c in J['nodes'].items():
  p=subprocess.run(['kubectl','exec','-n','default',c['pod'],'-c','hostctl','--','nsenter','-t','1','-m','--','cat',str(R/'STATUS.json')],capture_output=True,text=True,timeout=30);assert p.returncode==0,p.stderr[-500:];v=json.loads(p.stdout);assert v['phase']=='ROLE_RESOURCES_READY_NO_MODEL_STARTED' and not v['models_started'];allrows[n]=v
 result=save('ALL15_ROLE_RESOURCES_READY_NO_MODEL_STARTED',nodes=allrows,source_ready={'flash':True,'vllm':True,'sglang':True},repaired_nodes=N,archive_transfer_repeated=False,prior_failure_preserved=True)
 t=R/'STATUS.json.import-repair.tmp';t.write_text(json.dumps(result,indent=2)+'\n');t.replace(R/'STATUS.json');print(json.dumps({'phase':result['phase'],'nodes':15,'model_requests':0}))
except Exception as e:save('FAILED_IMPORT_REPAIR_PRESERVED',error=str(e));raise
