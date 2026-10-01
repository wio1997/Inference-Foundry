"""Detached read-only task monitor; no inference, service control or replay."""
import hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,'/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime')
from phase_runner import atomic_json,process_identity,same_process,utc

def monitor(job_path):
    job=json.loads(Path(job_path).read_text());root=Path(job['result']['path']).parent;run=Path(job['inputs'][0]['path']);identity={'host':socket.gethostname(),**process_identity(os.getpid())}
    snapshot=root/'monitor_snapshot.json'
    while True:
        state=json.loads((run/'state.json').read_text());active=state['status']=='running' and same_process(state.get('owner'));terminal=state['status'] in ('completed','failed','cancelled','needs_reconciliation')
        command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','root@172.16.10.167','tail -n 4 /data/tiankuan/wio/glm52-pd/deploy/logs/DP_run38_2.log /data/tiankuan/wio/glm52-pd/deploy/logs/DP_run38_3.log']
        sample=subprocess.run(command,capture_output=True,text=True,timeout=10)
        p_tail=subprocess.run(['tail','-n','4','/data/tiankuan/wio/glm52-pd/deploy/logs/DP_run38_0.log','/data/tiankuan/wio/glm52-pd/deploy/logs/DP_run38_1.log'],capture_output=True,text=True)
        observed={'Node0_tail':p_tail.stdout,'at':utc(),'monitor':identity,'controller':state,'controller_process_alive':active,'Node1_tail':sample.stdout,'tail_exit_codes':{'Node0':p_tail.returncode,'Node1':sample.returncode},'tail_stderr':{'Node0':p_tail.stderr,'Node1':sample.stderr},'kind':'read_only_native_four_DP_cohort_log_reduction'}
        final=run/'formal_summary.json'
        if final.exists():observed['formal_summary']=json.loads(final.read_text())
        atomic_json(snapshot,observed)
        with (root/'monitor_samples.jsonl').open('a') as out:out.write(json.dumps(observed)+'\n')
        status='completed' if terminal and state['status']=='completed' else 'needs_decision' if terminal or not active else 'running'
        evidence={'id':'snapshot','path':str(snapshot),'sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),'bytes':snapshot.stat().st_size,'locator':'controller state/identity and physical Node0/Node1 log tails at last heartbeat'}
        result={'schema_version':1,'job_id':job['job_id'],'status':status,'summary':'Run controller '+state['status']+', phase '+str(state['active_stage'])+'; inspect monitor snapshot; no SLA promotion','execution':{'inner_exit_code':0 if terminal else None,'acceptance':'passed' if status=='completed' else 'unverified','processes':[{'host':identity['host'],'role':'read-only monitor','pid':os.getpid(),'readiness':'ready','evidence_ids':['snapshot']}] if status=='running' else []},'findings':[],'evidence':[evidence],'unknowns':['Finite diagnostic/formal window cannot certify stable tail or service capacity'],'decision_request':'Controller terminal failure or owner disappeared; reconcile exact phase without replay' if status=='needs_decision' else None,'next_check_at':None}
        atomic_json(job['result']['path'],result)
        if status!='running':return 0
        time.sleep(45)

def launch(job_path):
    root=Path(job_path).parent
    with (root/'monitor.log').open('ab') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'work',job_path],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(200):
        if (root/'result.json').exists():print(json.dumps({'monitor_pid':child.pid,'result':str(root/'result.json')}));return 0
        if child.poll() is not None:return child.returncode
        time.sleep(.05)
    return 85
if __name__=='__main__':raise SystemExit({'launch':launch,'work':monitor}[sys.argv[1]](sys.argv[2]))
