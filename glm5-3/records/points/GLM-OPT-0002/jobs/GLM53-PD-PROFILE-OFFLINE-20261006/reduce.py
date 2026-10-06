"""CPU-only audit of the completed profiling phase and original artifacts."""
import json,pathlib,subprocess,sys,hashlib,time
J=pathlib.Path(__file__).resolve().parent
job=json.loads(pathlib.Path(sys.argv[1]).read_text());R=pathlib.Path(job['run_directory'])
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
state=json.loads((R/'state.json').read_text());assert state['status']=='completed',state
phase=json.loads((R/'profile.phase.json').read_text());assert phase['exit_code']==0 and not phase['timed_out']
role_data={};evidence=[]
for h in ['166','167']:
 argv=['/usr/bin/python3',str(J/'reduce_trace.py'),str(R/('profiles_'+h))]
 if h=='167':argv=['ssh','-o','BatchMode=yes','root@172.16.10.167']+argv
 p=subprocess.run(argv,capture_output=True,timeout=400)
 (J/('reduce_'+h+'.stderr')).write_bytes(p.stderr)
 assert p.returncode==0,p.stderr[-2000:]
 data=json.loads(p.stdout);write(J/('device_'+h+'.json'),data);role_data[h]=data
 for name in ['device_'+h+'.json','reduce_'+h+'.stderr']:
  f=J/name;evidence.append(dict(id=name,path=str(f),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),locator='Original all-rank device tracks/event indices;CPU-only audit'))
clients=[]
for label in ['warmoff','measureoff','profileon']:
 events=json.loads((R/(label+'_events.json')).read_text());result=json.loads((R/(label+'_result.json')).read_text());tokens=[]
 for event in events:
  if event['value']:
   for c in event['value'].get('choices',[]):
    for tok in c.get('token_ids') or []:tokens.append(dict(token_id=tok,arrived_ns=event['arrived_ns']))
 assert len(tokens)==8,(label,len(tokens))
 clients.append(dict(label=label,result=result,token_ids=[t['token_id'] for t in tokens],token_chunks=[dict(arrived_ns=e['arrived_ns'],tokens=sum(len(c.get('token_ids') or []) for c in e['value'].get('choices',[]))) for e in events if e['value'] and any(c.get('token_ids') for c in e['value'].get('choices',[]))],first_to_last_token_TPOT_ms=(tokens[-1]['arrived_ns']-tokens[0]['arrived_ns'])/1e6/7))
summary=dict(run_id='GLM-RUN-0249',controller=state,phase_exit=phase['exit_code'],all32_nonempty_device_traces=all(d['all16_nonempty_device_traces'] for d in role_data.values()),clients=clients,formal_SLA_accepted=False,semantic_final_answer_accepted=False,Current=None,performance_gain=None,limits=['8-token diagnostic scope;client TPOT from token-ID arrival groups includes intra-MTP burst zero intervals;no SLA quantiles or stablecapacity','No crosshost calibrated criticalpath/removable-wait claim;all-rank trace is coverage,not causal readiness proof'])
write(J/'summary.json',summary)
f=J/'summary.json';evidence.append(dict(id=f.name,path=str(f),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),locator='Completed actual phase,all-rank coverage and actual SSE token-ID arrival reduction'))
write(pathlib.Path(job['result']['path']),dict(schema_version=1,job_id=job['job_id'],status='completed',summary='CPU-only all-rank trace/client reduction complete;actual coverage in summary,not a performance verdict.',execution=dict(inner_exit_code=0,acceptance='passed',processes=[]),findings=[dict(kind='fact',text='Actual completed profile phase reduced without new NPU work',scope=dict(run_id='GLM-RUN-0249',profiler=True),evidence_ids=['summary.json'])],evidence=evidence,unknowns=summary['limits'],decision_request=None,next_check_at=None))
