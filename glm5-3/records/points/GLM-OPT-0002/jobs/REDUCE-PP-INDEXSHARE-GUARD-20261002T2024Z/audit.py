from pathlib import Path
import json,sys,hashlib,subprocess,shlex,ast,re,urllib.request
sys.path.insert(0,'/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime')
from phase_runner import atomic_json,utc
j=Path(__file__).parent;p=j.parents[1];old=p/'jobs/CPU-TP8PP2-DCP8-GEOMETRY-20261002T2016Z'
err=(old/'cpu.stderr').read_text();assert 'IndexShare group crosses a pipeline-parallel stage boundary: PP rank 1/2 owns layers [39, 78)'in err and 'create_engine_config'in err and not(old/'result.json').exists()
acks=[json.loads(l)for l in(old/'cpu.stdout').read_text().splitlines()if l.startswith('{"event":')];assert len(acks)==2 and [x['event']for x in acks]==['task_acl_init','task_acl_finalize']and all(x['returncode']==0 for x in acks)
other=p/'jobs/CPU-TP8PP2-PARTITION40-20261002T2020Z';assert 'owns layers [40, 78)'in(other/'cpu.stderr').read_text();assert not(other/'result.json').exists()
acks40=[json.loads(l)for l in(other/'cpu.stdout').read_text().splitlines()if l.startswith('{"event":')];assert len(acks40)==2 and all(x['returncode']==0 for x in acks40)
model_path=Path('/data/tiankuan/wio/GLM-5.2-w8a8/config.json');model=json.loads(model_path.read_text());assert model['indexer_types'][40]=='shared'and model['indexer_types'][42]=='full'
tree=ast.parse((p/'jobs/REDUCE-PROFILE105-20261002T1934Z/audit.py').read_text());check=next(ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='check'for t in n.targets));ast.parse(check)
owners=json.loads((p/'runs/GLM-RUN-0104/adopted_model_identities.json').read_text());members=json.loads((p/'runs/GLM-RUN-0104/native_member_identities.json').read_text());physical={}
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def metric(b):
 d={}
 for l in b.decode().splitlines():
  if l.startswith('vllm:'):
   name=l.split('{')[0].split()[0]
   if name.endswith('_total')or name in ['vllm:num_requests_running','vllm:num_requests_waiting','vllm:kv_cache_usage_perc']:d[name]=d.get(name,0)+float(l.rsplit(' ',1)[1])
 return d
for key,o in owners.items():
 ws=[x for x in members[key]['owned_targets']if x['pid']in members[key]['npu_worker_pids']];policy=dict(schema_version=1,cohort_id='GLM-COHORT-0089'if key=='D0'else'GLM-COHORT-0104',budget_tokens=4096,prefill_threshold_tokens=1024,prefill_cadence=1,serial=3 if key=='D0'else 1)
 args=['python3','-c',check]
 if o['host']=='167':args=['ssh','-o','BatchMode=yes','root@172.16.10.167',shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,workers=ws,policy=policy)).encode(),capture_output=True,timeout=90);(j/(key+'.stdout')).write_bytes(z.stdout);(j/(key+'.stderr')).write_bytes(z.stderr);z.check_returncode();physical[key]=json.loads(z.stdout)
 with http.open('http://172.16.10.'+o['host']+':'+str(o['port'])+'/metrics',timeout=10)as r:b=r.read();assert r.status==200
 (j/(key+'.metrics')).write_bytes(b);now=metric(b);before=metric((p/('runs/GLM-RUN-0106/after_0_'+key+'.metrics')).read_bytes());assert all(now[k]==v for k,v in before.items()if k.endswith('_total'));assert all(now[k]==0 for k in ['vllm:num_requests_running','vllm:num_requests_waiting','vllm:kv_cache_usage_perc'])
code="from pathlib import Path;import json,hashlib;f=Path('/vllm-workspace/vllm-ascend/vllm_ascend/platform.py');s=f.read_text().splitlines();print(json.dumps(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),lines='\\n'.join('%d:%s'%(i+1,s[i])for i in range(372,439)))))"
z=subprocess.run(['docker','exec','glm52-single','python3','-c',code],capture_output=True,timeout=30);z.check_returncode();source=json.loads(z.stdout)
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
out=dict(at=utc(),original_job_handoff='INVALID',original_job_result_missing=True,candidate_verdict='NOT_APPLICABLE',reason='ActualnativeCPU default39,39 PPpartition splits IndexShare dependency atlayer39; nativecorrectly rejects beforeconfigreturn/workers; conditionalNOT_APPLICABLE, no guardbypass',native_guard=source,SDK=acks,current_P89_D104_unchanged_idle=physical,native_vllm_total_counters_exactly_same_as106=True,models=0,requests=0,effective_output_tokens=0,original_cpu_stderr=ref(old/'cpu.stderr'),original_cpu_stdout=ref(old/'cpu.stdout'),original_source=ref(old/'cpu_probe.py'),limits=['PureCPtransfer geometry availability does not establish fullnativeconfig feasibility; oldprobe didnotreachgeometrymethod','Candidate rejected withincurrentnativeSFA class, not a globalhardware/impossibility proof; TP8PP2DCP8 separatelyunderstudy','NoGPUrequest/modelsignals/current/KEEP/capacity'])
out.update(partition40_author_index_error=dict(reason='GPT misread relative34:44slice asglobal40 full; actual40shared, nativecorrectrejection, originalsource/raw frozen',SDK=acks40,stdout=ref(other/'cpu.stdout'),stderr=ref(other/'cpu.stderr'),source=ref(other/'cpu_probe.py')),model_config=ref(model_path),full_indexer_layers=[i for i,v in enumerate(model['indexer_types'])if v.lower()=='full'],next_candidate='NativeVLLM_PP_LAYER_PARTITION42,36 aligns full42 withoutmodelcalculationchanges; separateactualCPUjob pending')
atomic_json(j/'reduction.json',out);b=(j/'reduction.json').read_bytes();atomic_json(j/'result.json',dict(schema_version=1,job_id=j.name,status='completed',summary='Readonly defaultPP39 and authored40 boundaryCPUjobINVALID/nativeIndexShare guard/SDK0/noinference/unchangedAPI2NPU32/counters/idle audited',execution=dict(inner_exit_code=0,acceptance='passed',processes=[]),findings=[],evidence=[dict(id='reduction',path=str(j/'reduction.json'),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator='exactnativeguard/rawCPUerror/SDK/ownership/counters')],unknowns=out['limits'],decision_request=None,next_check_at=None));print(json.dumps(dict(candidate='NOT_APPLICABLE',models=0,requests=0,SDK0=True)))
