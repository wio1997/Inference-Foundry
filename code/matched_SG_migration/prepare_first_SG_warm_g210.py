from pathlib import Path
import hashlib,json
T=Path(__file__).resolve().parent
pair={role:json.loads((T/f'G208_G181_readiness_{port}/readiness.json').read_text()) for role,port in [('p',2201),('d',2202)]}
assert pair['p']['functional_completion']['passed']
for role,v in pair.items():
 a=v['API']['/server_info']['body'];a=a.get('server_args',a)
 assert a['tp_size']==a['ep_size']==8 and a['pp_size']==1 and a['context_length']==263296
 assert a['max_running_requests']==16 and a['speculative_num_steps']==3 and a['speculative_num_draft_tokens']==4
 assert not a['disable_overlap_schedule'] and not a['disable_radix_cache']
 assert v['native_dispatch_ledger_count']==16 and v['PD_registration_count']==8
 if role=='d':
  assert a['cuda_graph_config']['decode']['backend']=='full'
  assert len([l for l in v['graph_log_evidence'] if 'Capture target verify NPU graph end.' in l])==8
  assert any('Capture draft decode CUDA graph end.' in l for l in v['graph_log_evidence'])
  assert any('Capture draft extend CUDA graph end.' in l for l in v['graph_log_evidence'])
gate={'schema':1,'task_id':T.name,'engine':'sglang','ready':True,'pd_transfer_verified':True,'configuration_verified':True,'resources_verified':True,'max_model_len':263296,'model_id':'glm53flash','proxy_url':'http://<HOST>:18080','cold_mode':'novel_prefixes','warmup_dataset_phases':['warmup','warmup2','warmup3','warmup4','warmup5'],'formal_not_previously_submitted':True,'cold_prefix_audit':'client_vllm_g42/cold_prefix_audit.json','warmup_only':True,'warmup_pause_after':1,'P_compiler_cache_verified_after_warmups':False,'service_epoch':'sglang_G181_matched1_2','canonical_workload_guard':'node3:sg_matched_workload_guard_g177 pid1860169','roles':{},'scope':'Onlyfirstunscoredfull32KC16/1024 warmup; pauseafterfirstforPcache evidence; no formal claims or submissions; originalG182router503 retained andnewprefixG204completed'}
for role,v in pair.items():
 c=v['container'];url='http://<HOST>:19083' if role=='p' else 'http://<HOST>:19084'
 gate['roles']['prefill' if role=='p' else 'decode']={'ready':True,'container_id':c['Id'],'started_at':c['State']['StartedAt'],'base_url':url,'metrics_url':url+'/metrics','prefix_hit_counter':'sglang:cached_tokens_total'}
body='''from pathlib import Path
import datetime,hashlib,json,socket,subprocess,time,urllib.request
assert socket.gethostname()=='<ROLE_NODE>'
root=Path('<TASK_ROOT>');client=root/'client_vllm_g42';runroot=root/'benchmark_pipeline_sg_g199';d=root/'sg_pd_g181'
assert not (client/'SG_G181_warm_g210_handle.json').exists() and not (client/'formal_submissions_sglang.json').exists()
stage=json.loads((runroot/'stage_receipt.json').read_text());assert stage['datasets_reused_byte_identical'] and len(stage['profiles'])==24
state=json.loads((d/'state.json').read_text());assert state['container_id']==gate['roles']['decode']['container_id']
assert str(d/'node_controller.py').encode() in Path('/proc',str(state['pid']),'cmdline').read_bytes()
for device in range(8):
 own=json.loads((Path('<COORD_ROOT>')/f'<ROLE_NODE>-npu{device}.owner.json').read_text());assert own['state']=='claimed' and own['pid']==state['pid'] and own['run_directory']==str(d)
top=subprocess.check_output(['docker','top','yxy-glm53-aisbench-0002-g42','-eo','pid,args'],text=True);assert not any('ais_bench' in line for line in top.splitlines()[1:])
with urllib.request.urlopen('http://<HOST>:18080/workers',timeout=10) as q:workers=json.load(q)
assert {(w['worker_type'],w['url']) for w in workers['workers'] if w['is_healthy'] and w['load']==0}=={('prefill','http://<HOST>:19083'),('decode','http://<HOST>:19084')}
for role,g in gate['roles'].items():
 with urllib.request.urlopen(g['base_url']+'/health',timeout=30) as q:assert q.status==200
gate['validated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();path=root/'service_gate_sglang_g210_first_warm.json';path.write_text(json.dumps(gate,indent=2))
argv=['python3','-u',str(runroot/'run_aisbench_cold_hot.py'),'--root',str(root),'--engine','sglang','--container','yxy-glm53-aisbench-0002-g42','--image-id','sha256:05358d58d23f2679c71e48d93797d5d2a28d72eedc7fb4900479c89955465d76','--gate',str(path),'--usage-contract',str(root/'client_usage_contract_g45.json'),'--run-id','SG_G181_C16_32K_g210','--lengths','32768','--warmup-passes','5','--total-timeout','7200','--phase-timeout','5400']
q=subprocess.run(argv+['--preflight-only'],capture_output=True,text=True,timeout=90);(client/'SG_G181_preflight_g210.log').write_text(q.stdout+q.stderr);assert q.returncode==0,(q.stdout+q.stderr)[-2500:]
with (client/'SG_G181_warm_g210.log').open('x') as log:child=subprocess.Popen(['flock','--nonblock','<COORD_ROOT>/<ROLE_NODE>-client-g42.lock',*argv],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(2);h={'pid':child.pid,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':argv,'bound_s':7200,'scope':'Onlyfirstfull32KC16/1024unscoredwarmupthenpauseforPcache; warmup_onlytrue, noformalrequests','gate_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256((runroot/'run_aisbench_cold_hot.py').read_bytes()).hexdigest()};(client/'SG_G181_warm_g210_handle.json').write_text(json.dumps(h,indent=2));print('G210_SG_FIRST_WARM_START_JSON '+json.dumps({'handle':h,'actual_live':child.poll() is None}))
'''
(T/'start_first_SG_warm_g210.sh').write_text("python3 - <<'REMOTE_PY'\ngate="+repr(gate)+'\n'+body+'REMOTE_PY\n')
(T/'service_gate_sglang_g210_template.json').write_text(json.dumps(gate,indent=2))
print('G181 actualreadiness accepted; prepared bounded firstfullshape warmup only')
