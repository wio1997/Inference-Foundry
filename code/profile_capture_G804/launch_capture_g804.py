from pathlib import Path
import datetime,hashlib,json,socket,subprocess,time
import urllib.request
assert socket.gethostname()=='<ROLE_HOSTNAME>'
root=Path('/PRIVATE_TASK_ROOT');r=root/'vllm_pd_g727';o=r/'warmup_g791';run=root/'client_ds41_g503/runs/vllm/native032_profile727_warm791'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
proof=json.loads((o/'P_proof_g800.json').read_text());parent=json.loads((o/'parent_all5_g805.json').read_text())
assert proof['audit']['passed'] and not proof['audit']['changes'] and parent['accepted_unscored_raw'] and parent['last2_stability_and_D_compile_passed']
assert proof['audit']['after_files']>0 and parent['profile_input_dataset_sha256']=='5cf5dbc8174409a627c8a3439f4b07e0ca99d53ac19c187ed4988eff354d0c00'
for name,digest in parent['original_run_files_SHA256'].items():assert sha(run/name)==digest,name
assert json.loads((run/'state.json').read_text())['phase']=='warmup_diagnostics_complete_pending_compile_audit'
h=json.loads((o/'actual_handle_warmup45.json').read_text());p=Path('/proc',str(h['pid']),'cmdline');assert not(p.exists() and b'run_aisbench_cold_hot.py' in p.read_bytes())
top=subprocess.check_output(['docker','top','yxy-ds41-aisbench-g503','-eo','pid,args'],text=True);assert not any('ais_bench' in l for l in top.splitlines()[1:])
gate=json.loads((o/'service_gate.json').read_text());assert gate['profile_epoch']==727 and gate['warmup_only'] and not gate['formal_not_previously_submitted']
gate.update(validated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),P_D_last_two_warmups_compile_verified=True,profile_warmup_parent_sha256=sha(o/'parent_all5_g805.json'),P_compiler_proof_sha256=sha(o/'P_proof_g800.json'),scope='Actualall5unscoredwarming/P+Dcompiler/stabilitycomplete. Thisgateauthorizesonlyfresh16diagnosticcapture; originalfirstformalneverresent.')
path=o/'profile_capture_gate_g804.json';assert not path.exists();path.write_text(json.dumps(gate,indent=2)+'\n')
inputs=root/'client_ds41_g503/profile_inputs_g772/manifest.json';assert json.loads(inputs.read_text())['dataset_sha256']==parent['profile_input_dataset_sha256']
frozen=json.loads(inputs.read_text());dataset=root/'client_ds41_g503/profile_inputs_g772/profile_32768.jsonl';assert sha(dataset)==frozen['dataset_sha256']
rows=[json.loads(line) for line in dataset.read_text().splitlines()];tokenize=[]
for row in rows:
 request=urllib.request.Request(gate['roles']['prefill']['base_url']+'/tokenize',data=json.dumps({'model':gate['model_id'],'prompt':row['question']}).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(request,timeout=30) as response:reply=json.load(response)
 assert reply['count']==32768;tokenize.append({'count':reply['count'],'question_sha256':hashlib.sha256(row['question'].encode()).hexdigest(),'token_ids_sha256':hashlib.sha256(json.dumps(reply['tokens']).encode()).hexdigest()})
assert len(tokenize)==16
for actual,expected in zip(tokenize,frozen['records']):assert actual['question_sha256']==expected['question_sha256'] and actual['token_ids_sha256']==expected['token_ids_sha256']
(o/'actual_tokenize_profile_inputs_g804.json').write_text(json.dumps({'model_generation_requests':False,'cache_warming_requests':False,'all16_exact32768_IDs_match_CPU':True,'records':tokenize},indent=2)+'\n')
driver=r/'capture_native_profile_g774.py';assert sha(driver)==parent['capture_driver_sha256']
output=root/'client_ds41_g503/profile_native727_g804';assert not output.exists()
argv=['python3','-u',str(driver),'--gate',str(path),'--inputs',str(inputs),'--output',str(output),'--prefix','ds41-native727-g804','--workload-timeout','300']
with (r/'capture_g804.log').open('x') as stream:p=subprocess.Popen(argv,cwd=root,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(2);handle={'pid':p.pid,'argv':argv,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'inner_CPU_workload_bound_s':300,'profile_start_HTTP_bound_s':120,'profile_stop_export_HTTP_bound_s':900,'scope':'Firstfresh16/C16/32768/1024diagnosticcapture; noformalresubmit. Inspectexistinghandles/exportsontimeout; neverautoretry.'};(r/'capture_actual_handle_g804.json').write_text(json.dumps(handle,indent=2)+'\n');print('G804_NATIVE_PROFILE_CAPTURE_START_JSON '+json.dumps({'handle':handle,'actual_live':p.poll() is None,'immediate_returncode':p.poll(),'state':json.loads((output/'state.json').read_text()) if (output/'state.json').exists() else None}),flush=True)
