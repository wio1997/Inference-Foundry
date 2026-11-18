"""Build a once-only formal resume only from parent-reviewed final P/D receipts."""
from pathlib import Path
import argparse,base64,json

p=argparse.ArgumentParser();p.add_argument('--d-proof',type=Path,required=True);p.add_argument('--p-proof',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('start_first256_corrected_formal_g514.sh'));a=p.parse_args()
d=json.loads(a.d_proof.read_text());prefill=json.loads(a.p_proof.read_text())
assert d['state']['phase']=='warmup_diagnostics_complete_pending_compile_audit'
assert d['current_D_proof']['full_warm4_and5_active_D_cache_equal']
assert len(prefill['files'])==636 and prefill['actual_P_controller_pid']==936231
assert prefill['container_id']=='0e8232575c7a4d9ad4065eda5d9682b9096876f030ddb1602c8c42dfdd30f5dc'
for item in d['audits'].values():
    assert item['fully_validated'] and item['requests']==item['successful_requests']==16
    assert item['input_tokens']==262144 and item['output_tokens']==1024
assert len(d['audits'])==5
assert all(d['current_D_proof']['timing_relative_changes'][k]<=.05 for k in ['duration_s','ttft_mean_ms'])
payload=base64.b64encode(json.dumps({'D':d['current_D_proof'],'D_proof_sha256':d['current_D_proof_sha256'],'P':prefill}).encode()).decode()
body=r"""from pathlib import Path
import base64,datetime,hashlib,json,socket,subprocess,time
assert socket.gethostname()=='<ROLE_NODE>'
parent=json.loads(base64.b64decode(PAYLOAD))
root=Path('<TASK_ROOT>');c=root/'client_vllm_g42';run=c/'runs/sglang/SG_G316_C16_256K_g404'
old=json.loads((c/'SG_G316_256K_resume_g445_handle.json').read_text());proc=Path('/proc',str(old['pid']),'cmdline');assert not proc.exists() or b'run_aisbench_cold_hot.py' not in proc.read_bytes()
assert json.loads((run/'state.json').read_text())['phase']=='warmup_diagnostics_complete_pending_compile_audit'
assert not list(run.glob('cold_*')) and not list(run.glob('hot_*'))
manifest=json.loads((run/'run_manifest.json').read_text());assert hashlib.sha256((run/'run_manifest.json').read_bytes()).hexdigest()==parent['D']['original_manifest_sha256']
deadline=datetime.datetime.fromisoformat(manifest['benchmark_started_at'])+datetime.timedelta(seconds=manifest['total_timeout_seconds']);assert deadline.isoformat()=='2026-11-18T08:42:32.657316+00:00'
assert datetime.datetime.now(datetime.timezone.utc)<deadline-datetime.timedelta(seconds=60)
for name in ['run_aisbench_cold_hot.py','audit_aisbench.py']:
    p=root/'benchmark_pipeline_sg_g488'/name
    expected={'run_aisbench_cold_hot.py':'07b8f4df719bd8064d4bc413295fcf46eb7eb216f062c596bc4ca9bf203b64f3','audit_aisbench.py':'88a3c37c4bb1fa2787cd1f8959e87128ea45b24315b6421b80c7b73cceda9425'}[name]
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
proofpath=root/'sg_pd_g316/D_256K_warm_current_cache_proof_g496.json'
assert hashlib.sha256(proofpath.read_bytes()).hexdigest()==parent['D_proof_sha256']
assert json.loads(proofpath.read_text())==parent['D']
gate=json.loads((root/'service_gate_sglang_g404_first_warm.json').read_text())
assert gate['warmup_only'] is True
assert gate['roles']['decode']['container_id']==parent['D']['active_D_container_id']
assert gate['roles']['prefill']['container_id']==parent['P']['container_id']
ledger=json.loads((c/'formal_submissions_sglang.json').read_text());assert not any(x['input_tokens']==262144 for x in ledger['submissions'])
mirror=root/'P_256K_independent_cache_proof_g514.json';assert not mirror.exists();mirror.write_text(json.dumps(parent['P'],indent=2)+'\n')
gate.update(warmup_only=False,P_compiler_cache_verified_after_warmups=True,validated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),corrected_D_warm_cache_proof=str(proofpath),corrected_D_warm_cache_proof_sha256=parent['D_proof_sha256'],P_compile_proof=str(mirror),collector_correction_scope='Current D physical roots; original warm snapshots and manifest immutable. First formal only; no benchmark deadline reset.')
gatepath=root/'service_gate_sglang_g514_first256_formal.json';assert not gatepath.exists();gatepath.write_text(json.dumps(gate,indent=2)+'\n')
argv=list(old['argv']);assert argv[2]==str(root/'benchmark_pipeline_sg_g199/run_aisbench_cold_hot.py');argv[2]=str(root/'benchmark_pipeline_sg_g488/run_aisbench_cold_hot.py');argv[argv.index('--gate')+1]=str(gatepath)
check=subprocess.run(argv+['--preflight-only'],cwd=root,capture_output=True,text=True,timeout=120);(c/'SG_G316_256K_first_formal_g514_preflight.log').write_text(check.stdout+check.stderr);assert check.returncode==0,(check.stdout+check.stderr)[-2000:]
result=json.loads(check.stdout.strip().splitlines()[-1]);assert result['preflight']=='passed' and result['resume_completed_to_skip']==[f'warmup{i}_262144' for i in range(1,6)]
handlepath=c/'SG_G316_256K_first_formal_g514_handle.json';assert not handlepath.exists()
with (c/'SG_G316_256K_first_formal_g514.log').open('x') as log:
    child=subprocess.Popen(['flock','--nonblock','<COORD_ROOT>/<ROLE_NODE>-client-g42.lock',*argv],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(2);h={'pid':child.pid,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':argv,'original_deadline_utc':deadline.isoformat(),'scope':'First256cold32/hot32 only after five independentwarm16, parent P636/currentD411 fullinterval proof and primary stability. Completedwarmups reaudited/notresent; correctedcollector only, originalmanifest/assets/serviceIDs/deadline/firstclaimledger preserved.'};handlepath.write_text(json.dumps(h,indent=2)+'\n');print('G514_FIRST256_CORRECTED_FORMAL_START_JSON '+json.dumps({'handle':h,'actual_live':child.poll() is None}),flush=True)
"""
a.output.write_text("python3 - <<'REMOTE_CODE'\nPAYLOAD="+repr(payload)+'\n'+body+'REMOTE_CODE\n')
print('Prepared concrete first256 formal resume; no requests sent:',a.output)
