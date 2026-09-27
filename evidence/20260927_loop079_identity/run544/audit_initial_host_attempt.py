import ast,collections,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path('/data/wio/Inference_Foundry'); old=ROOT/'evidence/20260927_loop079_identity/run542'; out=ROOT/'evidence/20260927_loop079_identity/run544';out.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
orig=old/'validator_original.py';fixed=ROOT/'scripts/loop079_formal_ledger_server_validate.py'
assert sha(orig)=='88e5997e1e4edcab840e1b792eb8aa6c3dbf5907d62f9b81f0b363c5de595dee'
assert sha(fixed)=='52454f39a1fdf45c2e52981ef22a7b7b1e3c299338b049141c286a1a685b6473'
assert orig.read_text().replace('            if phase == "measured":\n                require(done["target_graph_mode"] == "CUDAGraphMode.FULL",\n                        f"{phase}: Runner did not report FULL target graph")','            require(done["target_graph_mode"] == "FULL",\n                    f"{phase}: Runner did not report FULL target graph")')==fixed.read_text()
args=['--ledger-dir',str(old/'ledger'),'--client-report',str(old/'client_admission.json'),'--warmup-report',str(old/'warmup_client_admission.json'),'--warmup-dir',str(old/'warmup_client'),'--measured-dir',str(old/'measured_client'),'--phase-transitions',str(old/'phase_transitions.jsonl'),'--run-id','LOOP079-RUN542']
checks=[]
for name,script in [('original',orig),('corrected',fixed)]:
 r=subprocess.run([sys.executable,str(script),*args,'--output',str(out/(name+'_server_replay.json'))],capture_output=True,text=True)
 (out/(name+'_replay.log')).write_text(r.stdout+r.stderr)
 checks.append({'name':name,'exit':r.returncode})
 if name=='original':assert r.returncode!=0 and 'Runner did not report FULL target graph' in r.stderr
 else:assert r.returncode==0,r.stderr
assert (out/'corrected_server_replay.json').read_bytes()==(ROOT/'evidence/20260927_loop079_identity/run543/server_admission_replayed.json').read_bytes()
dataset='/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl'
r=subprocess.run([sys.executable,str(ROOT/'scripts/loop079_formal_ledger_client_validate.py'),'--dataset',dataset,'--warmup-dir',str(old/'warmup_client'),'--measured-dir',str(old/'measured_client'),'--output',str(out/'client_replay.json')],capture_output=True,text=True)
(out/'client_replay.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
assert (out/'client_replay.json').read_bytes()==(old/'client_admission.json').read_bytes()
assert (old/'source_before.sha256').read_bytes()==(old/'source_after.sha256').read_bytes()
assert (old/'scripts_before.sha256').read_bytes()==(old/'scripts_after.sha256').read_bytes()
install=json.loads((old/'install.json').read_text()); restore=json.loads((old/'restore.json').read_text())
assert install['installed'] and restore['restored'] and not restore['helper_sha_drift'] and install['files']==restore['files']
for v in restore['files'].values(): assert sha(Path(v['path']))==v['original']
status=dict(l.split('=',1) for l in (old/'cleanup_status.txt').read_text().splitlines())
assert status['final_exit']==status['run_exit']==status['admission_exit']=='1'
for k in ('stop_exit','stop_verify_exit','restore_exit','sha_exit','sha_compare_exit','script_sha_exit','script_compare_exit'):assert status[k]=='0'
assert (old/'server_post_count.txt').read_text().strip()=='96'
rows=[json.loads(l) for p in (old/'ledger').glob('pid*.jsonl') if '.flush.' not in p.name for l in p.read_text().splitlines()]
summary={}
for phase in ('warmup','measured'):
 phase_rows=[r for r in rows if r.get('phase')==phase]; done=[r for r in phase_rows if r['event']=='runner_done'];bulk=[r for r in phase_rows if r['event']=='scheduler_append' and r['bulk']]
 assert len(done)==32 and {r['target_graph_mode'] for r in done}=={'FULL'}
 summary[phase]={'runner_done':len(done),'graph_modes':sorted({r['target_graph_mode'] for r in done}),'prebulk_G_sum':sum(r['g_before'] for r in bulk),'bulk_incoming_sum':sum(len(r['incoming_raw_ids']) for r in bulk),'bulk_admitted_sum':sum(len(r['admitted_raw_ids']) for r in bulk),'rank0_cycles':[r['cycles'] for r in sorted(done,key=lambda r:r['cohort']) if r['rank']==0]}
inputs=[orig,fixed,old/'validator_selftest_original.py',old/'cleanup_status.txt',old/'stop_verify.log',old/'source_before.sha256',old/'source_after.sha256',old/'scripts_before.sha256',old/'scripts_after.sha256',old/'install.json',old/'restore.json',old/'client_admission.json',old/'warmup_phase_barrier.json',old/'warmup_client_admission.json',old/'phase_transitions.jsonl',old/'server_validate.log',old/'server_post_count.txt',ROOT/'evidence/20260927_loop079_identity/run543/server_admission_replayed.json',ROOT/'../vllm_ascend_26/framework/vllm/vllm/config/compilation.py']
inputs+=list((old/'ledger').glob('pid*.jsonl'))+list((old/'warmup_client').glob('*.json'))+list((old/'measured_client').glob('*.json'))
manifest={str(p):sha(p) for p in sorted(inputs)}
(out/'input_hashes.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
result={'verdict':'PASS_posthoc_Host_lineage_only','original_Run542_controller_status':'FAIL_preserved','checks':checks,'client_replay_exit':r.returncode,'client_replay_byte_identical':True,'corrected_server_replay_byte_identical_to_Run543':True,'phase_summary':summary,'source_restore_valid':True,'raw_input_hash_count':len(manifest),'original_validator_sha256':sha(orig),'corrected_validator_sha256':sha(fixed)}
(out/'audit_results.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps(result))
