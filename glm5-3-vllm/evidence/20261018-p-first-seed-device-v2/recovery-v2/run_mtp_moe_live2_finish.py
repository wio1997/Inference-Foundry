from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time
repo=Path('/Users/wio/work/Inference-Foundry-execution-plan')
r=repo/'glm5-3-vllm/evidence/20261018-p-first-seed-device-v2/recovery-v2'
b=Path('/Users/wio/work/glm53-execution-evidence-backup/20261018-p-first-seed-device-v2/recovery-v2')
start=time.monotonic()
def record(p):
 return {'path':str(p.relative_to(repo)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
while True:
 analysis=list((r/'mtp-moe-live2-c4-copy-v2').glob('*/metrics.json'))
 backups=[list((r/n).glob('*/artifacts-backup.json')) for n in ['mtp-moe-graph-export','mtp-moe-live2-c4-export']]
 if analysis and all(backups):break
 assert time.monotonic()-start<1500
 time.sleep(5)
x=json.loads(analysis[0].read_text());assert x['completed'] and x['all16_selected_shape_SDMA_samples']
proof=analysis[0].parent
q=subprocess.run([sys.executable,str(repo/'glm5-3-vllm/tools/backup_artifacts.py'),'--run',str(proof),'--control','/tmp/glm53-active-20261018.sock','--backup-root',str(b/'mtp-moe-live2-c4-copy'),'--chunk-size-mib','8'],cwd=repo,capture_output=True,text=True,timeout=240)
(r/'mtp-moe-live2-c4-copy-backup-collector.txt').write_text(q.stdout+q.stderr);assert q.returncode==0,q.stdout+q.stderr
physical={'completed':True,'all16_selected_shape_SDMA_samples':True,'all16_added_copy_SDMA':x['all16_added_copy_SDMA'],'source':record(analysis[0]),'ranks':[{k:z[k] for k in ['rank','candidate_input_copies','closed_copy_chains','unclosed_copy_chains','observed_selected_shapes','copy_coverage_by_shape','all_observed_copy_data_tasks_SDMA','data_task_types']} for z in x['ranks']],'limitation':x['limitation'],'decision_scope':'Existing same native contiguous device copy path; each actual selected shape on each rank has physical SDMA samples, and no linked task is non-SDMA. Full instance closure remains explicitly false where API/device connection missing. Bounded performance experiment permitted; no claim every new copy instance was individually proved.'}
(r/'mtp-moe-graph-live2-physical-gate-v1.json').write_text(json.dumps(physical,indent=2)+'\n')
receipts=[v[0] for v in backups]+[proof/'artifacts-backup.json']
for p in receipts:
 y=json.loads(p.read_text());a=Path(y['local_archive']);assert a.exists() and a.stat().st_size==y['bytes']
(r/'mtp-moe-graph-live2-backup-gate-v1.json').write_text(json.dumps({'completed':True,'sources':[record(p) for p in receipts],'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_related_cpu_export_and_backup_finished_before_formal':True},indent=2)+'\n')
print(json.dumps({'phase':'physical_sample_and_backup_gate','ranks':len(x['ranks']),'closed_copies':sum(z['closed_copy_chains'] for z in x['ranks']),'unclosed_copies':sum(z['unclosed_copy_chains'] for z in x['ranks'])}),flush=True)
for name in ['run_mtp_moe_live2_formal.py','run_mtp_moe_live2_cancel.py']:
 q=subprocess.run([sys.executable,str(r/name)],cwd=repo)
 assert q.returncode==0,(name,q.returncode)
print('formal_and_cancellation_finished',flush=True)
