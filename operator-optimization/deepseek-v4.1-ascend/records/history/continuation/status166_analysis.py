import pathlib,json,socket,time
r=pathlib.Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030');out={'hostname':socket.gethostname(),'now':time.time()}
p=r/'reference-latest.json'
if p.exists():
 j=json.loads(p.read_text());out['reference_latest']=j;p=pathlib.Path(j['root'])/'state.json'
 if p.exists():out['reference_state']=json.loads(p.read_text())
p=r/'moe-observer-latest.json'
if p.exists():
 j=json.loads(p.read_text());out['observer_latest']=j;root=pathlib.Path(j['run_root']);out['observer_files_ready']=root.exists()
 for n in ['controller-state.json','controller-error.json','top-level-error.json','capture-result.json','restore-result.json','controller-finished.json','value-review-state.json']:
  p=root/n
  if p.exists():out[n]=json.loads(p.read_text())
 p=root/'value-review/summary.json'
 if p.exists():
  summary=json.loads(p.read_text());out['value_review_summary']={k:v for k,v in summary.items() if k not in ['matched_cohort_examples','mismatch_examples']}
 p=pathlib.Path(j['stderr'])
 if p.exists():out['controller_stderr_tail']=p.read_text()[-2500:]
 p=root/'runtime/logs/service.log'
 if p.exists():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-6000));out['observer_service_log_tail']='\n'.join(f.read().decode(errors='replace').splitlines()[-12:])
print(json.dumps(out))
