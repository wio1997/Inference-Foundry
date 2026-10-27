from pathlib import Path
import subprocess,json,time,re,traceback,hashlib
p=Path('/data/tiankuan/wio/910c-deepinfer-172');v='prefill-oproj-v15';status=p/'runs/prefill-oproj-v15-profile-status.json';assert not status.exists();s={'state':'WAITING_FOR_TWO_ROUNDS','purpose':'one C8/32768/1-token operator diagnostic after formal runs; not a benchmark score','started':time.time()}
def save(state):s.update(state=state,updated=time.time());status.write_text(json.dumps(s,indent=2))
try:
 save('WAITING_FOR_TWO_ROUNDS')
 for _ in range(180):
  f=p/'runs/prefill-oproj-v15-benchmark-status.json'
  if f.exists() and json.loads(f.read_text())['state']!='RUNNING':break
  time.sleep(2)
 else:raise RuntimeError('formal rounds unfinished; no diagnostic dispatched')
 assert json.loads(f.read_text())['state']=='PASS','formal correctness failure; do not profile a rejected path'
 progress=json.loads((p/'benchmarks'/v/'progress.json').read_text());assert len(progress['phases'])==2 and all(x['state']=='PASS' for x in progress['phases'])
 if progress['overall_goal_met']:save('SKIPPED_GOAL_MET');raise SystemExit(0)
 control=p/'serving'/v
 for job in (control/'jobs').iterdir():assert len(list(job.glob('result-dp*.json')))==2 and not list(job.glob('error-rank*.txt')),job
 smi=subprocess.check_output(['npu-smi','info'],text=True);pids=[int(x) for x in re.findall(r'^\|\s*\d+\s+\d+\s*\|\s*(\d+)\s*\|',smi,re.M)];assert len(pids)==16
 for pid in pids:
  cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode();assert 'ascend910c_w8a8_worker.py' in cmd and '/serving/prefill-oproj-v15' in cmd,('other owner',pid)
 old=control/'jobs'/progress['phases'][1]['path_proof']['job'];body=json.loads((old/'request.json').read_text());assert len(body['requests'])==8 and all(len(x['token_ids'])==32768 for x in body['requests'])
 for row in body['requests']:row['max_tokens']=1;row['ignore_eos']=True
 body['profile_npu']=True;jobid='profile-oproj-v15-prefill';directory=control/'jobs'/jobid;directory.mkdir(exist_ok=False);(directory/'request.json').write_text(json.dumps(body,separators=(',',':')))
 tmp=control/'profile-current.tmp';tmp.write_text(json.dumps({'id':jobid}));tmp.replace(control/'current.json');s.update(job=str(directory),npu_pids=pids,formal_rounds_added=0);save('CAPTURING_AND_ANALYSING')
 for _ in range(180):
  errors=list(directory.glob('error-rank*.txt'));assert not errors,[x.read_text()[-1000:] for x in errors]
  reports=list(directory.rglob('kernel_details.csv'));traces=list(directory.rglob('trace_view.json'))
  if len(list(directory.glob('result-dp*.json')))==2 and reports and traces:break
  time.sleep(2)
 else:raise RuntimeError('profile/report timeout; retain all raw artifacts')
 # Synchronous trace handler exports its reports after runner completion.
 time.sleep(3)
 for dp in range(2):
  new=json.loads((directory/f'result-dp{dp}.json').read_text());ref=json.loads((old/f'result-dp{dp}.json').read_text());assert new['cached_tokens']==ref['cached_tokens'];assert len(new['generated_token_ids'])==4
  assert all(len(a)==1 and a[0]==b[0] for a,b in zip(new['generated_token_ids'],ref['generated_token_ids']))
 s.update(kernel_reports=[str(f) for f in reports],traces=[str(f) for f in traces],first_tokens_match_frozen=8,actual_cached_tokens=235904);save('PASS')
except BaseException as exc:
 if isinstance(exc,SystemExit) and exc.code==0:raise
 s['error']=repr(exc);save('FAIL');traceback.print_exc();raise
