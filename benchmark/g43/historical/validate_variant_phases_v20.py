"""Verify only terminal AISBench phases; preserve raw evidence and actual input identity."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,math

p=argparse.ArgumentParser();p.add_argument('--engine',choices=['sglang','vllm'],required=True);p.add_argument('--variant',choices=['p4k_screen','p4k_full','p2k_screen','p2k_full','p2kh768_screen','p2kh768_full','t4_screen','t4_full','v3_repeat32k','p4kh1280_screen','p4kh1280_full','p4kh1280mhc_screen','p4kh1280mhc_full','p3kh1024mhc_screen','p3kh1024mhc_full','p3kh1024mhcg2_screen','p3kh1024mhcg2_full','v2_repeat32k','p3kh1024mhcg33_screen','p3kh1024mhcg33_full','ascops35_screen','ascops35_full','ascops36_screen','ascops36_full','ascops37_screen','ascops37_full','ascops38_screen','ascops38_full','ascops39_screen','ascops39_full','ascops42_screen','ascops42_full','ascops43_screen','ascops43_full','ascops43r2_screen','ascops43r2_full'],required=True);a=p.parse_args()
r=Path(__file__).resolve().parent
state=json.loads((r/f'aisbench_run_state_{a.engine}_{a.variant}.json').read_text());root=r/'results'/a.engine/state['run_id']
events=[json.loads(line) for line in (r/f'logs/aisbench_{a.engine}_{a.variant}_runner.log').read_text().splitlines() if line.startswith('{')]
terminal={(e['input_tokens'],e['subphase']) for e in events if e.get('run_id')==state['run_id'] and e.get('phase')=='aisbench_cli_terminal' and e.get('exit_code')==0}
spec=importlib.util.spec_from_file_location('accepted_AISBench_analyzer',r/'analyze_aisbench_results.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
checks=[];metrics=[]
for length in [32768,65536,131072,262144]:
 for phase,count in [('warmup',16),('formal',32)]:
  if (length,phase) not in terminal:continue
  report=root/f'{length}_{phase}_request_validation.json'
  paths=list((root/f'{phase}_{length}').rglob('*_details.jsonl'));assert len(paths)==1,paths
  detail=paths[0];sha=hashlib.sha256(detail.read_bytes()).hexdigest()
  if report.is_file():
   check=json.loads(report.read_text());assert check['details_sha256']==sha,'Previously accepted detail changed'
  else:
   rows=[json.loads(line) for line in detail.read_text().splitlines() if line.strip()];assert len(rows)==count
   assert all(row['success'] is True and row['input_tokens']==length and row['output_tokens']==1024 for row in rows)
   expected=[json.loads(line)['question'] for line in (r/f'datasets/{phase}_input_{length}.jsonl').read_text().splitlines() if line.strip()]
   assert sorted(row['input'] for row in rows)==sorted(expected),'Actual request inputs differ from shared exact dataset'
   common=detail.with_name(detail.name.replace('_details.jsonl','.json'));d=json.loads(common.read_text())
   def metric(k):
    assert set(d[k])=={'total'};return float(str(d[k]['total']).split()[0])
   assert metric('Total Requests')==metric('Success Requests')==count and metric('Failed Requests')==0
   assert metric('Max Concurrency')==16 and metric('Total Input Tokens')==count*length and metric('Total Generated Tokens')==count*1024
   duration=metric('Benchmark Duration')/1000;assert duration>0
   assert math.isclose(metric('Output Token Throughput'),count*1024/duration,rel_tol=1e-6,abs_tol=.001)
   check={'input_tokens':length,'phase':phase,'successful':count,'output_tokens':1024,'max_concurrency':16,'exact_actual_inputs_match_shared_dataset':True,'details':str(detail.relative_to(root)),'details_sha256':sha,'duration_s':duration,'output_tokens_per_s':metric('Output Token Throughput')}
   report.write_text(json.dumps(check,indent=2)+'\n')
  checks.append(check)
  if phase=='formal':
   path=root/f'verified_results_{length}.json'
   row=module.analyze(root,length)
   if path.is_file():assert json.loads(path.read_text())==[row]
   else:path.write_text(json.dumps([row],indent=2)+'\n')
   metrics.append(row)
samples=[json.loads(line) for line in (root/'npu_ownership_samples.jsonl').read_text().splitlines() if line.strip()]
assert samples and all(not x['foreign'] and x['own_npu_process_count']>=16 for x in samples)
for check in checks:
 prefix=f"{check['input_tokens']}_{check['phase']}_"
 assert any(x['tag']==prefix+'before' for x in samples) and any(x['tag']==prefix+'after' for x in samples)
result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'engine':a.engine,'run_id':state['run_id'],'completed_phases_verified':len(checks),'formal_profiles_verified':len(metrics),'phase_checks':checks,'formal_metrics':metrics,'ownership_samples_verified':len(samples),'all_four_profiles_complete':len(checks)==8 and len(metrics)==4,'whole_model_quality_verified':False,'scope':'Actual token/success/C16/input identity and aggregate arithmetic; no repeatability or precision-equivalence acceptance'}
(root/'completed_phase_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
