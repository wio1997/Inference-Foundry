"""Audit exact returned client artifacts; no remote access or output-bitwise gate."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 run=ROOT/'runs/20261017T1779-vllm-shared-long-semantic-v2';receipt=json.loads((run/'receipt.json').read_text());assert receipt['rc']==0 and all(x==0 for x in receipt['model_precheck'].values());raw=(run/'node-43.json').read_bytes();outer=json.loads(raw);assert outer['rc']==0
 rows=[x.split('=',1)[1] for x in outer['stdout'].splitlines() if x.startswith('GLM53_LONG_ARTIFACTS=')];assert len(rows)==1;artifacts=json.loads(rows[0])
 for name,item in artifacts.items():
  if 'text' in item:assert len(item['text'].encode())==item['bytes'] and hashlib.sha256(item['text'].encode()).hexdigest()==item['sha256']
 report=json.loads(artifacts['semantic-result.json']['text']);assert report['semantic_pass'] and len(report['cases'])==len(report['gateway'])==6 and report['external_backend_HTTP_attempts']==report['successful_backend_inferences']==12 and report['actual_successful_output_tokens']==2408
 assert report['native_versions']['ais_bench_benchmark']=='3.1.0' and report['native_versions']['torch']=='2.10.0+cpu'
 assert len(report['metadata_gates'])==len(report['cache_readback_guards'])==2
 selected=[]
 for case in [0,1]:
  factual=report['cases'][case*3:case*3+3];assert [x['case'] for x in factual]==['cold-PD','prefix-warmup-PD','cross-DP-storage-PD']
  for row in factual:
   assert row['count_pass'] and row['pd_lifecycle_pass'] and row['semantic_pass'] is not False
   usage=row['native']['extra_details_data']['glm53_service_usage'];assert usage['prompt_tokens']==row['required_input'] and usage['completion_tokens']==row['required_output']
   assert row['required_input']==(74400 if row['case']=='prefix-warmup-PD' else 80000) and row['required_output']==(1 if row['case']=='prefix-warmup-PD' else 600)
   if row['required_output']==600:assert row['observed_answer_fields'][0]==['731924680531','865103792468'][case]
  gate=report['metadata_gates'][str(case)];m=gate['metadata'];assert gate['pass'] and gate['rc']==0 and m['init_bm'] is False and m['init_rc']==m['close_rc']==0 and m['known_key_count']==581 and len(m['bytes_by_key'])==581 and all(x==12369920 for x in m['bytes_by_key']) and m['object_bytes']==7186923520
  proof=report['cache_readback_guards'][str(case)];assert proof['pass'] and proof['native_P_cached_tokens']==proof['expected_reusable_prefix_tokens']==74368
  p=report['gateway'][case*3+2]['route']['P'];assert p['global_dp']==case*2+1
  selected.append({'semantic_case':case,'expected_fact':factual[0]['expected_answer'],'cold_and_cross_fact_pass':True,'actual_input_output':{'cold':[80000,600],'warm':[74400,1],'cross':[80000,600]},'cross_P_global_DP':p['global_dp'],'native_P_cached_tokens':proof['native_P_cached_tokens'],'known_prefix_keys':581,'known_unique_object_bytes':m['object_bytes'],'metadata_only_init_bm':False,'SDK_payload_read_attribution':'Requires independently captured native model full-batch SDK audit; metadata is not payload readback.'})
 (run/'semantic-result.local.json').write_text(artifacts['semantic-result.json']['text']);(run/'artifact-index.json').write_text(json.dumps({k:{a:b for a,b in v.items() if a!='text'} for k,v in artifacts.items()},indent=2)+'\n')
 result={'project_id':'glm53','framework':'vllm','shared_run_id':'20261017T1766-vllm-shared-pool-v4','pass':True,'qualification':'SIX_LONG_FACT_COUNT_LIFECYCLE_KNOWN_KEYS_AND_NATIVE_P_PREFIX_CACHE_GUARDS_PASSED_NATIVE_POSTAUDIT_PENDING','backend_HTTP_attempts':12,'successful_backend_inferences':12,'successful_output_tokens':2408,'logical_functional_cases':6,'cases':selected,'whole_long_output_equality_required':False,'whole_long_output_hash_observations':report['whole_output_hash_equality'],'formal_requests':0,'baseline_accepted':False,'source_evidence_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [run/'receipt.json',run/'node-43.json',run/'semantic-result.local.json']}}
 (ROOT/'results/vllm-shared-long-semantic-v2.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
