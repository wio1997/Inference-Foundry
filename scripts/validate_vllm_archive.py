"""Offline source/evidence and migration checks; no model or NPU execution."""
from pathlib import Path
import ast,hashlib,json,re
R=Path(__file__).resolve().parents[1]
def main():
 scope=json.loads((R/'metadata/framework_scope.json').read_text());assert scope['active_framework']=='vllm' and scope['branch'].startswith('vllm-')
 assert not (R/'code/sglang').exists() and not (R/'code/sg_pd_g334').exists()
 for entry in json.loads((R/'provenance/migration.json').read_text())['selected_existing_files']:
  p=R/entry['artifact'];assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['branch_published_sha256'],p
  if entry['byte_identical']:assert entry['source_published_sha256']==entry['branch_published_sha256']
 for e in json.loads((R/'evidence/source_provenance.json').read_text()):assert hashlib.sha256((R/e['artifact']).read_bytes()).hexdigest()==e['sha256']
 for e in json.loads((R/'evidence/components/provenance.json').read_text()):assert hashlib.sha256((R/e['artifact']).read_bytes()).hexdigest()==e['published_sha256']
 g=json.loads((R/'evidence/native_graph_contract.json').read_text());assert not g['actual_full_model_target_and_draft_graph_replay_verified']
 for e in g['sources'].values():assert hashlib.sha256((R/e['artifact']).read_bytes()).hexdigest()==e['sha256']
 source=R/'evidence/connector_source_G359.json'
 if source.is_file():
  for e in json.loads(source.read_text())['files']:assert hashlib.sha256((R/e['artifact']).read_bytes()).hexdigest()==e['published_sha256']==e['original_sha256']
  contract=json.loads((R/'evidence/connector_contract_G365.json').read_text());assert contract['passed'] and len(contract['cases'])==9
  for name,digest in contract['source_SHA256'].items():assert hashlib.sha256((R/'code/fixed_connectors_G359'/name).read_bytes()).hexdigest()==digest
 audit=R/'evidence/PR12_source_comparison_G370.json'
 if audit.is_file():
  a=json.loads(audit.read_text());assert a['compared_files']==113 and len(a['files'])==113
  assert a['counts']=={'byte_identical':39,'different':44,'missing_from_fixed':30,'samepath_missing_with_relocated_same_bytes':6}
  for e in a['selected_native_sources']:assert hashlib.sha256((R/e['artifact']).read_bytes()).hexdigest()==e['sha256']
  q=json.loads((R/'evidence/PR12_quant_contract_G372.json').read_text());assert q['passed'] and len(q['cases'])==8
  for label,rel in [('fixed0.30','fixed'),('PR12','reference')]:assert hashlib.sha256((R/'code/pr12_quant_contract'/rel/'patch_dspark.py').read_bytes()).hexdigest()==q['source_sha256'][label]
 receipt=json.loads((R/'evidence/components/G50_vllm/receipt.json').read_text());result=receipt['results'];assert result['all8_passed'] and result['worker_exit_codes']==[0]*8
 assert {x['rank'] for x in result['reports']}==set(range(8));cases=graphs=replays=0
 for report in result['reports']:
  assert report['passed'] and report['device_name']=='Ascend910B1' and len(report['cases'])==10 and len(report['graphs'])==2
  for c in report['cases']:assert c['bitwise_bf16_match'] and c['global_vocab_start']==384000000
  for v in report['graphs']:assert v['bitwise_bf16_match'] and v['changed_host_weights_and_NPU_ids_replays']==3 and len(set(v['output_sha256']))==3
  cases+=len(report['cases']);graphs+=len(report['graphs']);replays+=sum(x['changed_host_weights_and_NPU_ids_replays'] for x in report['graphs'])
 assert (cases,graphs,replays)==(80,16,48)
 assert receipt['state']['phase']=='stopped_released' and receipt['state']['cleanup']['all8_process_lists_clear']
 failed=json.loads((R/'evidence/components/G49_vllm_import_failure/receipt.json').read_text());assert failed['state']['phase']=='failed_released' and not failed['results']['all8_passed']
 ref=json.loads((R/'metadata/reference_20261117_assessment.json').read_text());assert ref['same_config_bytes'] and ref['reference_config_sha256']==ref['current_pinned_config_sha256']
 assert ref['current_required']['framework']=='vLLM-Ascend 0.30.0rc2' and ref['reference']['framework']=='vLLM-Ascend 0.32.0rc1'
 assert ref['reference']['draft_enforce_eager'] and ref['current_required']['target_and_draft_full_graph']
 readiness=json.loads((R/'evidence/readiness.json').read_text());assert not readiness['full_model_PD_accepted'] and readiness['formal_performance_results'] is None
 links=0
 for p in R.rglob('*'):
  if not p.is_file() or '.git' in p.parts or '__pycache__' in p.parts:continue
  if p.suffix=='.py':ast.parse(p.read_text())
  data=p.read_bytes();assert not re.search(rb'(?:lifang|Hzzgkj)'+rb'@|117\.131\.156\.71|39\.183\.161\.133|/data1?/tiankuan|bms-lifang-000[1-4]',data),p
  if p.suffix=='.md':
   for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
    if '://' in link or link.startswith('#'):continue
    assert (p.parent/link.split('#')[0]).is_file(),(p,link);links+=1
 print(json.dumps({'framework_scope':'vllm','migration_and_native_SHA_verified':True,'actual_component_cases':cases,'graphs':graphs,'changed_replays':replays,'relative_links':links,'source_syntax_privacy':True,'fullmodel_PD_performance_accepted':False}))
if __name__=='__main__':main()
