#!/usr/bin/env python3
"""Validate only the bounded source/cost claims actually made by this review."""
from pathlib import Path
from hashlib import sha256
import json,math
ROOT=Path('audit/execution-search')
def main():
 r=json.loads((ROOT/'model-gate-source-review.json').read_text());old=json.loads((ROOT/'worker-protocol-source-review.json').read_text());c=json.loads((ROOT/'model-gate-source-controls.json').read_text());state=json.loads((ROOT/'model-gate-state.json').read_text());p=json.loads((ROOT/'model-gate-parent-prices.json').read_text())
 assert len(r['sources'])==len({s['runtime_path'] for s in r['sources']})==27
 for s in r['sources']:assert sha256(Path(s['local_path']).read_bytes()).hexdigest()==s['runtime_sha256']
 for d in [r,p]:
  for x in d['inputs']:assert sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256']
 assert len(c['records'])==c['controls']==85 and c['reader_sha256']==sha256(Path('tools/probe_model_gate_source_controls.py').read_bytes()).hexdigest()
 assert len(state['flags']['processes'])==21 and all(x['status']=='READ' and x['VLLM_MOE_SKIP_PADDING'] is None for x in state['flags']['processes'])
 assert state['descriptor']['sha256']=='2f4020a0a12daa621cfd2ff9d09a2c73e5674af7c624ef47df5c66796c3774bf' and state['descriptor']['global_rotation_present']
 assert len(r['branches'])==len(old['branches'])==156;changes=[];notes=[]
 for a,b in zip(old['branches'],r['branches']):
  assert (a['root_owner'],a['root_method'],a['line'],a['sha256'],a['condition'])==(b['root_owner'],b['root_method'],b['line'],b['sha256'],b['condition'])
  if a['condition_state']!=b['condition_state']:changes.append((b['root_owner'],b['line']));assert a['condition_state']=='UNRESOLVED_INPUT_OR_RUNTIME' and b['condition_state'].startswith('MODEL_FIXED_')
  if 'model_source_leg_review' in b:notes.append((b['root_owner'],b['line']));assert b['model_source_leg_review']['observed_outcomes']==[]
  assert not b['observed_outcomes'] and not b['fusion_search_complete']
 assert len(changes)==3 and len(notes)==5 and sum(b['condition_state']=='UNRESOLVED_INPUT_OR_RUNTIME' for b in r['branches'])==34
 counter=next(x for x in c['records'] if x['kind']=='FIA_graph_padding_is_not_SP_modulo_mask');assert counter['rank']==7 and counter['global_row']>=counter['actual_tokens_example'] and counter['produced_mask'] is False and counter['actual_domain_padding_would_be'] is True
 cap=next(x for x in c['records'] if x['kind']=='actual_source_A3_capability_set');assert cap['hash_vision_native'] and not cap['unrotated_Engram_gate']
 for x in c['records']:
  if x['kind']=='original_Engram_gate_dispatch' and x['rotation_non_none']:assert x['events']==['generic_gate']
  if x['kind']=='original_attention_metadata_and_empty_query_guard' and not x['metadata_none'] and not x['local_tokens']:assert x['events']==['global_cache_prepare','TP_exchange','O_projection']
 primary=json.loads((ROOT/'model-rotated-engram-parent-primary.json').read_text());a1=ROOT/'model-rotated-engram-parent-a1-label-correction.json';assert primary['original_preprice_primary_SHA256']==sha256(a1.read_bytes()).hexdigest();assert 'quantized wkv' in json.loads(a1.read_text())['scope'][0] and 'descriptor FLOAT' in primary['scope'][0]
 for name in ['model-pad-row-parent-primary.json','model-rotated-engram-parent-primary.json']:
  for pin in json.loads((ROOT/name).read_text())['inputs']:assert sha256(Path(pin['path']).read_bytes()).hexdigest()==pin['sha256']
 assert len(p['source_intersections'])==6
 for s in p['row_parent_separate_DFC_shape_cohort']['summaries']:assert s['representative_ranks']==2 and s['representative_ep']==[0,8]
 assert len(p['row_parent_separate_DFC_shape_cohort']['groups'])==4
 assert math.isclose(p['Engram_original_C8_first_token']['whole_parent_scopes']['gate_us']['remaining_0']['maximum_request_percent'],.4896826548506013,rel_tol=1e-6)
 for d in [r,c,p]:assert not d['whole_chain_complete'] and not d['new_primitive_admissions'] and not d['new_framework_replacement_admissions'] and not d['new_NPU_work'] and not d['new_profile'] and not d['service_requests'] and not d['remote_writes']
 names=['model-gate-dependencies.json','model-gate-consumer-dependencies.json','model-gate-state.json','model-gate-selector-lexical.json','model-gate-source-controls.json','model-gate-control-source-excerpts.json','model-gate-source-review.json','model-gate-parent-prices.json','model-gate-review-history.json','model-pad-row-parent-primary.json','model-rotated-engram-parent-primary.json','model-rotated-engram-parent-a1-label-correction.json','model-gate-frontier.json','模型guard与SP行及旋转Engram交付判断.md']
 out=dict(status='PASS_BOUNDED_MODEL_GUARD_SOURCE_AND_QUALIFIED_OLD_COST_NOT_NATIVE_ADMISSION',source_pins=27,initial_exec_flags=21,source_controls=85,original_sites=156,new_bindings=3,bound=122,pending=34,model_guard_source_legs=5,old_Engram_source_intersections=6,DFC_shape_representative_ep=[0,8],all16_DFC_or_full_request_claim=False,every_replacement_bitwise=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,whole_chain_complete=False,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],artifacts=[dict(path=str(ROOT/n),sha256=sha256((ROOT/n).read_bytes()).hexdigest()) for n in names],verifier_sha256=sha256(Path(__file__).read_bytes()).hexdigest());(ROOT/'model-gate-integrity.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['artifacts','verifier_sha256']}))
if __name__=='__main__':main()
