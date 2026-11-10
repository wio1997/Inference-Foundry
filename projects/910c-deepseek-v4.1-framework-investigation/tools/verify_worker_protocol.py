#!/usr/bin/env python3
"""Verify bounded source constraints and replay the whole-family price ledger."""
from pathlib import Path
from hashlib import sha256
import json,math,statistics
ROOT=Path('audit/execution-search')
def main():
 r=json.loads((ROOT/'worker-protocol-source-review.json').read_text());old=json.loads((ROOT/'input-resource-source-review.json').read_text());price=json.loads((ROOT/'worker-position-parent-price.json').read_text());controls=json.loads((ROOT/'worker-execution-protocol-controls.json').read_text());primary=json.loads((ROOT/'worker-position-parent-primary.json').read_text());fine=json.loads(Path('audit/framework-timing-171/function-analysis.json').read_text())
 assert len(r['sources'])==24 and len({s['runtime_path'] for s in r['sources']})==24
 for s in r['sources']:assert sha256(Path(s['local_path']).read_bytes()).hexdigest()==s['runtime_sha256']
 for x in r['inputs']:assert sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256']
 pin=primary['source_dependency_manifest'];assert sha256(Path(pin['path']).read_bytes()).hexdigest()==pin['sha256']
 assert len(r['branches'])==len(old['branches'])==156
 changed=[];reviewed=[]
 for a,b in zip(old['branches'],r['branches']):
  assert (a['root_owner'],a['root_method'],a['line'],a['sha256'],a['condition'])==(b['root_owner'],b['root_method'],b['line'],b['sha256'],b['condition'])
  assert b['observed_outcomes']==[] and not b['fusion_search_complete']
  if a['condition_state']!=b['condition_state']:
   assert a['condition_state']=='UNRESOLVED_INPUT_OR_RUNTIME' and b['condition_state'].startswith('WORKER_FROZEN_');changed.append(b['line'])
  if 'worker_source_leg_review' in b:reviewed.append(b['line']);assert not b['worker_source_leg_review']['actual_branch_population']
 assert len(changed)==29 and len(reviewed)==61 and sum(b['condition_state']=='UNRESOLVED_INPUT_OR_RUNTIME' for b in r['branches'])==37
 assert r['scoped_source_bindings']==119 and len(controls['cases'])==13 and len(r['source_default_and_factory_controls'])==13
 fn='vllm_ascend.worker.model_runner_v1:NPUModelRunner.execute_model';catalog=fine['host_logs']['catalog'];s=next(s for s in r['sources'] if s['runtime_path']==catalog[fn]['path']);assert s['runtime_sha256']==catalog[fn]['sha256']
 assert (catalog[fn]['line'],catalog[fn]['end_line'])==(2229,2678)
 assert not any('valid_sampled_token_count' in n or 'correct_spec_decode_token_counts' in n for n in catalog)
 vals=[x['prepare_free_function_partition_ms'][fn] for x in fine['ranks']];assert math.isclose(statistics.mean(vals),price['relaxed_parent_mean_11_cycle_Free_ms'],abs_tol=1e-10)
 assert price['diagnostic_pct']<.165 and price['request_scenarios'][0]['conditional_mean_batch_pct']<.221
 assert len(price['ranks'])==16 and all(x['closed_cycles']==11 for x in price['ranks'])
 for d in [r,controls,price]:assert not d['whole_chain_complete'] and not d['new_NPU_work'] and not d['new_profile'] and not d['service_requests'] and not d['remote_writes']
 assert not r['new_primitive_admissions'] and not r['new_framework_replacement_admissions']
 names=['worker-protocol-source-review.json','worker-protocol-dependencies.json','worker-protocol-aux-dependencies.json','worker-protocol-launch-flags.json','worker-execution-protocol-controls.json','worker-execution-protocol-source-excerpts.json','worker-protocol-gate-source-excerpts.json','worker-position-parent-primary.json','worker-position-parent-price.json','worker-protocol-control-harness-history.json','worker-protocol-frontier.json','worker状态协议与早期CPU坐标交付判断.md']
 out=dict(status='PASS_BOUNDED_WORKER_SOURCE_PROTOCOL_AND_SAVED_PRICE_NOT_NATIVE_ADMISSION',source_pins=24,original_sites=156,new_bindings=29,bound=119,pending=37,worker_source_leg_notes=61,opaque_protocol_cases=13,default_factory_controls=13,every_replacement_bitwise=True,whole_chain_complete=False,new_primitive_admissions=0,new_framework_replacement_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],artifacts=[dict(path=str(ROOT/n),sha256=sha256((ROOT/n).read_bytes()).hexdigest()) for n in names],verifier_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
 (ROOT/'worker-protocol-integrity.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['artifacts','verifier_sha256']}))
if __name__=='__main__':main()
