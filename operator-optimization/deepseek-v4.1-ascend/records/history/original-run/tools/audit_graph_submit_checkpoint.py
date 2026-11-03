"""Verify the newly integrated export chain and synchronize the checkpoint."""
from pathlib import Path
import json,subprocess,sys
T=Path(__file__).resolve().parents[1]
def read(n):return json.loads((T/n).read_text())
e=read('evidence/graph-submit-diagnosis.json');chain=read('model-execute-wait-chain-v2/summary.json')
assert e['captured_layer0_calls']==104 and e['unique_candidate_within_10ms']==99 and e['no_candidate_within_10ms']==5
assert chain['calls_checked']==chain['one_execute_one_wait_count']==chain['also_unique_wait_contains_source_validated_layer_count']==99
assert chain['name_counts']=={'MODEL_EXECUTE':99,'MODEL_WAIT_COMPLETE':99}
assert [x['ordinal'] for x in e['paired_interior_forward_groups']]==[1,6,7]
assert all(x['api_dp0_minus_dp1_us']>0 for x in e['paired_interior_forward_groups'])
records=read('records.json');cause=next(x for x in records['module_causes'] if x['id']=='C_GRAPH_SUBMIT_SKEW')
assert cause['status']=='HYPOTHESIS_WITH_STRONGER_EXPORTED_ASSOCIATION' and cause['end_to_end_impact'] is None
report=(T/'REPORT.md').read_text();assert '99/104' in report and '时钟误差' in report and '不能断言调度器' in report
assert len(json.dumps(read('profile-summary.json')).encode())<32768
(T/'evidence/REVIEW-GRAPH-SUBMIT.json').write_text(json.dumps({'status':'PASS_WITH_EXPLICIT_CLOCK_AND_CAUSAL_GAPS',
 'checks':['99 exact CANN-to-model task chains','5 unmatched retained','3 paired interior groups','Cause recorded as hypothesis, no end-to-end time invented',
           'Report keeps clock and request identity limitations'],
 'does_not_prove_cause_or_physical_ceiling':True},indent=2)+'\n')
r=subprocess.run([sys.executable,str(T/'tools/audit_partial_progress.py')],capture_output=True,text=True,timeout=360)
assert r.returncode==0,r.stderr
print(json.dumps({'graph_chain_review':'PASS_WITH_EXPLICIT_CLOCK_AND_CAUSAL_GAPS','partial_audit_and_sync':json.loads(r.stdout)},indent=2))
