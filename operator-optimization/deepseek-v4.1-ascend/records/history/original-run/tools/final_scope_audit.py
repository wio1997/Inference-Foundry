# -*- coding: utf-8 -*-
"""Requirement audit of partial, explicitly bounded diagnostic delivery."""
from pathlib import Path
from fractions import Fraction
from collections import Counter
import ast,datetime,hashlib,json
T=Path(__file__).resolve().parents[1];Q=Path('/Users/wio/work/173-sglang-matched-172-benchmark')
def read(p):return json.loads((T/p).read_text())
def write(p,x):(T/p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256((T/p).read_bytes()).hexdigest()
for f in T.rglob('*.json'):json.loads(f.read_text())
for f in (T/'tools').glob('*.py'):ast.parse(f.read_text())
contract=read('contract.json');assert sha(contract['graph_scope']['original_graph_path'])==contract['graph_scope']['original_graph_sha256']
r=read('records.json');source_count=0
for s in r['sources']:
 if s.get('snapshot_path') and s.get('sha256'):assert sha(s['snapshot_path'])==s['sha256'];source_count+=1
for c in r['fusion_candidates']:
 if c.get('candidate_graph_path') and c.get('candidate_graph_sha256'):assert sha(c['candidate_graph_path'])==c['candidate_graph_sha256'],c['id']
assert len(r['fusion_candidates'])==31 and len(r['candidate_relationships'])>0
scan=read('fusion-scan.json');assert scan['status']=='PAUSED_WITH_GAPS' and not scan['whole_model_exhaustion_proven'] and scan['stop_reason']['overall_task_paused'] is False
assert 'edge 643' in ' '.join(scan['uncovered_items']);assert (T/scan['partial_metadata_graph']).exists()
replay=json.loads((Q/'operator-ceiling169-v2-069-fusion-replay.output.txt').read_text());assert replay['status']=='PASS_BOUNDED_STRUCTURAL_REPLAY_ONLY' and replay['g1_cross_sets']==22802
coverage=read('coverage/summary.json');assert coverage['node_rows']==623 and coverage['edge_rows']==1029
m=read('module-analysis.json');targets=[x for x in m['modules'] if x['module_id'].startswith('target.layers.')];assert len(targets)==40
for x in targets:
 v=x['metrics'][0];assert x['call_count']==48 and x['rank_count']==16 and abs(float(Fraction(v['numerator'])/Fraction(v['denominator']))-v['value'])<1e-12
low=min(targets,key=lambda x:x['metrics'][0]['value']);assert low['module_id']=='target.layers.0.forward'
assert m['rankings'][0]['lowest_module_ids']==['target.layers.0.forward']
assert len(m['performance_comparisons'])==76
sensitivity=read('evidence/conditional-ratio-sensitivity.json');assert len(sensitivity['objects'])==7 and sensitivity['formal_same_scope_ratio']=='UNKNOWN'
for x in m['performance_comparisons']:
 if x.get('formal_ratio_status')=='UNKNOWN_SCOPE_MEMBERSHIP_NOT_PROVEN':assert x['measured_over_ceiling_ratio'] is None and x['time_gap_ratio'] is None
for x in m['performance_comparisons']:
 if x.get('bound_status')=='CONDITIONAL_CEILING' and x.get('calculation_path'):
  v=read(x['calculation_path']);assert Fraction(v['layers'][0]['time_lower_from_known_terms']['exact_fraction'])==Fraction(x['time_lower_L']['exact_fraction'])
  assert Fraction(x['throughput_upper_U']['exact_fraction'])==Fraction(x['effective_work_Q'])/Fraction(x['time_lower_L']['exact_fraction'])
for cid in ['C_ARRIVAL_IMBALANCE','C_HBM_OR_SMALL_M','C_OBSERVER_OR_MAPPING','C_LAYER0_SMALL_M','C_LAYER0_MEMORY','C_LAYER0_EXPERT_SKEW','C_LAYER0_PATH_OR_MEASUREMENT','C_GRAPH_SUBMIT_SKEW']:
 x=next(y for y in r['module_causes'] if y['id']==cid)
 assert x['supporting_evidence'] and x['counter_evidence'] and x['missing_evidence'] and x.get('end_to_end_impact') is None
raw=read('evidence/used-raw-source-sha256.json');assert raw['count']==97 and len(raw['records'])==97
raw['host_visible_trace_root']='/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces';raw['path_mapping']='stage container /work maps to host /data/tiankuan/wio/main';write('evidence/used-raw-source-sha256.json',raw)
idx=read('profile-index.json');assert idx['used_raw_hashes']['indexed_file_matches']==17 and sum(bool(x.get('sha256')) for x in idx['files'])>=33
assert (T/'bounds-input.json').exists() and (T/'profiling-plan.json').exists() and (T/'profile-summary.json').stat().st_size<32768
q=read('evidence/q-projection-16rank-diagnosis.json');q['formal_same_scope_ratio_status']='UNKNOWN_SCOPE_MEMBERSHIP_NOT_PROVEN';q['conditional_observed_median_over_ideal_is_sensitivity_only']=True;write('evidence/q-projection-16rank-diagnosis.json',q)
p=T/'REPORT.md';s=p.read_text();s=s.replace('中位样本与该条件模型的比值约 118.1。它是采样算子与一个宽松条件模型的差距，不是可达到的加速比，也不是框架整体上界。','若样本确属该条件模型所限定的域，中位时间与条件下界的算术比约 118.1；同域尚未证实，正式比值为 UNKNOWN。这不是可达到的加速比，也不是框架整体上界。');p.write_text(s)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
audit={'reviewer':'GPT primary controller, with independent read-only review and locally verified ZCode execution','timestamp':now,'status':'PARTIAL_DELIVERY_ACCEPTED_WITH_EXPLICIT_UNKNOWNS','requirements':[
 {'id':'P1','result':'PASS_WITH_GAPS','evidence':['contract.json','graphs/G0.json','graphs/G1.json'],'remaining':'Primitive, alias, and dynamic branch mapping incomplete.'},
 {'id':'P2','result':'PASS_WITH_GAPS','evidence':['fusion-scan.json','coverage/summary.json','candidates/','records.json'],'remaining':'Structural lists and local source-boundary candidates are not complete semantic coverage; F31 metadata partial.'},
 {'id':'P3','result':'PASS_RECORDED_GAP_STOP','evidence':['fusion-scan.json','tools/verify_fusion_enumeration_readonly.py','zcode/operator-ceiling169-v2-069-fusion-replay.output.txt'],'remaining':'NEXT_INVOCATION_RAW edge 643 excluded from in-step enumeration; G1/G5 not merged; no global exhaustion claim.'},
 {'id':'P4','result':'PASS_SCOPED_RANKING','evidence':['module-analysis.json','profile-summary.json','captured-layer-attribution/','eager-target-attribution-v2/','engram-kernel-profile/'],'remaining':'Whole-framework minimum, physical NPU busy rate, 720 eager attention paths and complete Engram ownership UNKNOWN.'},
 {'id':'P5','result':'PASS_SYMBOLIC_OR_CONDITIONAL_WITH_UNKNOWNS','evidence':['bounds-input.json','bounds/','calculations/','evidence/IDEAL-MODEL.md','evidence/conditional-ratio-sensitivity.json'],'remaining':'Seven matched-scope formal ratios, hard capacity and unconditional numeric throughput ceiling UNKNOWN; alternative-domain arithmetic only as sensitivity.'},
 {'id':'P6','result':'PASS_HYPOTHESES_WITH_UNKNOWNS','evidence':['records.json module_causes','evidence/captured-layer0-engine-diagnosis.json','evidence/graph-submit-diagnosis.json'],'remaining':'MC2 internal causal wait, communication exposure, scheduling cause, expert distribution and end-to-end impact UNKNOWN.'},
 {'id':'P7','result':'PASS_PARTIAL_PACKAGE_READY','evidence':['REPORT.md','profile-index.json','evidence/used-raw-source-sha256.json','evidence/TRACE-EXPORT-SHA256.json'],'remaining':'Unreferenced raw export hashes not all captured; no claimed global graph/physical ceiling.'}],
 'constraint_reviews':[{'constraint':'Standard independent dense INT8 work for Wq_b','result':'CONDITIONAL','reason':'MNK valid only in declared standard full-output no-reuse domain; measured sample membership not established.'},{'constraint':'353.8944 TMAC/s per logical device at 1800 MHz','result':'CONDITIONAL','reason':'Profiler metadata and official model provide estimator, not proven 9382 hard frequency/throughput cap.'},{'constraint':'Memory, communication, synchronization lower bounds','result':'UNKNOWN','reason':'Required bytes, hard capacities, aliasing and causal dependency not established.'},{'constraint':'Overlapping resources and block pipeline','result':'PASS_FOR_RELAXATION_ONLY','reason':'Ideal model uses max of separately valid resource lower bounds, not unsupported sums.'},{'constraint':'Reported numerical ratio','result':'UNKNOWN','reason':'Observed run membership in narrow domain not proven; numbers retained only as conditional arithmetic sensitivity.'}],
 'integrity':{'versioned_source_records_sha_checked':source_count,'candidate_graph_sha_checked':len(r['fusion_candidates']),'module_proxy_ratios_reconstructed':40,'referenced_raw_hash_records':97,'replayed_cross_module_sets':22802},'not_proven':['global fusion exhaustion','whole-framework lowest utilization module','causal communication waiting','unconditional 910C numeric throughput ceiling'],'no_new_npu_work':True,'no_target_code_changes':True}
write('evidence/final-scope-review.json',audit)
st=read('STATE.json');st.update(state='PARTIAL_DELIVERY_READY',current_phase='P7',last_update=now,checkpoint_status='INDEPENDENTLY_AUDITED_DELIVERABLE_WITH_GAPS',next_read_only_action='Preserve explicit UNKNOWN conditions and deliver partial diagnostic package; no implementation or tuning.');write('STATE.json',st)
print(json.dumps({'status':audit['status'],'requirements':len(audit['requirements']),'candidate_graphs':len(r['fusion_candidates']),'sources':source_count,'formal_ratios':'UNKNOWN'}))
