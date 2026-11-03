"""Derive comparable and explicitly separate performance cohorts from frozen evidence."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
paths = {
    'target_types': 'agents/diagnosis/target-fullspan-types.json',
    'engram_candidates': 'agents/diagnosis/engram-target-span-recheck.json',
    'draft_calls': 'draft-mc2-analysis/strict_draft_calls.csv',
    'draft_target_windows': 'agents/diagnosis/draft-target-operator-decomposition.json',
}
inputs = {k: hashlib.sha256((ROOT/v).read_bytes()).hexdigest() for k,v in paths.items()}
target = json.loads((ROOT/paths['target_types']).read_text())
rows = [x for x in target['rows'] if x['run'] == 2]
assert len(rows) == 16 and len({x['rank'] for x in rows}) == 16
selected = ['DispatchFFNCombine','QuantBatchMatmulV3','HcPre','VllmAscendQuantLightningIndexerV2','TransposeBatchMatMul','MatMulV2','SparseFlashMla','HcPost','ScatterNdUpdateSk']
assert all(x['cross_type_overlap_excess_ns'] == 0 for x in rows)
table = []
for name in selected:
    table.append({'type':name,'records_per_rank':sorted({x['type_event_count'][name] for x in rows}),
                  'median_union_us':median(x['type_union_ns'][name]/1000 for x in rows),
                  'median_rank_fraction':median(x['type_union_ns'][name]/x['span_ns'] for x in rows)})
eng = json.loads((ROOT/paths['engram_candidates']).read_text())
eng_by_rank = {x['rank']:x for x in eng['samples'] if x['run'] == 2}
anonymous = []
for i,x in enumerate(rows):
    assert eng_by_rank[x['rank']]['target_span_ns'] == x['span_ns']
    anonymous.append({'sample':i+1,'span_ns':x['span_ns'],
                      'type_union_ns':x['type_union_ns'],'type_event_count':x['type_event_count'],
                      'all_types_union_ns':x['all_types_union_ns'],
                      'engram_compatible_pair_ns':eng_by_rank[x['rank']]['two_matmul_duration_ns']})
draft = list(csv.DictReader((ROOT/paths['draft_calls']).open()))
assert len(draft) == 162
draft_anon = [{'window_us':float(x['bracket_duration_us']),'moe_us':float(x['mc2_duration_us'])} for x in draft]
windows = json.loads((ROOT/paths['draft_target_windows']).read_text())['windows']
target0 = [(x['end']-x['start'])/1000 for x in windows if x['kind'] != 'draft']
def stats(xs):
    return {'n':len(xs),'min':min(xs),'median':median(xs),'max':max(xs)}
out = {'status':'EXISTING_PROFILE_COHORTS_NOT_END_TO_END', 'input_sha256':inputs,
       'target_mainstream':{'profile_run':2,'rank_count':16,'layers':40,
                           'span_us':stats([x['span_ns']/1000 for x in rows]),'types':table,
                           'exported_union_fraction_median':median(x['all_types_union_ns']/x['span_ns'] for x in rows),
                           'anonymous_samples':anonymous},
       'engram_shape_compatible_only':{'run2_pair_us':stats([x['engram_compatible_pair_ns']/1000 for x in anonymous]),
                                      'run2_pair_fraction_median':median(x['engram_compatible_pair_ns']/x['span_ns'] for x in anonymous),
                                      'included_in_MatMulV2_label':True,
                                      'logical_MAC_per_selected_task':3*6144*25600,
                                      'conditional_BF16_weight_bytes':6144*25600*2,
                                      'nominal_compute_only_us':3*6144*25600/(24*1800000000*4096)*1e6},
       'draft_separate_cohort':{'rank_count':len({x['rank'] for x in draft}),
                                'block_window_us':stats([x['window_us'] for x in draft_anon]),
                                'moe_task_us':stats([x['moe_us'] for x in draft_anon]),
                                'sum_moe_us':sum(x['moe_us'] for x in draft_anon),
                                'sum_window_us':sum(x['window_us'] for x in draft_anon),
                                'sum_moe_over_sum_window':sum(x['moe_us'] for x in draft_anon)/sum(x['window_us'] for x in draft_anon),
                                'median_per_window_moe_fraction':median(x['moe_us']/x['window_us'] for x in draft_anon),
                                'target0_separate_window_us':stats(target0),'anonymous_samples':draft_anon},
       'limits':['Mainstream target span excludes other streams and full requests; exported coverage is not NPU utilization.',
                 'Type labels do not uniquely identify all modules; Engram-compatible matmul rows are already included in MatMulV2.',
                 'Draft cohort is 162 strictly selected windows from 15 ranks, pooled across blocks, with different work from target.',
                 'All timings are profiled; no production critical-path shares or speedup are inferred.']}
p=ROOT/'publication-draft/evidence-summary.json'
j=json.loads(p.read_text());j['stage_consolidation']['whole_model_overview']=out
p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n')
small={k:v for k,v in out.items() if k not in ['target_mainstream','draft_separate_cohort']}
small['target_mainstream']={k:v for k,v in out['target_mainstream'].items() if k!='anonymous_samples'}
small['draft_separate_cohort']={k:v for k,v in out['draft_separate_cohort'].items() if k!='anonymous_samples'}
print(json.dumps(small,ensure_ascii=False,indent=2))
