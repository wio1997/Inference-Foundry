#!/usr/bin/env python3
"""Stream saved rank records and compare strictly matched physical operator groups."""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import statistics

root=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--all-groups',action='store_true',help='also emit every eligible group; default emits compact summary')
args=parser.parse_args()
inp=root/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
hier=root/'diagnostic-20261029/sidecar-hierarchy-20261029.json'
scopefile=root/'diagnostic-20261029/scope-summary-v2.json'
expected='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
expectedhier='e4ffaf3cacaeebddb6beb249d9f48d76a3800dc5f02528cdc2ae46e6c3a8a023'
expectedscope='3cf4a030e5e665a485bb9dccd77d9d8c77308bf425cc1d01cb4e16ba78d7ec6b'
def loadverified(path, sha):
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('input hash mismatch')
    return json.loads(raw)
h=loadverified(hier,expectedhier);scope=loadverified(scopefile,expectedscope)
scope_ranks={x['rank_directory']:x for x in scope['ranks']}
scope_map={};eps=defaultdict(set)
for s in h['layer_samples']:
    if s['role']!='conditional_target':continue
    key=(s['ep'],s['call_id'])
    if key in scope_map:raise ValueError('sidecar duplicate')
    scope_map[key]=(s['layer_index'],s['parent_digest'],s['pid'])
    eps[s['parent_digest']].add(s['ep'])
if len(scope_map)!=1280 or len(eps)!=4 or any(len(x)!=8 for x in eps.values()):raise ValueError('bad hierarchy')
fields=['Input Shapes','Output Shapes','Input Data Types','Output Data Types','Input Formats','Output Formats']
groups=defaultdict(lambda:defaultdict(list));coverage=Counter();rank_audit=[];seen_ep=set();seen_scope=set();seen_kind=Counter()
sourcehash=hashlib.sha256()
def number(value):
    if value is None or isinstance(value,bool):return None
    try:
        x=Decimal(str(value).strip())
        return x if x.is_finite() else None
    except (InvalidOperation,ValueError):return None
with inp.open('rb') as f:
    for line in f:
        sourcehash.update(line)
        x=json.loads(line);kind=x.get('kind');seen_kind[kind]+=1
        if kind=='wrapper':
            if x['hierarchy_input_sha256']!=expectedhier or x['scope_summary_sha256']!=expectedscope:raise ValueError('wrapper mismatch')
            continue
        if kind=='header':
            if x['hierarchy_sha256']!=expectedhier:raise ValueError('header mismatch')
            continue
        if kind=='footer':
            if x['rank_errors'] or x['missing_expected_ep'] or x['unexpected_ep'] or x['rank_dirs_seen']!=16 or x['completed_ranks']!=16:raise ValueError('footer incomplete')
            continue
        if kind!='rank':raise ValueError('unknown line type')
        rank=x['data'];ep=rank['ep'];pid=rank['pid']
        if ep in seen_ep or rank['status']!='completed':raise ValueError('bad or duplicate rank')
        seen_ep.add(ep)
        prof=scope_ranks.get(rank['rank'])
        if not prof or prof.get('trace_status')!='read' or prof['trace_sha256']!=rank['input_sha256']['trace']:raise ValueError('trace provenance mismatch')
        hashes=rank['input_sha256']
        if any(not isinstance(hashes.get(k),str) or len(hashes[k])!=64 for k in ('trace','kernel_csv','metadata')):raise ValueError('missing input hash')
        rank_audit.append({'ep':ep,'rank_record_sha256':hashlib.sha256(line).hexdigest(),
                           'trace_matches_scope_sha':True,'kernel_csv_sha256':hashes['kernel_csv'],'metadata_sha256':hashes['metadata']})
        if rank['counts'].get('target_scope_count')!=80:raise ValueError('incomplete rank scope')
        local_tasks=0
        for layer in rank['layers']:
            cid=layer['layer_scope_id'];index=layer['layer'];digest=layer['scheduler_digest']
            key=(ep,cid)
            if key in seen_scope or scope_map.get(key)!=(index,digest,pid) or layer['trace_scope_count']!=1 or not layer['sidecar_thread_matches_trace']:
                raise ValueError('scope join error')
            seen_scope.add(key);n=0
            for group in layer['operator_groups']:
                tasks=group['tasks']
                if group['task_count']!=len(tasks):raise ValueError('group count mismatch')
                for task in tasks:
                    row=task['csv'];physical=group['physical_shape_dtype_format']
                    if any(row.get(k)!=physical.get(k) for k in fields):raise ValueError('shape descriptor mismatch')
                    if row.get('Name')!=group['operator_name']:raise ValueError('operator name mismatch')
                    shape=json.dumps(physical,sort_keys=True,separators=(',',':'))
                    groupkey=(digest,index,row['Name'],row['Type'],shape,row.get('Accelerator Core'))
                    groups[groupkey][ep].append(row)
                    n+=1;local_tasks+=1
            if n!=layer['matched_task_count']:raise ValueError('layer count mismatch')
        if local_tasks!=rank['counts']['target_tasks_uniquely_matched']:raise ValueError('rank count mismatch')
        coverage['tasks']+=local_tasks
        for k,v in rank['counts'].items():
            if isinstance(v,int):coverage['extraction_'+k]+=v
if sourcehash.hexdigest()!=expected:raise ValueError('all-kernel data hash mismatch')
if seen_kind!=Counter({'wrapper':1,'header':1,'rank':16,'footer':1}) or seen_ep!=set(range(16)) or seen_scope!=set(scope_map) or coverage['tasks']!=53918:raise ValueError('coverage mismatch')
eligible={'aic':[],'aiv':[]}
exclusions={'aic':Counter(),'aiv':Counter()}
task_metric_counts={'aic':Counter(),'aiv':Counter()}
physical_status=Counter()
for key,rankrows in groups.items():
    digest,layer,name,typ,shape,core=key
    shapeobj=json.loads(shape)
    if any(not isinstance(shapeobj.get(field),str) or shapeobj[field].strip().lower() in ('','n/a','na','null','none','unknown') for field in fields):
        physical_status['missing_shape_dtype_or_format']+=1
        for z in exclusions:exclusions[z]['missing_shape_dtype_or_format']+=1
        continue
    expected_ep=eps[digest]; actual=set(rankrows)
    if actual!=expected_ep:
        physical_status['missing_one_or_more_matching_rank_groups']+=1
        for z in exclusions:exclusions[z]['missing_one_or_more_matching_rank_groups']+=1
        continue
    sizes={len(v) for v in rankrows.values()}
    if len(sizes)!=1:
        physical_status['rank_task_count_mismatch']+=1
        for z in exclusions:exclusions[z]['rank_task_count_mismatch']+=1
        continue
    physical_status['full_8_ranks_equal_count']+=1
    records=[r for e in sorted(rankrows) for r in rankrows[e]]
    for engine in ('aic','aiv'):
        cykey=engine+'_total_cycles';rkey=engine+('_mac_ratio' if engine=='aic' else '_vec_ratio')
        reasons=set();num=Decimal(0);den=Decimal(0);values=[]
        for row in records:
            cy=number(row.get(cykey));ratio=number(row.get(rkey))
            if cy is None:reasons.add('cycles_missing_or_invalid');task_metric_counts[engine]['cycles_missing_or_invalid']+=1
            elif cy==0:reasons.add('cycles_zero_reported');task_metric_counts[engine]['cycles_zero_reported']+=1
            elif cy<0:reasons.add('cycles_negative');task_metric_counts[engine]['cycles_negative']+=1
            else:task_metric_counts[engine]['cycles_positive']+=1
            if ratio is None or not 0<=ratio<=1:reasons.add('ratio_missing_or_invalid');task_metric_counts[engine]['ratio_missing_or_invalid']+=1
            elif ratio==0:task_metric_counts[engine]['ratio_zero_displayed']+=1
            if cy is not None and cy>0 and ratio is not None and 0<=ratio<=1:
                num+=cy*ratio;den+=cy;values.append(ratio)
        if reasons:
            for r in reasons:exclusions[engine][r]+=1
            exclusions[engine]['any_metric_invalid_or_nonpositive_cycle']+=1
            continue
        eligible[engine].append({'digest':digest,'layer':layer,'operator_name':name,'csv_type':typ,
               'core':core,'physical_shape_dtype_format':shapeobj,'per_rank_task_count':next(iter(sizes)),
               'task_count':len(records),'eps':sorted(rankrows),'weighted_displayed_ratio':str(num/den),
               'min_displayed_ratio':str(min(values)),'max_displayed_ratio':str(max(values))})

def minima(entries):
    results=[]
    for digest in sorted(eps):
        by_name=defaultdict(list)
        for g in entries:
            if g['digest']==digest:by_name[(g['operator_name'],g['csv_type'])].append(g)
        best=[min(v,key=lambda x:(Decimal(x['weighted_displayed_ratio']),x['layer'])) for v in by_name.values()]
        best.sort(key=lambda x:(Decimal(x['weighted_displayed_ratio']),x['operator_name'],x['layer']))
        results.append({'digest':digest,'eps':sorted(eps[digest]),'distinct_operators_with_eligible_groups':len(best),'lowest_operators':best[:10]})
    return results
paired=defaultdict(lambda:defaultdict(list))
for g in eligible['aic']:
    if (g['operator_name'],g['csv_type']) in (('DispatchFFNCombine','DispatchFFNCombine'),('SparseFlashMla','SparseFlashMla')):
        paired[(g['digest'],g['layer'])][g['operator_name']].append(g)
pair_groups=[];pair_exclusions=Counter()
for digest in sorted(eps):
    for layer in range(40):
        p=paired.get((digest,layer),{})
        a=p.get('DispatchFFNCombine',[]);b=p.get('SparseFlashMla',[])
        if len(a)!=1 or len(b)!=1:
            pair_exclusions['missing_or_ambiguous_eligible_operator_group']+=1
            if len(a)==0:pair_exclusions['mc2_missing_eligible']+=1
            if len(b)==0:pair_exclusions['sparse_mla_missing_eligible']+=1
            if len(a)>1:pair_exclusions['mc2_multiple_eligible']+=1
            if len(b)>1:pair_exclusions['sparse_mla_multiple_eligible']+=1
            continue
        if a[0]['eps']!=b[0]['eps']:pair_exclusions['ep_set_mismatch']+=1;continue
        pair_groups.append({'digest':digest,'layer':layer,'eps':a[0]['eps'],
                            'mc2_weighted_displayed_ratio':a[0]['weighted_displayed_ratio'],
                            'sparse_mla_weighted_displayed_ratio':b[0]['weighted_displayed_ratio']})
pairs=[]
for digest in sorted(eps):
    rows=[p for p in pair_groups if p['digest']==digest]
    pairs.append({'digest':digest,'paired_layer_groups':len(rows),
                  'mc2_median_weighted_displayed_ratio':str(statistics.median(Decimal(x['mc2_weighted_displayed_ratio']) for x in rows)) if rows else None,
                  'sparse_mla_median_weighted_displayed_ratio':str(statistics.median(Decimal(x['sparse_mla_weighted_displayed_ratio']) for x in rows)) if rows else None,
                  'layers':[x['layer'] for x in rows]})
out={'source_sha256':sourcehash.hexdigest(),'scope_sha256':expectedscope,'hierarchy_sha256':expectedhier,
     'jsonl_kind_counts':dict(seen_kind),'footer':'validated_no_rank_errors','ranks':rank_audit,'scope_count':len(seen_scope),
     'coverage':dict(coverage),'physical_group_count':len(groups),'physical_group_status':dict(physical_status),
     'aic_eligible_group_count':len(eligible['aic']),'aic_exclusions':dict(exclusions['aic']),'aic_task_metrics_in_physically_complete_groups':dict(task_metric_counts['aic']),
     'aiv_eligible_group_count':len(eligible['aiv']),'aiv_exclusions':dict(exclusions['aiv']),'aiv_task_metrics_in_physically_complete_groups':dict(task_metric_counts['aiv']),
     'aic_lowest_operators_by_digest':minima(eligible['aic']),'aiv_lowest_operators_by_digest':minima(eligible['aiv']),
     'aic_mc2_vs_sparse_mla_paired_by_digest':pairs,'paired_layer_groups':pair_groups,'pair_exclusions':dict(pair_exclusions),
     'limits':['Same recorded physical shape/dtype/format does not prove equal effective work.',
               'Rank shape/count matching is within each operator group, not between unlike operators.',
               'Zeros are displayed ratios, not evidence of inactivity; precision/rounding semantics unknown.',
               'Separate AIC and AIV rankings: different engines and denominators.',
               'Ratios and sums are task-level report arithmetic, not whole-device utilization, root cause, or online latency.',
               'Lowest is within captured eligible groups, not a global model minimum.',
               'Excluded flow joins and absent tasks remain unknown; source CSV hashes only record observed provenance, not prior immutability.']}
for reason in ('missing_one_or_more_matching_rank_groups','rank_task_count_mismatch','missing_shape_dtype_or_format','full_8_ranks_equal_count'):
    out['physical_group_status'].setdefault(reason,0)
if args.all_groups:
    out['all_aic_eligible_groups']=eligible['aic']
    out['all_aiv_eligible_groups']=eligible['aiv']
print(json.dumps(out,ensure_ascii=False,indent=2))
