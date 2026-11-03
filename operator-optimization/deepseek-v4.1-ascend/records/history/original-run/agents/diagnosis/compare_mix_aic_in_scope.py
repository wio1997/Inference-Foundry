#!/usr/bin/env python3
"""Read-only comparison of matched MIX_AIC tasks in each retained target scope.
Recorded task-level ratio proxies only; not module or whole-device utilization.
"""
import collections, hashlib, json, math, pathlib, statistics
root=pathlib.Path(__file__).resolve().parents[2]
p=root/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
rows=[]; ranks=[]; block_counts=collections.defaultdict(collections.Counter)
for line in p.open():
 o=json.loads(line)
 if o.get('kind')!='rank': continue
 d=o['data']; assert d['status']=='completed'; ranks.append(d['ep'])
 for scope in d['layers']:
  groups=collections.defaultdict(list)
  for g in scope['operator_groups']:
   for task in g['tasks']:
    csv=task['csv']
    if csv['Accelerator Core']=='MIX_AIC':
     groups[(csv['Name'],csv['Type'])].append(csv)
     block_counts[csv['Name']][(csv.get('Block Num'),csv.get('Mix Block Num'))]+=1
  for (name,typ),tasks in groups.items():
   entry={'ep':d['ep'],'layer':scope['layer'],'digest':scope['scheduler_digest'],'name':name,'type':typ,'count':len(tasks)}
   for engine in ('aic','aiv'):
    ratio='aic_mac_ratio' if engine=='aic' else 'aiv_vec_ratio'
    try:
     c=[float(t[engine+'_total_cycles']) for t in tasks]; r=[float(t[ratio]) for t in tasks]
     assert all(math.isfinite(v) and v>0 for v in c)
     assert all(math.isfinite(v) and 0<=v<=1 for v in r)
     entry[engine]=sum(x*y for x,y in zip(c,r))/sum(c)
    except (ValueError,TypeError,AssertionError,KeyError): entry[engine]=None
   rows.append(entry)
assert len(set(ranks))==len(ranks)==16
by_scope=collections.defaultdict(dict)
for r in rows:
 key=(r['ep'],r['digest'],r['layer']); k=(r['name'],r['type']); assert k not in by_scope[key];by_scope[key][k]=r
assert len(by_scope)==1280
base=('DispatchFFNCombine','DispatchFFNCombine')
assert all(base in v for v in by_scope.values())
others=sorted({k for v in by_scope.values() for k in v if k!=base})
result={'input_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'retained_scopes':len(by_scope),'ranks':len(ranks),'block_fields_by_name':{name:[{'Block Num':k[0],'Mix Block Num':k[1],'tasks':n} for k,n in sorted(counts.items())] for name,counts in sorted(block_counts.items())},'comparisons':{},'limits':['task-cycle-weighted exported ratios, not wallclock or device utilization','within same rank and layer scope; different operators and work; selection is not all device tasks','cross-stream overlap and intra-task units are not resolved','neither MAC nor vector ratios describe all computation or waiting','historical exporter formula and zero rounding unverified']}
for op in others:
 detail={}
 for engine in ('aic','aiv'):
  pairs=[]; invalid=0; both=0; by_digest=collections.defaultdict(lambda: {'n':0,'mc2_lower':0,'equal':0,'mc2_higher':0})
  for scope_key,v in by_scope.items():
   if op not in v: continue
   both+=1;a=v[base][engine];b=v[op][engine]
   if a is None or b is None:invalid+=1;continue
   pairs.append((a,b)); bucket=by_digest[scope_key[1]]; bucket['n']+=1; bucket['mc2_lower']+=a<b; bucket['equal']+=a==b; bucket['mc2_higher']+=a>b
  detail[engine]={'by_digest':dict(by_digest),'scopes_both_present':both,'eligible_scopes':len(pairs),'invalid_scopes':invalid,'mc2_lower':sum(a<b for a,b in pairs),'equal':sum(a==b for a,b in pairs),'mc2_higher':sum(a>b for a,b in pairs),'mc2_median_ratio':statistics.median(a for a,b in pairs) if pairs else None,'other_median_ratio':statistics.median(b for a,b in pairs) if pairs else None,'paired_difference_median':statistics.median(a-b for a,b in pairs) if pairs else None}
 result['comparisons'][op[0]+' | '+op[1]]=detail
out=pathlib.Path(__file__).with_name('mix-aic-in-scope-comparison.json')
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
