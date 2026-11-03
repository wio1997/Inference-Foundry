#!/usr/bin/env python3
import json,hashlib,collections,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[2]; src=R/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'; trip=R/'agents/fusion/f08-actual-triples.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(src)=='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
a=json.loads(trip.read_text())['records']; wanted={(r['ep'],r['scope_id'],r['flow_ids'][2]):r for r in a if r['rows']==8180}; assert len(wanted)==640
out=[]
with src.open() as fp:
 for line in fp:
  x=json.loads(line)
  if x.get('kind')!='rank':continue
  for s in x['data']['layers']:
   for g in s['operator_groups']:
    for t in g['tasks']:
     k=(x['data']['ep'],s['layer_scope_id'],t['flow_id'])
     if k not in wanted:continue
     r=wanted[k];c=t['csv']; assert r['stream']==c['Stream ID'] and r['tasks'][2]==c['Task ID'] and r['starts_us'][2]==t['device_start_us']
     out.append(dict(ep=r['ep'],layer=r['layer'],digest=r['digest'],scope=r['scope_id'],duration_us=float(t['trace_duration_us']),csv_duration_us=float(c['Duration(us)']),stream=c['Stream ID'],task=c['Task ID'],start_us=t['device_start_us'],csv_line=t['csv_line'],shape=c['Input Shapes'],output_shape=c['Output Shapes'],dtype=c['Input Data Types'],format=c['Input Formats'],block=c['Block Num'],core=c['Accelerator Core'],aiv_cycles=c['aiv_total_cycles'],aiv_vec_ratio=c['aiv_vec_ratio']))
assert len(out)==640 and len({(x['ep'],x['scope'],x['task']) for x in out})==640
s=sorted(out,key=lambda x:x['duration_us']);short=[x for x in out if x['duration_us']<50];long=[x for x in out if x['duration_us']>200];middle=[x for x in out if 50<=x['duration_us']<=200]
def counts(a,k):return dict(collections.Counter(str(x[k]) for x in a))
def group(a,keys):
 d=collections.defaultdict(list)
 for x in a:d['|'.join(str(x[k]) for k in keys)].append(x['duration_us'])
 return {k:{'n':len(v),'min':min(v),'median':statistics.median(v),'max':max(v)} for k,v in sorted(d.items())}
keys=['shape','output_shape','dtype','format','block','core']
common=[]
for u in short:
 for v in long:
  if all(u[k]==v[k] for k in ['ep','digest']+keys):common.append((u,v))
common.sort(key=lambda x:(x[0]['duration_us'], -x[1]['duration_us']))
report={'source_sha256':sha(src),'triples_sha256':sha(trip),'threshold_us':{'short_lt':50,'long_gt':200},'n':len(out),'short_n':len(short),'middle_n':len(middle),'long_n':len(long),'extremes':{'short':s[0],'long':s[-1]},'metadata_by_class':{name:{k:counts(vals,k) for k in keys+['digest','ep','layer']} for name,vals in [('short',short),('middle',middle),('long',long)]},'same_rank_digest_meta_pair_count':len(common),'same_rank_digest_meta_best_pair':common[0] if common else None,'by_rank_digest':group(out,['ep','digest']),'all_samples_sorted':s,'limitations':['time is profile-on; not production inference speed','task input shapes do not identify slot indices or values, alignment, actual cache stride, cache residency or concurrency','AIV total cycle and vector ratio are profiler derived aggregates, not a direct stall-cause breakdown','same rank/digest is not proof of equal data or dynamic workload']}
(R/'agents/fusion/f08-scatter-variation.json').write_text(json.dumps(report,indent=2)+'\n')
print('counts',len(short),len(middle),len(long),'pair_count',len(common));print('extremes',json.dumps(report['extremes'],indent=2));print('best same rank digest metadata pair',json.dumps(report['same_rank_digest_meta_best_pair'],indent=2)); print('short by rank',counts(short,'ep')); print('short by layer',counts(short,'layer'));print('short digests',counts(short,'digest'));print('long digests',counts(long,'digest'));print('middle',json.dumps(middle[:2],indent=2))
