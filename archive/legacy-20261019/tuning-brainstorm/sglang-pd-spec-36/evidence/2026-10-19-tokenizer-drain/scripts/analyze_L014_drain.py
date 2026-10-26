#!/usr/bin/env python3
"""Paired descriptive CPU analysis; preserve every sample and arrival."""
import hashlib,json,math,pathlib,random,statistics
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L014-analysis';O.mkdir(parents=True,exist_ok=False);protocol_path=R/'specs/L014-protocol.json';protocol=json.loads(protocol_path.read_text());reps=4000;seed=protocol['random_seed'];sources=[];cases=[];first=None
def effect(values,seed):
 logs=[math.log(v) for v in values];rng=random.Random(seed);samples=sorted(math.exp(statistics.mean(rng.choices(logs,k=len(logs)))) for _ in range(reps))
 return {'n':len(values),'geomean_A_over_other':math.exp(statistics.mean(logs)),'bootstrap_percentile95':[samples[int(.025*reps)],samples[int(.975*reps)-1]],'raw_paired_ratios':values}
for ci,tag in enumerate(['L014-R001','L014-R002','L014-R003']):
 path=R/'runs'/(tag+'-tokenizer-cpu')/'result.json';d=json.loads(path.read_text());assert d['probe_status']=='PASS' and len(d['samples'])==protocol['paired_blocks']*4
 if first is None:first=d
 else:assert all(d[k]==first[k] for k in ['source_hashes','model_hashes','protocol_sha256','candidate_sha256'])
 assert d['protocol_sha256']==hashlib.sha256(protocol_path.read_bytes()).hexdigest() and d['candidate_sha256']==protocol['candidate_sha256'];n=len(d['fixture_metadata']);expected=[x['ids_sha256'] for x in d['fixture_metadata']]
 for x in d['samples']:
  assert x['values_match'] and x['ids_sha256']==expected and len(x['request_latency_ns'])==n
  assert all(0<=a<=c<=x['wall_ns'] and c-a==lat for a,c,lat in zip(x['arrival_offsets_ns'],x['completion_offsets_ns'],x['request_latency_ns']))
  assert x['requested_delays_s']==protocol['arrival_delays_s'][d['case_key']]
 item={'run':tag,'case_key':d['case_key'],'sample_count':len(d['samples']),'fixture_metadata':d['fixture_metadata'],'metrics':{}}
 def value(x,metric):return x[metric] if not metric.startswith('request_latency_') else x['request_latency_ns'][int(metric.rsplit('_',1)[1])]
 metrics=['wall_ns','process_cpu_ns',*[f'request_latency_{i}' for i in range(n)]]
 for mi,metric in enumerate(metrics):
  stats={};medians={}
  for pair,other in [('AA','A1'),('AB','B')]:
   left=[];right=[]
   for block in range(protocol['paired_blocks']):
    rows={x['actor']:x for x in d['samples'] if x['block']==block and x['pair']==pair};assert set(rows)=={'A0',other};left.append(value(rows['A0'],metric));right.append(value(rows[other],metric))
   assert all(v>0 for v in left+right);stats[pair]=effect([a/b for a,b in zip(left,right)],seed+ci*100+mi*10+(0 if pair=='AA' else 1));medians[pair]={'A0':statistics.median(left),'other':statistics.median(right),'other_actor':other}
  aa=stats['AA']['bootstrap_percentile95'];ab=stats['AB']['bootstrap_percentile95'];item['metrics'][metric]={'paired':stats,'medians_ns':medians,'AB_interval_separated_from_AA_interval':ab[1]<aa[0] or ab[0]>aa[1]}
 counters={};arrivals={}
 for pair,actors in [('AA',['A0','A1']),('AB',['A0','B'])]:
  counters[pair]={};arrivals[pair]={}
  for actor in actors:
   rows=[x for x in d['samples'] if x['pair']==pair and x['actor']==actor];available=all(not x['cgroup_before'].get('unavailable') for x in rows);counters[pair][actor]={'available':available,'samples_throttled':sum(x['cgroup_delta'].get('nr_throttled',0)>0 for x in rows),'throttled_usec_total':sum(x['cgroup_delta'].get('throttled_usec',0) for x in rows),'throttled_time_ns_total':sum(x['cgroup_delta']['throttled_time'] for x in rows) if all('throttled_time' in x['cgroup_delta'] for x in rows) else None}
   arrivals[pair][actor]=[{'request':i,'requested_delay_s':protocol['arrival_delays_s'][d['case_key']][i],'actual_offset_ns':{'median':statistics.median(x['arrival_offsets_ns'][i] for x in rows),'min':min(x['arrival_offsets_ns'][i] for x in rows),'max':max(x['arrival_offsets_ns'][i] for x in rows)}} for i in range(n)]
 groups=[v for pair in counters.values() for v in pair.values()];clean=all(v['available'] and not v['samples_throttled'] for v in groups);wall=item['metrics']['wall_ns'];separated=wall['AB_interval_separated_from_AA_interval'];item.update({'cgroup':counters,'all_intervals_without_throttle':clean,'actual_arrivals':arrivals,'wall_conclusion':'B faster beyond measured AA' if clean and separated and wall['paired']['AB']['geomean_A_over_other']>1 else 'INCONCLUSIVE: AA overlap, throttle, or no positive wall effect','scope':'Observed synthetic generator only; requested arrival schedule common, actual arrival jitter retained. Per-request metric is CPU encode completion latency, not model TTFT.'});cases.append(item);sources.append({'path':str(path.relative_to(R)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
out={'analysis_script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),'candidate_sha256':protocol['candidate_sha256'],'bootstrap':{'resamples':reps,'unit':'paired block','method':'paired log ratios, bootstrap percentile95, no row filtering','seed_base':seed},'inputs':sources,'cases':cases,'scope':'CPU-only class candidate; no runtime service, PD/MTP/NPU, p99, or physical-peak claim.'};(O/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps([{'case':c['case_key'],'wall':c['metrics']['wall_ns']['paired']['AB'],'conclusion':c['wall_conclusion']} for c in cases]))
