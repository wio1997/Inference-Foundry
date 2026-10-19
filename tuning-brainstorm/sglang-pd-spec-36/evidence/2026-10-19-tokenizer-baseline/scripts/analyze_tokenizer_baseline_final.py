#!/usr/bin/env python3
"""Descriptive paired analysis of immutable per-case CPU measurements."""
import hashlib,json,math,pathlib,random,statistics
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L011-analysis-final';O.mkdir(parents=True,exist_ok=False);protocol=json.loads((R/'specs/L011-protocol-v2.json').read_text());seed=protocol['random_seed'];reps=4000
def effect(values,seed):
 logs=[math.log(x) for x in values];rng=random.Random(seed);boot=sorted(math.exp(statistics.mean(rng.choices(logs,k=len(logs)))) for _ in range(reps))
 return {'n':len(values),'geomean_A_over_other':math.exp(statistics.mean(logs)),'bootstrap_percentile95':[boot[int(.025*reps)],boot[int(.975*reps)-1]],'raw_paired_ratios':values}
summary=[];sources=[];first=None
for case_index,tag in enumerate(['L011-R003','L011-R004']):
 p=R/'runs'/(tag+'-tokenizer-cpu')/'result.json';d=json.loads(p.read_text());assert d['probe_status']=='PASS' and len(d['samples'])==protocol['paired_blocks']*4
 if first is None:first=d
 else:assert d['source_hashes']==first['source_hashes'] and d['model_hashes']==first['model_hashes'] and d['protocol_sha256']==first['protocol_sha256']
 expected=[x['ids_sha256'] for x in d['fixture_metadata']];assert all(x['values_match'] and x['ids_sha256']==expected for x in d['samples'])
 item={'case_key':d['case_key'],'run':tag,'requests':len(d['fixture_metadata']),'input_tokens':[x['tokens'] for x in d['fixture_metadata']],'sample_count':len(d['samples']),'metrics':{}}
 for metric in ['wall_ns','process_cpu_ns']:
  ratios={};medians={}
  for pair,denominator in [('AA','A1'),('AB','B')]:
   paired=[];left=[];right=[]
   for block in range(protocol['paired_blocks']):
    rows={x['actor']:x for x in d['samples'] if x['block']==block and x['pair']==pair};assert set(rows)=={'A0',denominator}
    a=rows['A0'][metric];b=rows[denominator][metric];assert a>0 and b>0;left.append(a);right.append(b);paired.append(a/b)
   ratios[pair]=effect(paired,seed+case_index*100+(0 if pair=='AA' else 10)+(0 if metric=='wall_ns' else 20));medians[pair]={'A0':statistics.median(left),'other':statistics.median(right),'other_actor':denominator}
  aa=ratios['AA']['bootstrap_percentile95'];ab=ratios['AB']['bootstrap_percentile95'];separated=ab[1]<aa[0] or ab[0]>aa[1]
  item['metrics'][metric]={'paired':ratios,'medians_ns':medians,'AB_interval_separated_from_AA_interval':separated,'interpretation':'A/other>1 means other faster; intervals descriptive, conditional on fixed CPU fixture.'}
 throttle={}
 for pair in ['AA','AB']:
  throttle[pair]={}
  for actor in (['A0','A1'] if pair=='AA' else ['A0','B']):
   rows=[x for x in d['samples'] if x['pair']==pair and x['actor']==actor];throttle[pair][actor]={'samples':len(rows),'cpu_stat_available':all(not x['cgroup_before'].get('unavailable') for x in rows),'samples_throttled':sum(x['cgroup_delta'].get('nr_throttled',0)>0 for x in rows),'throttled_usec_total':sum(x['cgroup_delta'].get('throttled_usec',0) for x in rows),'cpu_usage_counter_available':all('usage_usec' in x['cgroup_delta'] for x in rows),'cpu_usage_usec_total':sum(x['cgroup_delta']['usage_usec'] for x in rows) if all('usage_usec' in x['cgroup_delta'] for x in rows) else None}
 item['cgroup']=throttle;all_groups=[v for pair in throttle.values() for v in pair.values()];clean=all(x['cpu_stat_available'] and x['samples_throttled']==0 and x['throttled_usec_total']==0 for x in all_groups);item['all_samples_without_cgroup_throttle']=clean
 if not clean:item['wall_conclusion']='THROTTLE_OR_MISSING_CGROUP_DATA: retain raw effect, causal wall inference requires review'
 elif item['metrics']['wall_ns']['AB_interval_separated_from_AA_interval']:
  g=item['metrics']['wall_ns']['paired']['AB']['geomean_A_over_other'];item['wall_conclusion']='B slower than A beyond observed A/A variation' if g<1 else 'B faster than A beyond observed A/A variation'
 else:item['wall_conclusion']='INCONCLUSIVE relative to observed A/A variation'
 sources.append({'path':str(p.relative_to(R)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()});summary.append(item)
analysis={'analysis_script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((R/'specs/L011-protocol-v2.json').read_bytes()).hexdigest(),'bootstrap':{'resamples':reps,'percentile_interval':95,'unit':'paired block','seed_base':seed,'method':'Resample paired log ratios then exponentiate mean; no row filtering.'},'inputs':sources,'cases':summary,'scope':'Local CPU class parameter contrast only. No HTTP/PD/model/NPU outcome or calibrated physical peak. No p99 estimate. Startup/warmup/value checks outside measured intervals.'};(O/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n');print(json.dumps([{'case':x['case_key'],'wall_conclusion':x['wall_conclusion'],'wall_ab':x['metrics']['wall_ns']['paired']['AB'],'cpu_ab':x['metrics']['process_cpu_ns']['paired']['AB']} for x in summary]))
