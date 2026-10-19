from pathlib import Path
import hashlib,json,statistics
r=Path(__file__).resolve().parent
def read(name):
 p=next((r/name).glob('*/metrics.json'));x=json.loads(p.read_text());assert x['completed'];return x,{'path':str(p.relative_to(r)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def q(v,p):
 v=sorted(v);i=(len(v)-1)*p;return v[int(i)]+(v[min(int(i)+1,len(v)-1)]-v[int(i)])*(i%1)
a,sa=read('mtp-moe-graph-live2-c1');b,sb=read('p-bucket-control-c1');c,sc=read('mtp-moe-graph-live2-c4');d,sd=read('p-bucket-control-c4');f,sf=read('p-bucket-control-functional')
assert b['numeric_all_pass'] and d['numeric_all_pass'] and f['numeric_all_pass']
assert len([z for z in b['rows'] if z['phase']=='formal'])==12
assert len([z for z in d['rows'] if z['phase']=='formal'])==192 and len(d['blocks'])==6
ratios={case:{k:b['summary']['native']['cases'][case][k]/a['summary']['native']['cases'][case][k] for k in ['TTFT_ms','last_token_ms','mean_TPOT_ms']} for case in a['summary']['native']['cases']}
sequences={case:{'baseline_unique':len({tuple(z['token_ids']) for z in a['rows'] if z['case_id']==case and z['mode']=='native' and z['phase']=='formal'}),'new_unique':len({tuple(z['token_ids']) for z in b['rows'] if z['case_id']==case and z['phase']=='formal'}),'all_identical_across_P_windows':len({tuple(z['token_ids']) for x in [a,b] for z in x['rows'] if z['case_id']==case and z['mode']=='native' and z['phase']=='formal'})==1} for case in ratios}
pooled={}
for label,x in [('baseline',c),('bucket512',d)]:
 rows=[z for z in x['rows'] if z['mode']=='native' and z['phase']=='formal']
 pooled[label]={k:{'median':statistics.median(z[k] for z in rows),'p95_linear':q([z[k] for z in rows],.95)} for k in ['TTFT_ms','last_token_ms','mean_TPOT_ms']}
def count(lines):return {z.split('{',1)[0].split(':',1)[1]:float(z.rsplit(' ',1)[1]) for z in lines}
block_counters=[]
for i,block in enumerate(d['blocks']):
 before=count(block['native_counters_before']['33'])
 after=count(d['blocks'][i+1]['native_counters_before']['33'] if i+1<len(d['blocks']) else d['idle_checks'][-1]['native_counters']['33'])
 delta={k:after[k]-v for k,v in before.items()};block_counters.append({'index':i,'delta':delta})
result={'decision':'PENDING_REVIEW','sources':[sa,sb,sc,sd,sf],'C1_baseline':a['summary']['native'],'C1_bucket512':b['summary']['native'],'C1_after_over_before_ratios':ratios,'C1_sequence_comparisons':sequences,'C4_baseline':c['summary']['native'],'C4_bucket512':d['summary']['native'],'C4_after_over_before_rate':d['summary']['native']['output_tokens_per_s']/c['summary']['native']['output_tokens_per_s'],'C4_baseline_native_blocks':[z for z in c['blocks'] if z['mode']=='native'],'C4_bucket512_blocks':d['blocks'],'C4_pooled':pooled,'C4_bucket512_native_counter_deltas':block_counters,'C4_bucket512_sequence_comparisons':d['sequence_comparisons'],'scope':'Native configuration control, P [256,1024] to [256,512,1024], same loaded D native and stockproxy, observer/profile/export/transfers off, cold cases/output budgets unchanged. Sequential P windows, not randomized config ABBA. No custom mechanism claim; no general quality or maximum SLO capacity acceptance.'}
(r/'p-bucket-control-verdict-v1.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['C1_after_over_before_ratios','C1_sequence_comparisons','C4_after_over_before_rate','C4_pooled','C4_bucket512_native_counter_deltas']}))
