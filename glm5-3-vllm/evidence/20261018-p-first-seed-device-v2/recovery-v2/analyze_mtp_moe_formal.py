"""Reduce frozen observer-off finite C1/C4 results without a model request."""
from pathlib import Path
import hashlib,json,math,statistics
r=Path(__file__).resolve().parent

def read(name):
 p=next((r/name).glob('*/metrics.json'));x=json.loads(p.read_text());assert x['completed'];return x,{'path':str(p.relative_to(r)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def quantile(values,q):
 v=sorted(values);k=(len(v)-1)*q;return v[int(k)]+(v[min(int(k)+1,len(v)-1)]-v[int(k)])*(k%1)
c1,s1=read('mtp-moe-graph-live2-c1');c4,s4=read('mtp-moe-graph-live2-c4');cancel,sc=read('mtp-moe-graph-live2-cancel')
quality,sq=read('mtp-moe-graph-quality')
formal=[z for z in c4['rows'] if z['phase']=='formal'];assert len(formal)==384 and len(c4['blocks'])==12
assert all(z['numeric_pass'] and z['functional_pass'] for z in formal)
assert c1['numeric_all_pass'] and len([z for z in c1['rows'] if z['phase']=='formal'])==24
cycles=[]
for i in range(3):
 blocks=c4['blocks'][4*i:4*(i+1)];rates={m:sum(z['output_tokens'] for z in blocks if z['mode']==m)/sum(z['seconds'] for z in blocks if z['mode']==m) for m in ['native','adaptive']}
 cycles.append({'index':i,'rates':rates,'candidate_over_native':rates['adaptive']/rates['native']})
pooled={}
for mode in ['native','adaptive']:
 rows=[z for z in formal if z['mode']==mode]
 pooled[mode]={k:{'median':statistics.median(z[k] for z in rows),'p95_linear':quantile([z[k] for z in rows],.95)} for k in ['TTFT_ms','last_token_ms','mean_TPOT_ms']}
case_ratios={case:{k:c1['summary']['adaptive']['cases'][case][k]/c1['summary']['native']['cases'][case][k] for k in ['TTFT_ms','last_token_ms','mean_TPOT_ms']} for case in c1['summary']['native']['cases']}
equal_ratios={k:math.exp(statistics.mean(math.log(z[k]) for z in case_ratios.values())) for k in ['TTFT_ms','last_token_ms','mean_TPOT_ms']}
matched={rid:[s for s in lines if __import__('re').search(r'\b(?:abort(?:ed|ing)?|cancel(?:led|ed|ling|ing)?|disconnect(?:ed)?)\b',s.replace(rid,''),__import__('re').I)] for rid,lines in cancel['backend_abort_log_evidence'].items()}
assert sum(bool(z) for z in matched.values())==cancel['backend_abort_log_request_count']
assert cancel['client_close_requests']==10 and cancel['full_reuse_and_peer_responses']==16
result={'decision':'PENDING_REVIEW_OF_FULL_RESULTS','sources':[s1,s4,sc,sq],'C1':c1['summary'],'C1_case_ratios':case_ratios,'C1_equal_case_ratios':equal_ratios,'C1_sequence_comparisons':c1['sequence_comparisons'],'C4':c4['summary'],'C4_blocks':c4['blocks'],'C4_cycles':cycles,'C4_pooled':pooled,'C4_candidate_over_native':c4['summary']['adaptive']['output_tokens_per_s']/c4['summary']['native']['output_tokens_per_s'],'C4_sequence_comparisons':c4['sequence_comparisons'],'quality_reused_same_build_from_live1':{'summary':quality['summary'],'sequence_comparisons':quality['sequence_comparisons'],'full64_exact26of26_between_modes':True},'cancel':{k:cancel.get(k) for k in ['completed','client_close_requests','full_reuse_and_peer_responses','backend_abort_log_request_count','abort_counter_deltas']},'physical_evidence':{'live1':'16 ranks x50 exact added copies all SDMA_SQE','live2_C4':'619/624 exact copies linked SDMA; rank0 five CPU/API records without device-task association remain unproven; each observed selected N6/N18/N24 on every rank has direct SDMA samples. Do not claim all624 closed.'},'scope':'Same pinned all-SDMA native P->KV->D, official unmodified proxy29880, observer/profile/export/related transfer off, C1 fixed64 and balanced rotated C4 four slots x8 requests/block, ABBAx3, ordinary64/128. No SLO maximum capacity or general-quality acceptance. All block/cycle/case results retained; no profiled durations added as service cost.'}
(r/'mtp-moe-graph-formal-verdict-v1.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['C1_equal_case_ratios','C1_case_ratios','C4_candidate_over_native','C4_cycles','C4_pooled','C1_sequence_comparisons','C4_sequence_comparisons','cancel']}))
