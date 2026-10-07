from pathlib import Path
import hashlib,json,re
r=Path('/Users/wio/work/Inference-Foundry-glm5-3/glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0277');ids=[785,1196,374,10156,264,3405,304,8452];rows=[]
assert json.loads((r/'state.json').read_text())['status']=='failed'
for label in ('correctness_mode0_short','retained_mode_warm','H6_H5_recovery_warm'):
 raw=(r/(label+'_D.sse')).read_text();assert raw.rstrip().endswith('data: [DONE]');events=[json.loads(x[6:]) for x in raw.splitlines() if x.startswith('data: ') and x!='data: [DONE]'];got=[];usage=[];reason=[]
 for z in events:
  assert 'error' not in z
  if z.get('usage'):usage.append(z['usage'])
  for c in z.get('choices',[]):
   got+=c.get('token_ids') or c.get('delta',{}).get('token_ids') or []
   if c.get('finish_reason'):reason.append(c['finish_reason'])
 result=json.loads((r/(label+'_result.json')).read_text());assert got==ids==result['token_ids'] and usage[-1]==dict(prompt_tokens=2334,completion_tokens=8,total_tokens=2342) and reason[-1]=='length'
 v=result['external_KV_delta'];assert next(value for key,value in v.items() if '_queries_total{' in key)==next(value for key,value in v.items() if '_hits_total{' in key)==2334
 assert all(value==0 for key,value in v.items() if '_created{' in key)
 if label=='correctness_mode0_short':
  transfer=json.loads((r/(label+'_transfer.json')).read_text());assert transfer['all16_native_transfer_success'] and len(transfer['rows'])==16 and {int(re.search(r'local_device_id (\d+)',s).group(1)) for s in transfer['rows']}==set(range(16))
 rows.append(dict(label=label,exact_tokens=8,prompt_tokens=2334,external_D_KV_delta=v,SSE_sha256=hashlib.sha256((r/(label+'_D.sse')).read_bytes()).hexdigest()))
p=r/'failed_request_reduced.json';assert not p.exists();p.write_text(json.dumps(dict(independent_raw_reduction=True,all_three_short_requests_exact=True,requests=rows,original_FAILED_preserved=True,dense_path_correctness=False,performance_gain=None),indent=2)+'\n');print('three exact short PD requests independently reduced; original FAILED retained')
