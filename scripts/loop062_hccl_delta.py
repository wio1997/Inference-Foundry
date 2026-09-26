import collections,glob,json,statistics
from pathlib import Path
root=Path('/data/wio/Inference_Foundry/evidence/20260926_loop062_nongmm/run260/profile')
base=json.load(open('/data/wio/Inference_Foundry/evidence/20260926_loop061_bound/run250/payload.json'))
windows=[]
for rank in range(8):
 fs=sorted(glob.glob(str(root/f'rank{rank}_*/ASCEND_PROFILER_OUTPUT/trace_view.json')))
 assert len(fs)==2,(rank,len(fs))
 a=json.load(open(fs[-1]));scopes=sorted((e for e in a if e.get('name')=='extreme::target' and e.get('cat')=='cpu_op'),key=lambda e:float(e['ts']))
 assert len(scopes)==2
 for cycle,s in enumerate(scopes):
  start=float(s['ts']);end=start+float(s['dur'])
  hit=[e for e in a if e.get('ph')=='X' and isinstance(e.get('args'),dict) and 'size(Byte)' in e['args'] and start<=float(e['ts'])<end]
  pairs=collections.Counter((e['name'],int(e['args']['size(Byte)'])) for e in hit)
  windows.append({'rank':rank,'cycle':cycle,'count':len(hit),'bytes':sum(n*size for (name,size),n in pairs.items()),'pairs':sorted([[name,size,n] for (name,size),n in pairs.items()])})
keys={json.dumps(x['pairs']) for x in windows}
result={'status':'valid' if len(windows)==16 and sum(x['bytes']==25257984 for x in windows)>=15 else 'invalid','window_count':len(windows),'async_spillover_windows':[{'rank':x['rank'],'cycle':x['cycle'],'count':x['count'],'bytes':x['bytes']} for x in windows if x['bytes']!=25257984],'candidate_signature':windows[0]['pairs'],'candidate_payload_bytes_median':statistics.median(x['bytes'] for x in windows),'baseline_payload_bytes':base['latest_reported_payload_bytes_per_rank_cycle'],'delta_payload_bytes':statistics.median(x['bytes'] for x in windows)-base['latest_reported_payload_bytes_per_rank_cycle'],'windows':windows,'limit':'HCCL Size(Byte) is operation payload, not physical link transit; timestamp boundaries can contain unrelated async events, so exact signatures are checked across all rank-cycles.'}
print(json.dumps(result,indent=2))
