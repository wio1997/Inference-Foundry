#!/usr/bin/env python3
"""Join already-verified same-scope pad/gate and MC2 summaries; CPU only."""
import hashlib,json
from decimal import Decimal
from pathlib import Path
spec=[('agents/diagnosis/int64-pad-gating-scope.json','49e4028342df962d0724c7f95aac4537b4804f7227049ffa42bf22040829c218'),('agents/diagnosis/mc2-groups-rank-spread.json','6f8ad52652cb3568d8eb23d6cdb86c093b37f533332258d212b3a720fd2a76cd')]
a=[]
for name,digest in spec:
 data=Path(name).read_bytes();assert hashlib.sha256(data).hexdigest()==digest;a.append(json.loads(data))
pads={(r['ep'],r['layer'],r['scope']):r for r in a[0]['rows']};assert len(pads)==320
groups=[g for g in a[1]['groups'] if g['digest_prefix']=='7ddf7ae98a00'];assert len(groups)==1
out=[]
for layer in groups[0]['groups']:
 order=sorted((Decimal(r['duration_us']),int(ep)) for ep,r in layer['records'].items());assert len(set(v for v,_ in order))==8
 for position,(_,ep) in enumerate(order,1):
  m=layer['records'][str(ep)];p=pads[(ep,layer['layer'],m['scope'])];assert p['pad']['stream']==p['gate']['stream']==m['stream']
  out.append({'ep':ep,'layer':layer['layer'],'scope':m['scope'],'pad_to_gate_gap_us':p['gap_us'],'mc2_rank_shortest':position,'mc2_duration_us':m['duration_us']})
assert len(out)==320
summary={str(ep):{'long_gap_count':sum(Decimal(r['pad_to_gate_gap_us'])>100 for r in out if r['ep']==ep),'mc2_position_counts':{str(k):sum(r['mc2_rank_shortest']==k for r in out if r['ep']==ep) for k in range(1,9)}} for ep in range(8)}
assert summary['3']['long_gap_count']==40 and summary['3']['mc2_position_counts']['1']==40
assert summary['2']['long_gap_count']==39 and summary['2']['mc2_position_counts']['2']==40
assert all(summary[str(ep)]['long_gap_count']==0 for ep in (0,1,4,5,6,7))
exc=[r for r in out if r['ep']==2 and Decimal(r['pad_to_gate_gap_us'])<=100];assert len(exc)==1 and exc[0]['layer']==28
result={'threshold_us':'100; selected after observation, descriptive only','scope':'one profiled capture; same digest, rank, layer and scope','per_rank':summary,'ep2_exception':exc[0],'rows':out}
Path('agents/diagnosis/pad-gap-mc2-rank.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
