from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,statistics,tarfile
r=Path(__file__).resolve().parent;receipt=json.loads(next((r/'rope-cache-layout-final-33-snapshot').glob('*/artifacts-backup.json')).read_text());root=Path(receipt['local_archive']).parent;p=root/'rank0-getter.jsonl'
if not p.exists():
 with tarfile.open(receipt['local_archive'],'r:gz') as tar:
  members=[z for z in tar.getmembers() if '/rope-cache-layout-probe/rank-0-' in z.name and z.name.endswith('.jsonl')];assert len(members)==1;p.write_bytes(tar.extractfile(members[0]).read())
proof=json.loads(next((r/'rope-cache-layout-paths-v2').glob('*/metrics.json')).read_text());source=next(z for z in proof['files'] if z['path'].startswith('rank-0-'));assert p.stat().st_size==source['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==source['sha256']
frames=[json.loads(z) for z in p.read_text().splitlines()];patterns=Counter();group=defaultdict(list)
for f in frames:
 v=tuple((c['N'],c['use_cache']) for c in f['calls']);patterns[(f['phase'],v)]+=1;group[f['phase']].append({'calls':len(v),'cache_true':sum(c['use_cache'] for c in f['calls']),'cache_false':sum(not c['use_cache'] for c in f['calls'])})
result={'source':source,'completed':True,'phase_counts':{k:len(v) for k,v in group.items()},'phase_call_medians':{k:{q:statistics.median(z[q] for z in v) for q in ['calls','cache_true','cache_false']} for k,v in group.items()},'patterns':[{'phase':k[0],'N_and_use_cache':k[1],'frames':n} for k,n in patterns.most_common(12)],'scope':'Existing observed getter call order/shape only. Same N does not prove same positions, pointer/version or consumer lifetime. Input values were not logged, so result reuse remains a source/ownership question, not a validated optimization. Continuous-table fragment pair lookup is~0.05ms; any savings estimate must include callers and exposed wait, not simply multiply call counts.'}
(r/'rope-lookup-reuse-audit-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
