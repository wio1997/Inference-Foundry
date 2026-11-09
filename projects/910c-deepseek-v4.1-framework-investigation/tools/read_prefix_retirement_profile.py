#!/usr/bin/env python3
"""Read finished traces for cache-init labels and whole prepare-input parents."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from read_inactive_dp_profile_probe import ROOT

REMOTE = r'''
from pathlib import Path
from hashlib import sha256
from collections import defaultdict
import json,re,sys
root=Path(sys.argv[1]);assert root.is_relative_to('/data/tiankuan/wio');out=[]
paths=sorted((root/'traces').glob('dp*_pp0_tp*_*/ASCEND_PROFILER_OUTPUT/trace_view.json'));assert len(paths)==16
for p in paths:
 st=p.stat();raw=p.read_bytes();x=json.loads(raw);es=x.get('traceEvents',[]) if isinstance(x,dict) else x
 dp,tp=map(int,re.search(r'dp(\d+)_pp0_tp(\d+)_',str(p)).groups())
 parents=[];cache=[];generic=defaultdict(lambda:[0,0.])
 for e in es:
  if e.get('ph')!='X':continue
  name=e.get('name','');low=name.lower();cat=e.get('cat','')
  if name in ['prepare input','gloo:all_reduce','AscendCL@aclmdlRIExecuteAsync']:
   parents.append({k:e[k] for k in ['name','ts','dur','pid','tid','cat','args'] if k in e})
  if any(w in low for w in ['zero_kv_blocks','zero_block_ids','copy_kv_cache_blocks','fill_mxfp_v_scale']):
   cache.append({k:e[k] for k in ['name','ts','dur','pid','tid','cat','args'] if k in e})
  if any(w in low for w in ['zero','memset','fill']):
   a=generic[(cat,name)];a[0]+=1;a[1]+=e.get('dur',0)
 assert (st.st_size,st.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
 out.append(dict(rank=8*dp+tp,path=str(p),bytes=len(raw),mtime_ns=st.st_mtime_ns,sha256=sha256(raw).hexdigest(),parents=parents,cache_specific_events=cache,
  generic_name_navigation=[dict(category=k[0],name=k[1],n=v[0],sum_duration_us=v[1]) for k,v in generic.items()]))
print(json.dumps(dict(files=out,scope='Only completed independent profile first worker-window; absence of Python labels is not proof of no source call. Generic zero/fill totals are navigation, never cache-init gain.',new_NPU_work=0,new_profile_collection=0,service_requests=0,remote_writes=[]),separators=(',',':')))
'''


def main():
    r = subprocess.run(['/Users/wio/.local/bin/baai-ssh', '171', '--',
                        'python3', '-I', '-S', '-c', REMOTE, ROOT],
                       capture_output=True, text=True, timeout=115)
    if r.returncode:
        raise RuntimeError(r.stderr)
    x = json.loads(r.stdout)
    x['reader_sha256'] = sha256(Path(__file__).read_bytes()).hexdigest()
    p = Path('audit/execution-search/prefix-retirement-profile.json')
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps([dict(rank=q['rank'],parents=len(q['parents']),
                           cache_specific=len(q['cache_specific_events']),
                           generic_labels=len(q['generic_name_navigation'])) for q in x['files']]))


if __name__ == '__main__':
    main()
