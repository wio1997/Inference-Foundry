#!/usr/bin/env python3
"""Read finished frontend submit edges and two ranks' Engram hash markers."""
import hashlib
import json
from pathlib import Path
import subprocess
from read_inactive_dp_profile_probe import ROOT

REMOTE=r'''
from pathlib import Path
from hashlib import sha256
import gzip,json,sys
root=Path(sys.argv[1]);assert root.is_relative_to('/data/tiankuan/wio')
p,=list((root/'traces').glob('*.pt.trace.json.gz'));st=p.stat();raw=p.read_bytes();d=json.loads(gzip.decompress(raw));es=d.get('traceEvents',[]) if isinstance(d,dict) else d
parents={}
for e in es:
 a=e.get('args',{})
 if e.get('ph')=='X' and e.get('cat')=='python_function' and 'Python id' in a:
  parents.setdefault(a['Python id'],(e.get('name',''),a.get('Python parent id')))
rows=[]
for e in es:
 if e.get('ph')!='X':continue
 n=e.get('name','')
 if ('core_client.py' in n and any(t in n for t in ['add_request_async','_send_input'])) or 'encode_batch of tokenizers.Tokenizer' in n:
  chain=[];q=e.get('args',{}).get('Python id');seen=set()
  while q in parents and q not in seen and len(chain)<40:
   seen.add(q);name,q=parents[q];chain.append(name)
  rows.append(dict(event={k:e[k] for k in ['name','ts','dur','pid','tid','cat','args'] if k in e},ancestor_names=chain))
assert (st.st_size,st.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
frontend=dict(path=str(p),sha256=sha256(raw).hexdigest(),bytes=st.st_size,mtime_ns=st.st_mtime_ns,events=rows)
del es,d,parents
ranks=[]
for dp in [0,1]:
 p,=list((root/'traces').glob('dp'+str(dp)+'_pp0_tp0_*/ASCEND_PROFILER_OUTPUT/trace_view.json'));st=p.stat();raw=p.read_bytes();d=json.loads(raw);es=d.get('traceEvents',[]) if isinstance(d,dict) else d
 rows=[{k:e[k] for k in ['name','ts','dur','pid','tid','cat','args'] if k in e} for e in es if e.get('ph')=='X' and any(t in e.get('name','').lower() for t in ['engram','hash_ids','hash_cache','req_index'])]
 assert (st.st_size,st.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
 ranks.append(dict(dp=dp,path=str(p),sha256=sha256(raw).hexdigest(),events=rows))
print(json.dumps(dict(frontend=frontend,rank_hash_markers=ranks,new_NPU_work=0,new_profile_collection=0,remote_writes=[]),separators=(',',':')))
'''


def main():
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-c',REMOTE,ROOT],
                     capture_output=True,text=True,timeout=65)
    assert r.returncode==0,r.stderr
    d=json.loads(r.stdout);d['reader_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    Path('audit/execution-search/inactive-dp-frontend-edges.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(frontend_candidates=len(d['frontend']['events']),
                         markers=[len(r['events']) for r in d['rank_hash_markers']])))


if __name__=='__main__':
    main()
