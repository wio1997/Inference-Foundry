from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis');E=P/'native-execution-identity-v14';M=P/'native-PMU-complete-v18'
out=M/'F04-router-gating-native-callers-and-RmsNormCast-source-locations-v29.json';assert not out.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
fs=pin(E/'native-compute-fullchains-v14-v2.json','6c7bf4bd3a7dcf557327aeadc4a37ec4b9b23cb1ba675353bbd6e5d56a4e1ca0')
ix=pin(E/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4')
apis={d['database_index']:{a['api_rowid']:a for a in d['API_records']}for d in fs['databases']};bychain={};selected=[]
for ref in ix['databases']:
 di=ref['database_index'];d=pin(Path(ref['path']),ref['sha256']);cc={c['globalTaskId']:c for c in d['COMPUTE_definitions']}
 for t in d['TASK_records']:
  c=cc.get(t['globalTaskId'])
  if not c or c['opType_text']!='MoeGatingTopKHash':continue
  aa=[apis[di][key]for key in t.get('native_linked_API_rowids',[])if key in apis[di]]
  for a in aa:bychain[a['chain_key']]=fs['canonical_fullchains'][a['chain_key']]
  selected.append(dict(database_index=di,TASK=t,definition=c,native_APIs=aa))
code=r'''from pathlib import Path
import json,hashlib
root=Path('/vllm-workspace/vllm-ascend/csrc');matches=[]
if root.exists():
 for p in root.rglob('*rms_norm_cast*'):
  if p.is_file():matches.append(p)
  elif p.is_dir():matches.extend(q for q in p.iterdir()if q.is_file())
matches=sorted(set(matches));assert len(matches)<24
rows=[]
for p in matches:
 if p.suffix not in ['.h','.hpp','.cpp','.cc','.c','.cce','.json','.cmake']:continue
 assert p.stat().st_size<200000
 b=p.read_bytes();rows.append(dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),lines=len(b.decode().splitlines())))
binding=root/'torch_binding.cpp';loc=[]
if binding.exists():
 b=binding.read_bytes();ls=b.decode().splitlines();points=set()
 for j,line in enumerate(ls):
  if'npu_rms_norm_cast'in line:points.update(range(max(0,j-4),min(len(ls),j+5)))
 loc=[str(k+1)+': '+ls[k]for k in sorted(points)]
print(json.dumps(dict(exact_op_filename_matches=rows,torch_binding_exact_symbol_context=loc,scope='Only exactRmsNormCast filenames inknowncsrcroot andonebindingliteral;notwholekernel/SDKaudit.')))
'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=35);assert r.returncode==0,r.stderr
locations=json.loads(r.stdout)
headers=[dict(chain_key=k,project_frames=[f['frame']for f in v if'/vllm'in f['frame']],TASKs=sum(any(a['chain_key']==k for a in t['native_APIs'])for t in selected))for k,v in bychain.items()]
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),gating_TASKs=len(selected),native_linked_gating_TASKs=sum(bool(t['native_APIs'])for t in selected),native_fullchains=len(headers),
 gating_output_regimes=[dict(shapes=k[0],dtypes=k[1],TASKs=v)for k,v in Counter((t['definition']['outputShapes_text'],t['definition']['outputDataTypes_text'])for t in selected).items()],
 no_model_import_device_service_work=True,overall_complete=False)
result=dict(summary=summary,native_chain_headers=headers,gating_TASKs=selected,op_source_locations=locations)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();out.write_bytes(b)
print(json.dumps(dict(summary=summary,native_chain_headers=headers,op_source_locations=locations,full_result=str(out),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
