#!/usr/bin/env python3
"""Inventory static profiler source and offline DB schema; do not print raw data."""
import base64,json,pathlib,shlex,subprocess
remote=r'''import hashlib,json,os,pathlib,re,sqlite3
base=pathlib.Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler')
root=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binary-binding-draft/second-binding-run/offline-ep8-20261029')
status=json.loads((root/'status.json').read_text());item=next(x for x in status['dbs'] if x['path'].endswith('/ascend_pytorch_profiler_8.db'));db=root/item['path']
assert not pathlib.Path(str(db)+'-wal').exists() and not pathlib.Path(str(db)+'-journal').exists()
c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True);c.execute('PRAGMA query_only=ON')
tables=[x[0] for x in c.execute("SELECT name FROM sqlite_master WHERE type='table'")];selected=[]
for t in tables:
 if t=='PYTORCH_API' or re.search('TENSOR|MEMORY|ALLOCATION|STORAGE',t,re.I):
  cols=c.execute('PRAGMA table_info("'+t.replace('"','""')+'")').fetchall();selected.append({'table':t,'columns':[{'name':x[1],'type':x[2]} for x in cols]})
c.close()
patterns=re.compile(r'tensor_info|storage_offset|tensor_id|tensorid|input.?strides|input.?shapes|input.?dtypes',re.I)
files=[]
for p in sorted(base.rglob('*.py')):
 if p.is_symlink() or p.stat().st_size>600000:continue
 try:b=p.read_bytes();ls=b.decode().splitlines()
 except (OSError,UnicodeDecodeError):continue
 matches=[{'line':i+1,'text':line[:260]} for i,line in enumerate(ls) if patterns.search(line)]
 if matches:files.append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'matches':matches[:60]})
print(json.dumps({'db_sha256_expected':item['sha256'],'db_schema_selected':selected,'profiler_python_root':str(base),'source_files':files,'note':'schema and static-source text only; no tensor or trace contents read'}))'''
e=base64.b64encode(remote.encode()).decode();cmd='docker exec wio-vllm-v41-repro169-stage-20261028 nice -n 15 python3 -c '+shlex.quote("import base64;exec(base64.b64decode('"+e+"'))")
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,text=True,timeout=120,check=True);v=json.loads(r.stdout);o=pathlib.Path(__file__).parent/'tensorinfo-export-schema.json';o.write_text(json.dumps(v,indent=2)+'\n');print('tables',len(v['db_schema_selected']),'source_files',len(v['source_files']))
