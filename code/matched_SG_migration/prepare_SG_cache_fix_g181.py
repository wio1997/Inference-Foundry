from pathlib import Path
import ast,base64,hashlib,json
T=Path(__file__).resolve().parent;src=T/'sg_pd_g164';dst=T/'sg_pd_g181';dst.mkdir(exist_ok=True)
for f in src.rglob('*'):
 if not f.is_file():continue
 p=dst/f.relative_to(src);p.parent.mkdir(parents=True,exist_ok=True);s=f.read_text().replace('sg_pd_g164','sg_pd_g181')
 if f.name=='node_controller.py':
  s=s.replace("'-g164'","'-g181'").replace('generation=164','generation=181')
  before="  for k,v in env.items():args+=['-e',k+'='+v]"
  after="""  for value in env.values():
   if value.startswith('/task/'):
    directory=ROOT/value.removeprefix('/task/')
    assert RUN in directory.parents or directory==RUN,'Runtime writes must remain under current run'
    directory.mkdir(parents=True,exist_ok=True)
  for k,v in env.items():args+=['-e',k+'='+v]"""
  assert before in s;s=s.replace(before,after)
  before="  with (RUN/'server.log').open('x') as log:"
  after="""  cache_probe='import json,os;from pathlib import Path;keys='+repr(list(env))+';checks={k:dict(path=os.environ[k],exists=Path(os.environ[k]).is_dir(),writable=os.access(os.environ[k],os.W_OK)) for k in keys if os.environ[k].startswith(\"/task/\")};print(json.dumps(checks));assert all(v[\"exists\"] and v[\"writable\"] for v in checks.values())'
  probe=cmd(['docker','exec',NAME,'python3','-c',cache_probe],30)
  atomic(RUN/'runtime_directory_preflight.json',dict(rc=probe.returncode,stdout=probe.stdout,stderr=probe.stderr))
  assert probe.returncode==0,'Cache directory preflight failed before model loading: '+probe.stderr
  with (RUN/'server.log').open('x') as log:"""
  assert before in s;s=s.replace(before,after)
 if f.name.startswith('candidate_'):
  c=json.loads(s);c['revision']='G181: same fixed runtime/model/topology; create and verify every runtime cache directory before model startup, preserve original G164 failure';s=json.dumps(c,indent=2)
 p.write_text(s)
for f in dst.rglob('*.py'):ast.parse(f.read_text())
files={str(f.relative_to(dst)):dict(b64=base64.b64encode(f.read_bytes()).decode(),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in dst.rglob('*') if f.is_file()}
body='''from pathlib import Path
import base64,datetime,hashlib,json,socket,subprocess,time
host=socket.gethostname();role={'<ROLE_NODE>':'p','<ROLE_NODE>':'d'}[host]
root=Path('/data' if role=='p' else '/data1')/'tiankuan/yxy/glm53_pd_align_910b_20261116';old=root/'sg_pd_g164';r=root/'sg_pd_g181'
h=json.loads((old/'actual_launch_handle.json').read_text());assert not Path('/proc',str(h['pid']),'cmdline').exists()
s=json.loads((old/'state.json').read_text());assert s['phase']=='failed_released' and s['cleanup']['all8_process_lists_clear']
assert not r.exists(),'Existing successor needs reconciliation'
r.mkdir()
for name,e in files.items():
 raw=base64.b64decode(e['b64']);assert hashlib.sha256(raw).hexdigest()==e['sha256'];p=r/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
(r/'candidate.json').write_bytes((r/f'candidate_{role}.json').read_bytes())
with (r/'controller.log').open('x') as log:child=subprocess.Popen(['python3','-u',str(r/'node_controller.py')],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
handle={'pid':child.pid,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':hashlib.sha256((r/'node_controller.py').read_bytes()).hexdigest(),'lifetime_s':72000,'predecessor':'G164 terminal failed_released all8clear; original retained'};(r/'actual_launch_handle.json').write_text(json.dumps(handle,indent=2));time.sleep(3)
print('G181_SG_CACHE_FIX_START_JSON '+json.dumps({'handle':handle,'actual_live':child.poll() is None,'state':json.loads((r/'state.json').read_text()) if (r/'state.json').exists() else None}))
'''
(T/'start_SG_cache_fix_g181.sh').write_text("python3 - <<'REMOTE_PY'\nfiles="+repr(files)+'\n'+body+'REMOTE_PY\n')
obs=(T/'observe_matched_SG_brief_g180.sh').read_text().replace('sg_pd_g164','sg_pd_g181').replace("'-g164'","'-g181'").replace('G180_MATCHED_SG_BRIEF_JSON','G183_MATCHED_SG_BRIEF_JSON')
(T/'observe_matched_SG_brief_g183.sh').write_text(obs)
functional=(T/'sg_G164_native_PD_functional_g178.py').read_text().replace('sg_pd_g164','sg_pd_g181').replace('functional_PD_g178','functional_PD_g182').replace("s['generation']==164","s['generation']==181").replace('G164','G181')
ast.parse(functional);(T/'sg_G181_native_PD_functional_g182.py').write_text(functional)
start=(T/'start_functional_g178.sh').read_text()
# Rebuild the payload assignment independently; never mutate an encoded blob.
prefix="python3 - <<'REMOTE_PY'\n";start=start.removeprefix(prefix).removesuffix('REMOTE_PY\n')
tree=ast.parse(start)
for node in tree.body:
 if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='b64decode':
  old_payload=ast.literal_eval(node.value.args[0]);start=start.replace(repr(old_payload),repr(base64.b64encode(functional.encode()).decode()));break
old_hash=hashlib.sha256((T/'sg_G164_native_PD_functional_g178.py').read_bytes()).hexdigest();start=start.replace(old_hash,hashlib.sha256(functional.encode()).hexdigest())
start=start.replace('sg_pd_g164','sg_pd_g181').replace('functional_PD_g178','functional_PD_g182').replace('functional_actual_handle_g178','functional_actual_handle_g182').replace('functional_g178','functional_g182').replace('G178_FUNCTIONAL_START_JSON','G182_FUNCTIONAL_START_JSON')
ast.parse(start);(T/'start_functional_g182.sh').write_text(prefix+start+'REMOTE_PY\n')
print('G181 cache paths mkdir + actual container writable-directory gate; G182 functional source prepared')
