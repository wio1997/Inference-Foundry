from pathlib import Path
import json,sys,subprocess,shlex,ast,hashlib,concurrent.futures
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parents[1];s=json.loads((p/'runs/GLM-RUN-0106/state.json').read_text());assert s['status']=='completed'and not same_process(s['owner'])
idx=json.loads((p/'jobs/REDUCE-PROFILE105-20261002T1934Z/raw_index.json').read_text())
old=p/'jobs/EXPORT-PROFILE105-RANK0-20261002T1940Z'
tree=ast.parse((old/'export.py').read_text());helpers=[ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='code'for t in n.targets)]
assert len(helpers)==2
for code in helpers:ast.parse(code)
def remote(args,log,timeout=600,payload=None):
 z=subprocess.run(['ssh','-o','BatchMode=yes','root@172.16.10.167',shlex.join(args)],input=payload,capture_output=True,timeout=timeout)
 log.with_suffix('.stdout').write_bytes(z.stdout);log.with_suffix('.stderr').write_bytes(z.stderr);z.check_returncode();return z.stdout
def work(profile):
 raw=Path(profile['directory']);dest=Path('/data/tiankuan/wio/glm52-pd/deploy/private/profile_run105_export_remaining')/raw.name;folder=j/('pid'+str(profile['pid']));folder.mkdir()
 remote(['python3','-c',helpers[0],str(raw),str(dest)],folder/'copy',120)
 shell='source /usr/local/Ascend/ascend-toolkit/set_env.sh; exec /usr/local/Ascend/ascend-toolkit/latest/bin/msprof --export=on --output='+shlex.quote(str(dest))
 cmd=['docker','exec','glm52-single','/bin/bash','-c',shell];atomic_json(folder/'command.json',dict(at=utc(),argv=cmd,source=str(raw),derived=str(dest),GPU_requests=0))
 remote(cmd,folder/'export',600)
 orig=[x for x in idx['files']if Path(x['path']).is_relative_to(raw)]
 out=json.loads(remote(['python3','-c',helpers[1],str(dest),str(raw)],folder/'index',180,json.dumps(orig).encode()));out.update(pid=profile['pid'],source=str(raw),destination=str(dest));atomic_json(folder/'index.json',out);return out
results=[json.loads((old/'reduction.json').read_text())];atomic_json(j/'progress.json',dict(at=utc(),completed=1,total=16,CPU_export_parallelism=2))
with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
 futures={pool.submit(work,x):x for x in idx['profiles'][1:]}
 for f in concurrent.futures.as_completed(futures):
  results.append(f.result());atomic_json(j/'progress.json',dict(at=utc(),completed=len(results),total=16,CPU_export_parallelism=2,pids=[x.get('pid')for x in results]))
out=dict(at=utc(),profiles=results,profile_count=len(results),copied_raw_hashes_unchanged=sum(x['raw_copy_hashes_unchanged']for x in results),original_raw_index_sha256=hashlib.sha256((p/'jobs/REDUCE-PROFILE105-20261002T1934Z/raw_index.json').read_bytes()).hexdigest(),limits=['CPU-only exports of immutable-input copies for16workers; no GPUrequests/modelsignals/restarts or replay','NoCurrent/KEEP/capacity; task wait/overlap/commonwindow/multirankcriticalchain require GPT reduction'])
assert len(results)==16;atomic_json(j/'reduction.json',out);b=(j/'reduction.json').read_bytes();atomic_json(j/'result.json',dict(schema_version=1,job_id=j.name,status='completed',summary='Run105 all16worker CPU native export indexed, copied raw hashes unchanged, no GPU rerun',execution=dict(inner_exit_code=0,acceptance='passed',processes=[]),findings=[],evidence=[dict(id='reduction',path=str(j/'reduction.json'),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator='all16native exports/inputhashes/derivedfiles')],unknowns=out['limits'],decision_request=None,next_check_at=None));print(json.dumps(dict(profiles=len(results),raw_copy_hashes_unchanged=out['copied_raw_hashes_unchanged'])))
