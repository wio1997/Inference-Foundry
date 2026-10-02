from pathlib import Path
import json,sys,subprocess,re,hashlib,urllib.request,ast
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;root=Path("/data/tiankuan/wio");raw=subprocess.check_output(["ss","-ltnp"],text=True);(j/"sockets.txt").write_text(raw)
rows=[l for l in raw.splitlines()if re.search(r":8000\s",l)];pids={int(v)for l in rows for v in re.findall(r"pid=(\d+)",l)};assert len(rows)<=1 and len(pids)<=1
out=dict(at=utc(),scope="Read-only current task public8000 resource/source inventory; no signals/starts/inference",listeners=rows,owners=[],files=[],models=0,requests=0,signals=0)
for pid in pids:
 p=Path("/proc")/str(pid);b=(p/"stat").read_text();v=b[b.rfind(")")+2:].split()
 argv=[x.decode()for x in(p/"cmdline").read_bytes().split(bytes([0]))if x];cwd=str((p/"cwd").resolve());exe=str((p/"exe").resolve())
 out["owners"].append(dict(pid=pid,identity=dict(boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),start_ticks=v[19]),state=v[0],parent_pid=int(v[1]),argv=argv,cwd=cwd,exe=exe))
 candidates=[Path(x)for x in argv if x.endswith(".py")and x.startswith(str(root)+"/")]
 for f in candidates:
  assert f.resolve().is_relative_to(root)
  data=f.read_bytes();assert len(data)<524288
  dest=j/("source_"+f.name);dest.write_bytes(data)
  ast.parse(data.decode())
  out["files"].append(dict(path=str(f),frozen=str(dest),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
http=urllib.request.build_opener(urllib.request.ProxyHandler({}));health=[]
if pids:
 for endpoint in["health","metrics"]:
  try:
   with http.open("http://172.16.10.166:8000/"+endpoint,timeout=10)as res:b=res.read(2097152);status=res.status
   f=j/(endpoint+".wire");f.write_bytes(b);health.append(dict(endpoint=endpoint,status=status,path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
  except Exception as err:health.append(dict(endpoint=endpoint,error_type=type(err).__name__,error=str(err)))
out["readonly_endpoints"]=health;out["limits"]=["Public8000 inventory only; no native inference/API completion or performance claim","Existing APIfault-domain andownership semantics evaluated from actualsource; no browser/auth/secrets access, no cleanup authorization inferred from public port alone"]
atomic_json(j/"reduction.json",out);b=(j/"reduction.json").read_bytes()
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Read-only currentpublic8000 listener/source/boot-start inventory, no inference/modeloperations",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="current8000listener/identity/actualsource/GETonly")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(owners=out["owners"],files=out["files"],health=health)))

