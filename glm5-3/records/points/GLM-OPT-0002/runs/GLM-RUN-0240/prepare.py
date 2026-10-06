from common import *
prev=json.loads((p/"runs/GLM-RUN-0239/state.json").read_text());assert prev["status"]=="completed"and not same_process(prev["owner"])
cpu=p/"jobs/NATIVE-RESPONSES-COMPATIBILITY-CPU-20261006T020856Z-v5"
assert json.loads((cpu/"result.json").read_text())["status"]=="completed"and json.loads((cpu/"reduction.json").read_text())["CPU_gateway_VALID"]
for src in json.loads((cpu/"reduction.json").read_text())["sources"]:
 assert ref(src["path"])=={k:src[k]for k in ["path","bytes","sha256"]}
 assert (bundle/Path(src["path"]).name).read_bytes()==Path(src["path"]).read_bytes()
old=p/"runs/GLM-RUN-0239";proof=json.loads((old/"restored/public_service_proof.json").read_text())
assert same_process(proof["host"])and ref(proof["config"]["path"])==proof["config"]
assert [x.decode()for x in Path("/proc/"+str(proof["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x]==proof["argv"]
# V13 config must be validated under its own compiler, isolated from V14 imports.
check="import sys;sys.path.insert(0,"+repr(str(old/"runtime_bundle"))+");from native_engines_service_config import checked_config;checked_config("+repr(proof["config"]["path"])+");print('V13_CONFIG_VALID')"
run("166",["python3","-c",check],"old_V13_config")
m=json.loads((r/"restored/native_engines_resident.json").read_text())
for e in m["engines"]:
 for key,o in e["roots"].items():fresh(o,e["members"][key],"prepare_"+e["replica_id"]);native_idle(o["host"],9081 if o["host"]=="166"else 9900,"before")
snap=request("/control/replicas");assert len(snap["replicas"])==2and all(not x["active_requests"]and not x["draining"]and not x["group_faulted"]for x in snap["replicas"])
with http.open("http://127.0.0.1:8000/v1/responses/resp_glm_run211_D1_new",timeout=15)as response:raw=response.read();assert response.status==200
assert json.loads(raw)==json.loads((old/"resp_glm_run211_D1_new.wire").read_text())
(r/"before_STORE211.wire").write_bytes(raw)
atomic_json(r/"prepared_state.json",dict(at=utc(),native32_same_idle=True,old_public_exact=True,old_config_under_original_compiler=True,CPU_VALID=True,new_config_V14=True,models_signalled=0))
