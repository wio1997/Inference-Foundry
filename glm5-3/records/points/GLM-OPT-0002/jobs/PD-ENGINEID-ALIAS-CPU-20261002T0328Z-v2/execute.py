from pathlib import Path
import ast,json,hashlib,sys,textwrap,copy,logging,types,subprocess
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;s=j.parents[1]/"jobs/PD2-CPU-DIAGNOSTIC-20261002T0158Z/native_sources/0_mooncake_connector.py";raw=s.read_bytes();assert hashlib.sha256(raw).hexdigest()=="f30572890014d0244bf31d8e15dc5d93fb84147df37e0b2c1d1b01cbba6dd533"
path="/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py"
p=subprocess.run(["docker","exec","glm52-single","sha256sum",path],capture_output=True);p.check_returncode();assert p.stdout.decode().split()[0]==hashlib.sha256(raw).hexdigest()
tree=ast.parse(raw.decode());names=["context_parallel_parameters_check","get_kv_head_groups","get_cp_group_meta","get_local_remote_block_port_mappings"];nodes=[]
for name in names:
 found=[x for x in ast.walk(tree)if isinstance(x,ast.FunctionDef)and x.name==name];assert len(found)==1;nodes+=found
src=raw.decode();start=src.index("        if meta.remote_engine_id not in self.local_remote_block_port_mapping:")
end=src.index("        num_external_blocks = math.ceil(meta.num_external_tokens / self.block_size)",start);cache=textwrap.dedent(src[start:end]);assert "self.local_remote_block_port_mapping[meta.remote_engine_id] is None"in cache
aux=[x for x in ast.walk(tree)if isinstance(x,ast.FunctionDef)and x.name=="get_remote_port_send_num"];assert len(aux)==1;nodes+=aux
host=[x for x in ast.walk(tree)if isinstance(x,ast.FunctionDef)and x.name=="_get_remote_host_info_by_port"];assert len(host)==1;nodes+=host
logger=logging.getLogger("native_source_cpu");logger.addHandler(logging.NullHandler());rows=[]
for d_rank in [0,1]:
 for tp in range(16):
  self=types.SimpleNamespace(vllm_config=types.SimpleNamespace(kv_transfer_config=types.SimpleNamespace(kv_port=62100)),use_mla=False,use_sparse=True,tp_size=16,pcp_size=1,dcp_size=16,side_channel_port=62100+d_rank*16,handshake_port=62100+d_rank*16+tp,local_remote_block_port_mapping={},remote_port_send_num={})
  ns={"self":self,"prefill_tp_size":16,"r_blk":1,"copy":copy,"logger":logger,"RemotePortInfo":dict}
  exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),"nativeMooncakeCPmappingExtract","exec"),ns)
  self._get_remote_host_info_by_port=types.MethodType(ns["_get_remote_host_info_by_port"],self)
  def run(engine,port):
   ns["meta"]=types.SimpleNamespace(remote_engine_id=engine,remote_port=port,remote_pcp_size=1,remote_dcp_size=16,remote_multi_nodes_meta_mapping={},remote_host="172.16.10.166"if port==62000 else"172.16.10.167")
   exec(compile(cache,"nativeMooncakeEngineIDCacheExtract","exec"),ns);return copy.deepcopy(ns["local_remote_block_port_mapping"])
  first=run("shared-P",62000);second=run("shared-P",62016)
  assert first==second==[[62000+tp]]
  expected=ns["get_local_remote_block_port_mappings"]()[self.handshake_port];assert expected==[[62016+tp]]and second!=expected
  unique0=run("P-r0",62000);unique1=run("P-r1",62016);assert unique0==[[62000+tp]]and unique1==[[62016+tp]]
  rows.append({"D_rank":d_rank,"TP_rank":tp,"same_engine_id_first":first,"same_engine_id_switch":second,"fresh_expected":expected,"same_id_stale_mapping":True,"unique_engine_id_correct":True})
def ref(f):
 z=f.read_bytes();return{"path":str(f),"bytes":len(z),"sha256":hashlib.sha256(z).hexdigest()}
(j/"extracted_cache_control.py").write_text(cache);out={"at":utc(),"native_source":ref(s),"source_extract":[{"name":x.name,"first_line":x.lineno,"last_line":x.end_lineno}for x in nodes],"cache_control":ref(j/"extracted_cache_control.py"),"CPU_rank_cases":len(rows),"cases":rows,"same_engine_ID_alias_confirmed":True,"separate_P_rank_ID_control_passed":True,"NPU_contexts":0,"models":0,"inference":0,"signals":0,"limits":["Extractedactualnativecontrolfunctions/cacheblock CPUcounterexample, not fullworker/GPU/transport E2E","Run49usesonePsourceperD; alias doesnotinvalidate49singlepaircompletedwireifitpasses","Uniqueper-P-schedulerID neededbeforeheterogeneousPsource switching; actualnativeproducerIDs mustmatch returnedmetadata, cannotrewriteconsumerlabels","No operator/sourcepatch; legitimateengine_id configuration candidate, actualruntimeverification pending"]}
atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",{"schema_version":1,"job_id":j.name,"status":"completed","summary":"Actualnative CPmapping/cachecontrol32CPUrankcases: samePengineID differentPport stale mapping; uniqueP rankIDs control correct; zero NPU/models/requests/signals","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[],"evidence":[dict(id="reduction",locator="exactnativecacheblock+CPfunctions/32CPUcases",**ref(j/"reduction.json"))],"unknowns":out["limits"],"decision_request":None,"next_check_at":None})
print(json.dumps({"alias_confirmed":True,"cases":len(rows),"unique_ID_control":True,"reduction":ref(j/"reduction.json")}))

