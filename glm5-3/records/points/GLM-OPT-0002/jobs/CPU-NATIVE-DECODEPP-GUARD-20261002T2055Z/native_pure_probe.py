import ast,json,types,hashlib
from pathlib import Path
f=Path('/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py');tree=ast.parse(f.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='MooncakeConnectorWorker');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='_get_prefill_decode_size');ns={'VllmConfig':types.SimpleNamespace};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(f),'exec'),ns);rows=[]
for extra,want in [(dict(prefill=dict(dp_size=1,tp_size=16),decode=dict(dp_size=1,tp_size=8)),True),(dict(prefill=dict(dp_size=1,tp_size=16,pp_size=1),decode=dict(dp_size=1,tp_size=8,pp_size=1)),True),(dict(prefill=dict(dp_size=1,tp_size=16,pp_size=1),decode=dict(dp_size=1,tp_size=8,pp_size=2)),False)]:
 config=types.SimpleNamespace(kv_transfer_config=types.SimpleNamespace(get_from_extra_config=lambda name,default:extra.get(name,default)));o=types.SimpleNamespace()
 try:ns[fn.name](o,config);passed=True;error=None
 except AssertionError as e:passed=False;error=str(e);assert error=='decode pp size must be 1'
 assert passed==want;rows.append(dict(extra=extra,accepted=passed,error=error,attributes=vars(o)))
print(json.dumps(dict(source_path=str(f),source_bytes=f.stat().st_size,source_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),native_class=cls.name,method=fn.name,lines=[fn.lineno,fn.end_lineno],rows=rows,scope='Extracted original pure metadata method; no full constructor/workers/model/device allocation/SDK/inference; omitted decodePP defaults1 despite actualPP2 diagnostic. Full PD under DPP2 unsupported by this declared native config.')))
