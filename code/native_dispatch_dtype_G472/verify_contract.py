from pathlib import Path
import ast,enum,hashlib,json,types
T=Path(__file__).resolve().parent
base=T.parent/'glm53_pd_align_910b_20261116/G156_SG_runtime_transfer/runtime_sglang_g32'
src=base/'sglang/srt/layers/moe/utils.py';tree=ast.parse(src.read_text());nodes=[x for x in tree.body if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and x.name in ['DispatcherOutputDtype','get_deepep_output_dtype']];assert len(nodes)==2
setting=types.SimpleNamespace(moe=types.SimpleNamespace(deepep_dispatcher_output_dtype='int8'))
ns={'Enum':enum.Enum,'get_server_args':lambda:object(),'get_exec':lambda:setting,'envs':types.SimpleNamespace(SGLANG_DEEPEP_BF16_DISPATCH=types.SimpleNamespace(get=lambda:False)),'_is_npu':True}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),'exec'),ns)
rows=[]
for mode in ['normal','low_latency']:
 for requested in ['int8','auto']:
  setting.moe.deepep_dispatcher_output_dtype=requested
  for role,quant,expected in [('target_W4A8',{'dispatcher_output_dtype':'int8'},'int8'),('draft_BF16',{'dispatcher_output_dtype':'bf16'},'int8' if requested=='int8' else 'bf16')]:
   obj=types.SimpleNamespace(quant_config=quant,runner_backend=None,dispatch_mode=types.SimpleNamespace(value=mode));out=ns['get_deepep_output_dtype'](obj).value;assert out==expected;rows.append({'mode':mode,'server_arg':requested,'expert_type':role,'native_quant_config':quant,'actual_native_resolver_output':out})
method_source=base/'sglang/srt/hardware_backend/npu/quantization/moe_methods.py';mt=ast.parse(method_source.read_text());methods={}
for cls in mt.body:
 if isinstance(cls,ast.ClassDef) and cls.name in ['NPUW4A8Int8MoEMethod','NPUUnquantMoEMethod']:
  methods[cls.name]=[ast.unparse(x) for x in ast.walk(cls) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='_set_moe_dispatch_dtype']
assert methods=={'NPUW4A8Int8MoEMethod':["self._set_moe_dispatch_dtype(layer, 'int8')"],'NPUUnquantMoEMethod':["self._set_moe_dispatch_dtype(layer, 'bf16')"]}
a={'status':'PASSED_ACTUAL_FIXED_NATIVE_RESOLVER_CPU_CONTRACT','cases':rows,'native_method_dtype_setters':methods,'source_sha256':{str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [src,method_source]},'scope':'Original native Enum/resolver executed on CPU with config/group-interface fixtures, no torch tensor/NPU/model/CANN GMM execution. Shows forcedglobalINT8 overrides BF16 draft declaration; auto preserves W4A8 INT8 and draft BF16 in normal and low_latency. No fullmodel/quality/performance acceptance.'};(T/'G472_native_dispatch_dtype_contract.json').write_text(json.dumps(a,indent=2)+'\n');print('Actual native resolver CPU contract passed',len(rows),'cases; target int8/draft bf16 selected with auto, no NPU operation')
