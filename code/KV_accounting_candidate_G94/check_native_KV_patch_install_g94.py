"""CPU installation check on the actual native constructor with import-only module stubs; no model execution."""
from pathlib import Path
import ast,hashlib,json,sys,types
import kv_geometry_g94
ROOT=Path(__file__).resolve().parent
source=ROOT/'G86_native_planner_sources/sglang/srt/model_executor/pool_configurator.py'
cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='DSV4PoolConfigurator')
function=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
for name in ['sglang','sglang.srt','sglang.srt.model_executor','sglang.srt.hardware_backend','sglang.srt.hardware_backend.npu']:
 module=types.ModuleType(name);module.__path__=[];sys.modules[name]=module
native=types.ModuleType('sglang.srt.model_executor.pool_configurator');native.__file__=str(source);native._is_npu=True
unit=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),function],type_ignores=[]);ast.fix_missing_locations(unit);exec(compile(unit,str(source),'exec'),native.__dict__)
native.DSV4PoolConfigurator=type('DSV4PoolConfigurator',(),{'__init__':native.__dict__['__init__']});sys.modules[native.__name__]=native
utils=types.ModuleType('sglang.srt.hardware_backend.npu.utils');utils.is_npu_arch35=lambda:False;sys.modules[utils.__name__]=utils
out=ROOT/'kv_geometry_cpu_check_g94';out.mkdir(exist_ok=False)
(out/'KV_native_source_sha256.json').write_text(json.dumps({'pool_configurator.py':hashlib.sha256(source.read_bytes()).hexdigest()}))
old=native.DSV4PoolConfigurator.__init__.__code__;kv_geometry_g94.install(out);new=native.DSV4PoolConfigurator.__init__.__code__
assert old is not new and '_ds41_910b_cost_scope' in new.co_names and not new.co_freevars
record={'scope':'CPU installation/compilation of AST overlay on actual fixed native constructor with import-only stubs; constructor/model/NPU not executed','patched_actual_constructor_code':True,'hook_in_original_function_globals':'_ds41_910b_cost_scope' in native.__dict__,'closure_free':not new.co_freevars,'model_capacity_verified':False}
(out/'installation_check.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
