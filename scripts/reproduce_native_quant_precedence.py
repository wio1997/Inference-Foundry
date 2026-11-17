"""Run the fixed native metadata/resolver logic on CPU with a recording config class."""
from pathlib import Path
import ast
import hashlib
import json
import sys
import types

ROOT=Path(__file__).resolve().parent
if (ROOT/'metadata/config.json').exists():repo=ROOT
elif (ROOT.parent/'metadata/config.json').exists():repo=ROOT.parent
else:repo=ROOT/'foundry_sync/Inference-Foundry'
base=repo/'code/fixed_native_loader'
metadata_path=base/'checkpoint_quantization.py'
metadata_tree=ast.parse(metadata_path.read_text())
# The dictionary path never needs Transformers; retain its type name without importing the package.
metadata_tree.body=[n for n in metadata_tree.body if not (isinstance(n,ast.ImportFrom) and n.module=='transformers')]
metadata_module=types.ModuleType('_fixed_checkpoint_metadata_probe')
metadata_module.PretrainedConfig=type('PretrainedConfig',(),{})
sys.modules[metadata_module.__name__]=metadata_module
exec(compile(metadata_tree,str(metadata_path),'exec'),metadata_module.__dict__)
source=base/'weight_utils.py'
tree=ast.parse(source.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_quant_config')
unit=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),function],type_ignores=[])
ast.fix_missing_locations(unit)
class RecordingConfig:
    @classmethod
    def from_config(cls,config):return dict(config)
context={'get_quantization_config':lambda name:RecordingConfig,
         'resolve_checkpoint_quant_spec':metadata_module.resolve_checkpoint_quant_spec,
         'REQUANTIZATION_METHODS':set(),
         '_quark_draft_online_quant_config':lambda *args:None,
         '_resolve_explicit_draft_quant_config':lambda model,config:config}
exec(compile(unit,str(source),'exec'),context)
hf=json.loads((repo/'metadata/config.json').read_text())
assert hf['quantization_config']=={'quant_method':'ascend'}
model=types.SimpleNamespace(quantization='modelslim',hf_config=hf)
result=context['get_quant_config'](model,None,{'model':{}})
assert sorted(result)==['hf_config','packed_modules_mapping','quant_method']
assert result['quant_method']=='ascend'
assert hf['quantization_config']=={'quant_method':'ascend'}
print(json.dumps({'native_resolver_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                  'native_metadata_source_sha256':hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                  'selected_quantization':'modelslim','returned_key_names':sorted(result),
                  'file_based_ModelSlim_descriptors_read':False,
                  'scope':'Native dictionary metadata path and resolver, recording config class; no model/NPU workload'}))
