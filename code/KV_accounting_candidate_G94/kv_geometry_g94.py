"""Account the actual 910B PA_ND and side-indexer allocation; no cache dtype or model change."""
from pathlib import Path
import ast,hashlib,inspect,json,os,textwrap


def install(root):
    import sglang.srt.model_executor.pool_configurator as native
    from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
    cls=native.DSV4PoolConfigurator
    if getattr(cls,'_ds41_npu_bytes_installed',False):return
    expected=json.loads((root/'KV_native_source_sha256.json').read_text())['pool_configurator.py']
    if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()!=expected:raise RuntimeError('Native pool configurator source changed')
    def scope(self,kvc):
        return (native._is_npu and not is_npu_arch35()
                and getattr(kvc.model_config.hf_config,'model_type',None)=='deepseek_v41'
                and not self._unified and self.kv_cache_dtype_str=='bfloat16')
    original=cls.__init__;tree=ast.parse(textwrap.dedent(inspect.getsource(original)));function=tree.body[0]
    def attribute_target(node,name):
        return isinstance(node,ast.Assign) and any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='self' and t.attr==name for t in node.targets)
    result=[];changes=[]
    for statement in function.body:
        result.append(statement)
        # This entire if/elif/else chooses KV row storage before coefficients/fixed buffers are calculated.
        is_kv_choice=isinstance(statement,ast.If) and ast.unparse(statement.test)=='self._unified' and any(attribute_target(n,'kv_bytes') for n in ast.walk(statement))
        if is_kv_choice:
            result.extend(ast.parse('if _ds41_910b_cost_scope(self,kvc):\n    self.kv_bytes = self.attn_head_dim * 2').body);changes.append('BF16_PA_ND_KV_row')
        for attr in ['indexer_bytes_per_token','low_ratio_index_bytes']:
            if attribute_target(statement,attr):
                # Native NPU indexer retains the packed base and adds int8 K + one fp16 scale.
                result.extend(ast.parse(f'if _ds41_910b_cost_scope(self,kvc):\n    self.{attr} += self.indexer_head_dim + 2').body);changes.append(attr+'_NPU_side_buffers')
    if changes!=['BF16_PA_ND_KV_row','indexer_bytes_per_token_NPU_side_buffers','low_ratio_index_bytes_NPU_side_buffers']:raise RuntimeError('Unexpected fixed native byte-accounting layout')
    function.body=result
    module=ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),function],type_ignores=[]));source=ast.unparse(module)+'\n'
    context=dict(native.__dict__);context['_ds41_910b_cost_scope']=scope;exec(compile(module,str(root/'native_pool_configurator_overlay.py'),'exec'),context)
    updated=context['__init__']
    if original.__code__.co_freevars or updated.__code__.co_freevars:raise RuntimeError('Unexpected native constructor closure')
    native.__dict__['_ds41_910b_cost_scope']=scope
    original.__code__=updated.__code__;cls._ds41_npu_bytes_installed=True
    (root/f'KV_cost_overlay.pid{os.getpid()}.json').write_text(json.dumps({'native_source_sha256':expected,'changes':changes,'overlay_sha256':hashlib.sha256(source.encode()).hexdigest(),'BF16_KV_static_fraction_context_C16_graphs_unchanged':True,'actual_model_capacity_or_NPU_execution_validated':False},indent=2))
