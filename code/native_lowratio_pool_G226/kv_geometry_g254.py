"""Selected 910B BF16 + native FP4 low-ratio allocation pricing candidate.

Not installed in an active service. SWA/state/draft/model allocation still needs
its full-engine acceptance; this overlay fixes the verified low-ratio geometry.
"""
from pathlib import Path
import ast,hashlib,inspect,json,os,textwrap


def low_ratio_allocated_bytes(full_tokens,ratio,page_size=128):
    size=full_tokens//ratio
    physical_rows=(size+page_size+1)//page_size*page_size
    index_size=(full_tokens+page_size)//ratio
    index_rows=(index_size+64+63)//64*64
    return physical_rows*1024,index_rows*68


def low_ratio_padding_bytes(full_tokens,ratio,page_size=128):
    kv,index=low_ratio_allocated_bytes(full_tokens,ratio,page_size)
    return kv+index-(full_tokens//ratio)*(1024+68)


def install(root):
    import sglang.srt.model_executor.pool_configurator as native
    from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
    cls=native.DSV4PoolConfigurator
    if getattr(cls,'_ds41_selected_lowratio_bytes_g254',False):return
    expected=json.loads((root/'KV_native_source_sha256.json').read_text())['pool_configurator.py']
    assert hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()==expected
    def scope(self,kvc):
        return (native._is_npu and not is_npu_arch35()
                and getattr(kvc.model_config.hf_config,'model_type',None)=='deepseek_v41'
                and not self._unified and self.kv_cache_dtype_str=='bfloat16')
    original=cls.__init__;tree=ast.parse(textwrap.dedent(inspect.getsource(original)));fn=tree.body[0]
    def target(node,name):
        return isinstance(node,ast.Assign) and any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='self' and t.attr==name for t in node.targets)
    body=[];changes=[]
    for statement in fn.body:
        body.append(statement)
        kv_choice=isinstance(statement,ast.If) and ast.unparse(statement.test)=='self._unified' and any(target(n,'kv_bytes') for n in ast.walk(statement))
        if kv_choice:
            body.extend(ast.parse('self._ds41_selected_lowratio_scope_g254 = _ds41_cost_scope_g254(self,kvc)\nif self._ds41_selected_lowratio_scope_g254:\n    self.kv_bytes = self.attn_head_dim * 2').body);changes.append('BF16_PA_ND_row')
        if target(statement,'low_ratio_index_bytes'):
            body.extend(ast.parse('if self._ds41_selected_lowratio_scope_g254:\n    assert self.low_ratio_index_bytes == 68, "Selected native FP4 index pool requires 68 bytes/slot"').body);changes.append('native_FP4_68_unchanged')
    assert changes==['BF16_PA_ND_row','native_FP4_68_unchanged'];fn.body=body
    module=ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[]));context=dict(native.__dict__);context['_ds41_cost_scope_g254']=scope;exec(compile(module,str(root/'native_lowratio_cost_g254.py'),'exec'),context)
    assert not original.__code__.co_freevars and not context['__init__'].__code__.co_freevars
    native.__dict__['_ds41_cost_scope_g254']=scope;original.__code__=context['__init__'].__code__
    prior=cls._get_low_ratio_kv_padding_bytes
    def padding(self,full_tokens):
        if not getattr(self,'_ds41_selected_lowratio_scope_g254',False):return prior(self,full_tokens)
        return sum(low_ratio_padding_bytes(full_tokens,ratio,self.page_size)*self.num_layers(ratio) for ratio in [1,2])
    cls._get_low_ratio_kv_padding_bytes=padding;cls._ds41_selected_lowratio_bytes_g254=True
    (root/f'KV_cost_g254.pid{os.getpid()}.json').write_text(json.dumps({'native_source_sha256':expected,'changes':changes,'low_ratio_padding':'Exact selected PA_ND physical rows plus page64 FP4 reserved rows','unused_INT8_side_buffers_charged':False,'full_engine_capacity_accepted':False},indent=2))
