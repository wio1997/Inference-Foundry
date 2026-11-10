#!/usr/bin/env python3
"""Run original prefix-copy, draft-tail and MC2 geometry on row tags only."""
from pathlib import Path
from hashlib import sha256
from types import SimpleNamespace as NS
import ast, copy, json

ROOT = Path('audit/execution-search')

class Rows:
    def __init__(self, values, tail=(), data=None, indices=None):
        self.data = list(values) if data is None else data
        self.indices = list(range(len(self.data))) if indices is None else indices
        self.tail = tail
    @property
    def values(self): return [self.data[i] for i in self.indices]
    @property
    def shape(self): return (len(self.indices), *self.tail)
    def __getitem__(self, key):
        if isinstance(key, int): return self.data[self.indices[key]]
        return Rows([], self.tail, self.data, self.indices[key])
    def __setitem__(self, key, value):
        ids = self.indices[key]
        if isinstance(ids, int): ids = [ids]
        values = value.values if isinstance(value, Rows) else [value] * len(ids)
        assert len(ids) == len(values)
        for i, v in zip(ids, values): self.data[i] = v
    def fill_(self, value): self[:] = value; return self
    def zero_(self): return self.fill_(0)
    def copy_(self, other, **kwargs): self[:] = other; return self
    def pin_memory(self): return self
    def numel(self): return len(self.indices)
    def clamp_(self, min):
        self[:] = Rows([max(min, v) for v in self.values]); return self

def main():
    manifest = json.loads((ROOT / 'row-delivery-dependencies.json').read_text())
    sources = manifest['sources']; records = []; excerpts = []
    for source in sources:
        assert sha256(Path(source['local_path']).read_bytes()).hexdigest() == source['runtime_sha256']
    def node(suffix, owner, name):
        source = next(s for s in sources if s['runtime_path'].endswith(suffix))
        text = Path(source['local_path']).read_text(); tree = ast.parse(text)
        parent = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner) if owner else tree
        selected = next(n for n in parent.body if getattr(n, 'name', '') == name)
        excerpts.append(dict(runtime_path=source['runtime_path'], sha256=source['runtime_sha256'], owner=owner, name=name, first=selected.lineno, last=selected.end_lineno, lines=text.splitlines()[selected.lineno-1:selected.end_lineno]))
        return copy.deepcopy(selected)
    def compile_function(n, globals):
        n.decorator_list = []
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), n], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), '<original-row-delivery>', 'exec'), globals)
        return globals[n.name]
    copy_gpu = compile_function(node('/vllm/v1/utils.py', 'CpuGpuBuffer', 'copy_to_gpu'), dict(PIN_MEMORY=False))
    for pin in [False, True]:
        copy_gpu.__globals__['PIN_MEMORY'] = pin
        obj = NS(cpu=Rows(range(10, 18)), gpu=Rows([0] * 8))
        copy_gpu(obj, 8)
        obj.cpu[:] = Rows(range(20, 28)); copy_gpu(obj, 6)
        assert obj.gpu.values == [20, 21, 22, 23, 24, 25, 16, 17]
        records.append(dict(kind='original_CpuGpuBuffer_copy_to_gpu_8_then6', pin_memory=pin, values=obj.gpu.values, preserved_tail=[16,17]))
    # Select the actual original clamp and padded-position zero statements.
    pre = node('/worker/gpu_model_runner.py', 'GPUModelRunner', '_preprocess')
    statements = [copy.deepcopy(n) for n in ast.walk(pre) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and n.value.func.attr in ['clamp_', 'zero_']]
    assert len(statements) == 2
    g = dict(self=NS(input_ids=NS(gpu=Rows([4, -1, 6, 7, 8, 9, 16, 17]))), num_input_tokens=8, num_scheduled_tokens=6, positions=Rows(range(100,108)))
    g['self'].positions = g['positions']
    exec(compile(ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])), '<two-original-preprocess-statements>', 'exec'), g)
    assert g['self'].input_ids.gpu.values[-2:] == [16,17] and g['positions'].values[-2:] == [0,0]
    records.append(dict(kind='two_original_preprocess_statements', ids=g['self'].input_ids.gpu.values, positions=g['positions'].values, qualification='Only these two original statements, not full preprocess/model or numeric embedding. Position0 does not clear old nonnegative IDs.'))
    pad_draft = compile_function(node('/spec_decode/llm_base_proposer.py', 'AscendSpecDecodeBaseProposer', '_pad_draft_buffers'), {})
    for actual, padded, has_groups in [(20,24,True), (20,20,True), (20,24,False)]:
        draft = NS(input_ids=Rows(range(30)), positions=Rows(range(30)), _slot_mapping_buffer=Rows(range(30)), parallel_drafting_token_id=999, _dflash_num_context=5,
                   _per_group_query_slot_mapping_buffers={12:Rows(range(30))}, _per_group_context_slot_mapping_buffers={12:Rows(range(30))})
        if has_groups: draft._per_group_block_table_buffers = {}
        pad_draft(draft, actual, padded)
        changed = padded > actual and has_groups
        assert draft.input_ids[actual:padded].values == ([999] * (padded-actual) if changed else list(range(actual,padded)))
        if changed:
            assert draft.positions[actual:padded].values == [0]*4
            assert draft._per_group_query_slot_mapping_buffers[12][actual:padded].values == [-1]*4
            assert draft._per_group_context_slot_mapping_buffers[12][5:].values == [-1]*25
        records.append(dict(kind='original_draft_tail_writer', actual=actual, padded=padded, has_per_group_buffers=has_groups, writes_tail=changed, hidden_or_external_embed_writer=False))
    ctx = node('/ascend_forward_context.py', None, 'set_ascend_forward_context')
    chosen = next(n for n in ast.walk(ctx) if isinstance(n, ast.If) and ast.unparse(n.test) == 'reserved_mc2_mask is not None')
    # Execute the complete original reserved-mask branch in the caller's scalar environment.
    def make_mask(actual, padded):
        g = dict(reserved_mc2_mask=Rows([None]*padded), forward_context=NS(padded_num_tokens=padded), num_actual_tokens=actual)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(chosen)], type_ignores=[])), '<original-mask-branch>', 'exec'), g)
        return g['forward_context'].mc2_mask
    def pad(rows, padding, value=0): return Rows(rows.values + [value]*padding[-1], rows.tail)
    def split(rows, size, dim=0):
        assert rows.shape[0] % size == 0
        n=rows.shape[0]//size
        return [rows[i*n:(i+1)*n] for i in range(size)]
    extra = NS()
    g = dict(_EXTRA_CTX=extra, nn=NS(functional=NS(pad=pad)), torch=NS(tensor_split=split), _pad_tokens_with_cat=lambda x,n:Rows(x.values+[0]*(n-x.shape[0]),x.tail), MoEPrepareOutput=lambda **kw:NS(**kw), QuantType=NS(NONE='NONE'))
    prepare = compile_function(node('/ops/fused_moe/prepare_finalize.py', 'PrepareAndFinalizeWithMC2', 'prepare'), g)
    for actual, model_rows, global_padded in [(6,8,8),(20,20,24),(5,8,24),(0,8,24)]:
        local=(model_rows+7)//8
        for rank in range(8):
            extra.mc2_mask=make_mask(actual,global_padded); extra.padded_num_tokens=global_padded
            obj=NS(tp_size=8,tp_rank=rank,lora_context=None)
            out=prepare(obj,Rows(['hidden']*local,(5120,)),Rows(['logits']*local,(384,)),replace_allreduce=True)
            expected=([i < actual for i in range(global_padded)])[rank*local:(rank+1)*local]+[False]*(global_padded//8-local)
            assert out.mc2_mask.values == expected
            records.append(dict(kind='original_context_mask_and_MC2_SP_prepare', actual=actual, model_rows=model_rows, global_padded=global_padded, rank=rank, valid_bits=out.mc2_mask.values, output_rows=out.hidden_states.shape[0]))
    result=dict(status='PASS_ORIGINAL_PREFIX_DRAFT_TAIL_AND_MC2_GEOMETRY', sources=len(sources), controls=len(records), records=records,
                qualification='Integer/boolean/opaque row tags and original Python bodies/identified statements only. No Torch, native bit, loaded object, actual frame population, collective completion or admission.',
                every_replacement_bitwise=True, new_primitive_admissions=0, new_framework_replacement_admissions=0, whole_chain_complete=False,
                new_NPU_work=0, new_profile=0, service_requests=0, remote_writes=[], tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'row-delivery-controls.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'row-delivery-control-source.json').write_text(json.dumps(dict(excerpts=excerpts),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(controls=len(records),native_bit_claim=False,new_admissions=0)))

if __name__ == '__main__': main()
