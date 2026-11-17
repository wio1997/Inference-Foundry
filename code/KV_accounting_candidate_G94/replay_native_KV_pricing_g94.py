"""Replay native pricing branches and compare an independent allocation-shape calculation with OOM logs."""
from pathlib import Path
import ast,json,types,hashlib
root=Path(__file__).resolve().parent
source=root/'G86_native_planner_sources/sglang/srt/model_executor/pool_configurator.py'
tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DSV4PoolConfigurator')
names={'_get_paged_kv_bytes_per_token','_compressed_bytes_per_full_token','_get_bytes_per_full_token'}
methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
assert len(methods)==3
unit=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),ast.ClassDef(name='NativePricing',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[]);ast.fix_missing_locations(unit)
context={'_is_npu':True,'get_exec':lambda:types.SimpleNamespace(kernel=types.SimpleNamespace(dsv4_attn_backend='flashmla'))};exec(compile(unit,str(source),'exec'),context)
x=context['NativePricing']();x._unified=False;x.swa_cap_tokens=1;x.swa_ratio=.1;x.bytes_per_swa_token=0;x.stage_owner_layers={2:[2,8,14],1:[20]};x.kv_bytes=584;x.low_ratio_index_bytes=68
before=x._get_bytes_per_full_token()*41/40;assert before==1670.75
x.kv_bytes=512*2;x.low_ratio_index_bytes=68+128+2
after=x._get_bytes_per_full_token()*41/40;assert after==3131.375
allocations=[]
for role,full_tokens,oom_gib in [('P',6032512,2.88),('D',6148992,2.93)]:
    size=full_tokens//2;kernel_page_size=128
    physical_rows=(size+kernel_page_size+1)//kernel_page_size*kernel_page_size
    actual_bytes=physical_rows*512*2
    assert round(actual_bytes/2**30,2)==oom_gib
    allocations.append({'role':role,'native_planned_full_tokens':full_tokens,'C2_size':size,'NPU_physical_rows':physical_rows,'BF16_row_bytes':1024,'actual_single_C2_tensor_bytes_from_native_shape':actual_bytes,'actual_G82_OOM_requested_GiB':oom_gib})
report={'scope':'Actual fixed native pricing methods plus independent integer arithmetic for native NPU allocation shape; no NPU/model or capacity acceptance','native_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'kv_row_bytes_before':584,'actual_BF16_NPU_kv_row_bytes':1024,'low_ratio_indexer_bytes_before':68,'source_derived_FP4_base_plus_NPU_int8_FP16_side_bytes':198,'bytes_per_full_token_before':before,'bytes_per_full_token_after_accounting':after,'actual_OOM_shape_replay':allocations,'full_context_C16_BF16_DSPARK_graph_contract_unchanged':True}
(root/'G94_native_KV_pricing_replay.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
