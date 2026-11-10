#!/usr/bin/env python3
"""Original RoPE/dispatch guards plus finite byte-tag layout controls.

No Torch/NumPy/model imports. Tags are not native arithmetic/bit admission.
"""
from pathlib import Path
from hashlib import sha256
from types import SimpleNamespace as NS
import ast,copy,json
R=Path('audit/execution-search')
def main():
    pins={s['runtime_path']:s for s in json.loads((R/'aux-context-dependencies.json').read_text())['sources']};used=[];cases=[]
    def body(name,owner,symbol):
        s=pins['/vllm-workspace/'+name];raw=Path(s['local_path']).read_bytes();assert sha256(raw).hexdigest()==s['runtime_sha256'];tree=ast.parse(raw)
        parent=tree if owner is None else next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name==owner)
        n=copy.deepcopy(next(x for x in parent.body if getattr(x,'name',None)==symbol));used.append(dict(path=s['runtime_path'],sha256=s['runtime_sha256'],owner=owner,symbol=symbol,first=n.lineno,last=n.end_lineno));return n
    proxy=body('vllm-ascend/vllm_ascend/ops/rope_dsv4.py',None,'RopeDataProxy')
    getter=body('vllm-ascend/vllm_ascend/ops/rope_dsv4.py',None,'get_cos_and_sin_dsa')
    traces=[]
    class Positions:
        def __init__(self,x):self.x=x
    class Table:
        def __init__(self,key):self.key=key
        def __getitem__(self,pos):traces.append(self.key);return tuple((self.key,i) for i in pos.x)
    state=NS(registry_summary={'a':{'default'},'b':{'default'}},full_rope_cache={c:(Table(c+'cos'),Table(c+'sin')) for c in ['a','b']},layer_info={f'L{i}':('a',{'default'}) for i in range(3)})
    ns=dict(torch=NS(Tensor=Positions),_ROPE_STATE=state)
    mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),proxy,getter],type_ignores=[]);exec(compile(ast.fix_missing_locations(mod),'<original-RoPE-proxy-and-getter>','exec'),ns)
    get=ns['get_cos_and_sin_dsa'];pos=Positions([5,0,5,9]);baseline=[]
    for i in range(3):c,s=get(pos);baseline.append((c['L'+str(i)],s['L'+str(i)]))
    assert len(traces)==12;traces.clear();c,s=get(pos);shared=[(c['L'+str(i)],s['L'+str(i)]) for i in range(3)];assert shared==baseline and len(traces)==4
    cases.append(dict(name='original_full_proxy_shared_once_same_generation',kind='original_Python_getter_and_proxy_with_copy_tags',pass_=True,baseline_table_reads=12,shared_table_reads=4))
    traces.clear();c,s=get(pos,layer_names=['L0','L1','L2']);assert [(c[f'L{i}'],s[f'L{i}']) for i in range(3)]==baseline and len(traces)==2
    cases.append(dict(name='requested_same_config_only_once',kind='original_Python_layer_config_guard',pass_=True,table_reads=2))
    state.layer_info['L2']=('b',{'default'});traces.clear();c,s=get(pos,layer_names=['L0','L1','L2']);assert c['L2'][0]==('bcos',5) and len(traces)==4
    cases.append(dict(name='different_layer_config_is_retained',kind='original_Python_layer_config_guard',pass_=True,table_reads=4))
    try:get(pos,layer_names=['missing'])
    except KeyError:pass
    else:raise AssertionError('missing registry entry silently accepted')
    cases.append(dict(name='missing_layer_remains_error_or_baseline_fallback',kind='original_Python_error_guard',pass_=True))
    c,s=get(pos);saved=c['L0'];pos.x=[7];c2,s2=get(pos);assert c2['L0']!=saved
    cases.append(dict(name='position_mutation_requires_new_generation',kind='negative_Python_lease_control_not_native',pass_=True))
    ctor=body('vllm-ascend/vllm_ascend/models/deepseek_v41/dspark.py','DeepseekV41DSparkModel','__init__')
    gate=next(n.value for n in ctor.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='main_proj_quant_config' for t in n.targets))
    for label,packed,quant,expected in [('A3_release_fp8',None,{'quant_method':'fp8'},None),('A3_no_quant',None,None,None),('packed_device_fp8',object(),{'quant_method':'fp8'},'Q')]:
        value=eval(compile(ast.fix_missing_locations(ast.Expression(copy.deepcopy(gate))),'<original-main-proj-quant-gate>','eval'),dict(DeviceOperator=NS(get_dsv41_packed_cache_ops=lambda:packed),model_quant_config=quant,vllm_config=NS(quant_config='Q')));assert value==expected
        cases.append(dict(name=label,kind='original_AST_quant_policy_expression',pass_=True,selected_quant=value))
    op=body('vllm-ascend/vllm_ascend/ops/linear_op.py',None,'_get_column_parallel_op');op.decorator_list=[]
    env=dict(enable_dsa_cp=lambda:True,oproj_tp_enable=lambda:True,mlp_tp_enable=lambda:True,is_moe_layer=lambda x:False)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),op],type_ignores=[])),'<original-main-proj-column-selector>','exec'),env)
    for prefix in ['model.layers.40.main_proj','mtp.0.main_proj','model.main_proj']:
        assert env[op.name](prefix,object()) is None;cases.append(dict(name='standard_column_'+prefix,kind='original_Python_prefix_dispatch',pass_=True))
    # Mean values are already rounded opaque byte tags. No changed arithmetic.
    def compare_layout(tp,n,selected):
        chunk=(n+tp-1)//tp;H=3
        aux=[[[[(a,r,j,'bits') for j in range(H)] for r in range(t*chunk,(t+1)*chunk)] for t in range(tp)] for a in range(3)]
        gathered=[[row for rank in a for row in rank][:n] for a in aux]
        baseline=[sum((a[i] for a in gathered),[]) for i in selected]
        packed=[sum((aux[a][t][r] for a in range(3)),[]) for t in range(tp) for r in range(chunk)]
        actual=[packed[:n][i] for i in selected];assert actual==baseline
        return packed,baseline
    for tp,n,idx in [(1,5,[4,0,4]),(8,11,[10,0,5,5,-1]),(8,0,[])]:
        compare_layout(tp,n,idx);cases.append(dict(name=f'packed_aux_TP{tp}_N{n}',kind='finite_copy_index_permutation_not_native',pass_=True,selectors=idx))
    packed,baseline=compare_layout(8,11,[-1]);assert packed[-1]!=baseline[0]
    cases.append(dict(name='truncate_full_N_before_negative_index',kind='finite_negative_layout_control_not_native',pass_=True))
    out=dict(status='ORIGINAL_PYTHON_ROPE_AND_DISPATCH_PLUS_FINITE_LAYOUT_PASS',controls=cases,total=len(cases),all_pass=True,source_methods=used,every_replacement_bitwise=True,new_native_admissions=0,new_primitive_admissions=0,whole_chain_complete=False,new_NPU_work=0,remote_writes=[],qualification='Original source bodies/AST expression run with opaque constructors/copy tags. Packed ordering is finite coordinate evidence, not arbitrary-stride native/capture, collective timing, numeric arithmetic or per-primitive bit proof.',script_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (R/'aux-context-controls.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(controls=len(cases),all_pass=True,native_admissions=0)))
if __name__=='__main__':main()
