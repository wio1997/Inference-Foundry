#!/usr/bin/env python3
"""Source-scalar controls for nested versus flattened DSpark config fields.

The real config class body uses a pass-through superclass stub. This does not
instantiate Transformers or certify a loaded config object's representation.
"""
from pathlib import Path
from hashlib import sha256
from dataclasses import dataclass,replace
from types import SimpleNamespace as NS
import ast,copy,json
ROOT=Path('audit/execution-search')
def main():
    rows=[]
    for n in ['whole-forward-dependencies.json','whole-forward-binding-dependencies.json']:rows+=json.loads((ROOT/n).read_text())['sources']
    def parsed(suffix):
        r=next(r for r in rows if r['runtime_path'].endswith(suffix));p=Path(r['local_path']);assert sha256(p.read_bytes()).hexdigest()==r['runtime_sha256'];return r,ast.parse(p.read_text())
    cfgrow,t=parsed('/transformers_utils/configs/deepseek_v41.py');cls=copy.deepcopy(next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='DeepseekV41Config'))
    _,t=parsed('/vllm_ascend/utils.py');predicate=copy.deepcopy(next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='is_deepseek_v41'));predicate.decorator_list=[]
    normrow,t=parsed('/patch/platform/patch_speculative_config.py');normalizer=copy.deepcopy(next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='_normalize_deepseek_dspark_draft'));normalizer.decorator_list=[]
    class Base:
        def __init__(self,**kwargs):self.__dict__.update(kwargs)
        def update(self,values):self.__dict__.update(values)
    @dataclass
    class Arch:
        architectures:list
        model_type:str='deepseek_v41'
        is_mm_prefix_lm:bool=True
        num_experts:int=384
        text_model_type:str='deepseek_v41_text'
        num_experts_per_token:int=6
    ns=dict(PretrainedConfig=Base,replace=replace);mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),cls,predicate,normalizer],type_ignores=[]);exec(compile(ast.fix_missing_locations(mod),'<pinned-config-scalar-controls>','exec'),ns)
    frozen=json.loads((ROOT/'whole-forward-frozen-config.json').read_text());raw=frozen['model_config']['content'];before=copy.deepcopy(raw);controls=[]
    for mode in ['nested','original_flat_class_body_with_stub_super']:
        if mode=='nested':hf=Base(**copy.deepcopy(raw));hf.text_config=Base(**copy.deepcopy(raw['text_config']))
        else:hf=ns['DeepseekV41Config'](**copy.deepcopy(raw))
        selected=getattr(hf,'text_config',None) or hf
        initial=dict(experts=selected.n_routed_experts,topk=selected.num_experts_per_tok,vision_n_layers=getattr(selected,'vision_n_layers',0),has_text_config=getattr(hf,'text_config',None) is not None)
        draft=NS(hf_config=hf,model_arch_config=Arch(['DeepseekV41ForCausalLM']),registry=NS(inspect_model_cls=lambda architectures,config:('STATIC_STUB',architectures[0])))
        ns[normalizer.name](draft)
        result=dict(experts=selected.n_routed_experts,topk=selected.num_experts_per_tok,mtp_layers=selected.n_mtp_layers,vision_n_layers=getattr(selected,'vision_n_layers',0),constructor_vision_bias_predicate=getattr(selected,'vision_n_layers',0)>0,arch_is_mm_prefix_lm=draft.model_arch_config.is_mm_prefix_lm,sample_from_anchor_default=getattr(hf,'sample_from_anchor',True))
        assert (result['experts'],result['topk'],result['mtp_layers'])==(128,3,3) and raw==before
        controls.append(dict(mode=mode,initial=initial,result=result,target_raw_unchanged=True))
    assert controls[0]['result']['constructor_vision_bias_predicate']==False and controls[1]['result']['constructor_vision_bias_predicate']==True
    out=dict(status='PASS_TWO_SOURCE_CONFIG_REPRESENTATION_CONTROLS',controls=controls,sources=[dict(runtime_path=r['runtime_path'],sha256=r['runtime_sha256']) for r in [cfgrow,normrow]],parent_dispatch_sha256=sha256((ROOT/'whole-forward-dispatch.json').read_bytes()).hexdigest(),conclusion='Both source representations normalize expert/topK/layer counts, but a flattened class-body path retains vision_n_layers32. Therefore neither text-only DSpark nor arch.is_mm_prefix_lm=False proves gate.bias_vl absent: model.py212 tests vision_n_layers independently and has no is_draft_layer filter. Inspect actual private hf_text_config/gate fields and token predicate before pruning that route. This is a source counter-control, not a loaded-object bug claim.',qualification='Original config __init__ with pass-through PretrainedConfig superclass and original normalizer with registry stub. No actual Transformers initialization/runtime config identity, ownership or native bits established. Private inputs explicitly cloned in controls; target-unchanged is not independent proof of real SpecConfig ownership.',new_NPU_work=0,new_profile=0,service_requests=0,new_framework_replacement_admissions=0,remote_writes=[],reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'whole-forward-config-shape-controls.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(controls=len(controls),vision_predicates=[r['result']['constructor_vision_bias_predicate'] for r in controls],native_admissions=0)))
if __name__=='__main__':main()
