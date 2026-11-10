#!/usr/bin/env python3
"""Bounded source contracts, not loaded objects, primitive bits or full coverage."""
from __future__ import annotations
import ast, copy, inspect, json, sys
from collections import Counter
from dataclasses import dataclass
from enum import Enum, auto
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace as NS
sys.dont_write_bytecode = True
ROOT = Path('audit/execution-search')

def main():
    inputs = [ROOT / n for n in ['embedding-entry-primary.json', 'embedding-coordinate-dependencies.json', 'embedding-selected-launch.json', 'embedding-quant-description.json', 'whole-forward-frozen-config.json', 'v1-input-source-review.json']]
    primary, coordinates, launch, quant, frozen, previous = [json.loads(p.read_text()) for p in inputs]
    sources = {s['runtime_path']:dict(s) for s in primary['sources'] + coordinates['sources']}
    # The original HF class was already pinned in the preceding source review.
    hf = Path('audit/execution-search/startup-warmup-runtime-source/vllm/vllm/transformers_utils/configs/deepseek_v41.py')
    hfhash = '5e7052bca7ce79dfa801a3bff20fa2c287ec15d82cf999d003f309b30bf6c1f0'
    assert sha256(hf.read_bytes()).hexdigest()==hfhash
    refs = [dict(local_path=str(hf), sha256=sha256(hf.read_bytes()).hexdigest(), provenance='prior exact source, not newly acquired')]
    for s in sources.values(): assert sha256(Path(s['local_path']).read_bytes()).hexdigest() == s['runtime_sha256']
    def path(suffix):
        xs = [s for k,s in sources.items() if k.endswith(suffix)]; assert len(xs)==1, suffix
        return Path(xs[0]['local_path'])
    excerpts=[]
    def node(p, name, owner=None):
        tree=ast.parse(p.read_text())
        xs=tree.body if owner is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==owner).body
        n=next(n for n in xs if getattr(n,'name',None)==name)
        excerpts.append(dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest(),owner=owner,name=name,first=n.lineno,last=n.end_lineno,lines=p.read_text().splitlines()[n.lineno-1:n.end_lineno]))
        return copy.deepcopy(n)
    def execute(nodes, scope):
        mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
        exec(compile(ast.fix_missing_locations(mod),'<original-source-contract>','exec'),scope)
    def method(p,owner,name,scope=None):
        n=node(p,name,owner);n.decorator_list=[];g={} if scope is None else scope;execute([n],g);return g[name]
    def span(p,first,last,label):
        a=p.read_text().splitlines();assert 0<first<=last<=len(a)
        excerpts.append(dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest(),name=label,first=first,last=last,lines=a[first-1:last]))
    for suffix,ranges in [('/worker/model_runner_v1.py',[(3192,3259,'V1 normal/capture caller and executor retirement'),(3616,3653,'MM empty text span handling'),(3825,3905,'per-group alias and final unpadded draft metadata'),(4193,4205,'FULL capture embedding selection')]),('/models/deepseek_v41/model.py',[(1095,1240,'hash/lookup/mask outputs and DP participation'),(1392,1487,'auxiliary producer inputs and stream ownership'),(1518,1592,'SP copies/waits and all first-layer consumers')]),('/spec_decode/dspark_proposer.py',[(126,205,'actual group initialization'),(217,357,'pergroup aliases, common expansion, cad rebinding')]),('/models/deepseek_v41/dspark.py',[(248,275,'draft own lookup/repeat/SP order')]),('/worker/worker.py',[(144,155,'worker custom-op registration')]),('/quantization/configs/modelslim_config.py',[(550,578,'quant prefix mapping'),(749,766,'embedding quant method selection')]),('/models/deepseek_v41/common/mm_preprocess.py',[(251,263,'registered image supported limits')]),('/spec_decode/llm_base_proposer.py',[(378,390,'draft MM signature fallback'),(2653,2669,'physical query/context tail initialization')])]:
        for first,last,label in ranges:span(path(suffix),first,last,label)
    controls=[]
    def check(label,value,expected,qualification='Original source executed with explicit scalar/object stubs; no framework or native admission'):
        assert value==expected,(label,value,expected);controls.append(dict(label=label,value=value,expected=expected,qualification=qualification))

    class ConfigStub:
        def __init__(self,**kwargs): self.__dict__.update(kwargs)
    g={'PretrainedConfig':ConfigStub}; execute([node(hf,'DeepseekV41Config')],g)
    cfg=g['DeepseekV41Config'](**frozen['model_config']['content'])
    check('original_HF_vision_layers',cfg.vision_n_layers,32)
    check('original_HF_mm_prefix',cfg.is_mm_prefix_lm,True)
    convert=method(path('/transformers_utils/model_arch_config_convertor.py'),'ModelArchConfigConvertorBase','is_mm_prefix_lm')
    check('base_convertor_target_MM',convert(NS(hf_config=cfg),True),True)
    check('counterfactual_MM_disabled',convert(NS(hf_config=cfg),False),False)
    limit=method(path('/config/multimodal.py'),'MultiModalConfig','get_limit_per_prompt')
    mm=NS(language_model_only=False,limit_per_prompt={},enable_mm_embeds=False);mm.get_limit_per_prompt=lambda m:limit(mm,m)
    check('original_unspecified_image_limit',limit(mm,'image'),999)
    mm.language_model_only=True;check('counterfactual_language_only_image_limit',limit(mm,'image'),0);mm.language_model_only=False
    logger=NS(debug=lambda *a:None,info_once=lambda *a:None,warning_once=lambda *a:None)
    supports=method(path('/multimodal/registry.py'),'MultiModalRegistry','supports_multimodal_inputs',{'logger':logger})
    reg=NS(_create_processing_info=lambda *a,**k:NS(supported_mm_limits={'image':None}))
    mc=NS(is_multimodal_model=True,get_multimodal_config=lambda:mm,runner_type='generate')
    check('registered_image_source_contract',supports(reg,mc),True)
    mm.limit_per_prompt={'image':NS(count=0)};check('counterfactual_limit_zero',supports(reg,mc),False);mm.limit_per_prompt={}

    # Shared registry: use original PluggableLayer and original CustomOp.register_oot.
    p=path('/model_executor/custom_op.py');pl=node(p,'PluggableLayer');co=node(p,'register_oot','CustomOp')
    reduced=ast.ClassDef(name='CustomOp',bases=[],keywords=[],body=[co],decorator_list=[])
    g={'nn':NS(Module=type('Module',(),{})),'op_registry':{},'op_registry_oot':{},'logger':logger}
    execute([pl,reduced],g)
    Base=type('VocabParallelEmbedding',(g['PluggableLayer'],),{}); OOT=type('AscendVocabParallelEmbedding',(Base,),{})
    check('before_OOT_registration',type(Base()).__name__,'VocabParallelEmbedding')
    g['CustomOp'].register_oot(name='VocabParallelEmbedding')(OOT)
    check('CustomOp_registration_PluggableLayer_construction',type(Base()).__name__,'AscendVocabParallelEmbedding')
    try:g['CustomOp'].register_oot(name='VocabParallelEmbedding')(OOT)
    except AssertionError:duplicate=True
    else:duplicate=False
    check('duplicate_OOT_registration_rejected',duplicate,True)

    p=path('/ops/vocab_parallel_embedding.py');state={'rank':0,'pcp':1,'fine':False}
    g={'Enum':Enum,'auto':auto,'dataclass':dataclass,'get_tp_group':lambda:NS(rank_in_group=state['rank'],world_size=8),
       'get_current_vllm_config':lambda:NS(parallel_config=NS(prefill_context_parallel_size=state['pcp'])),
       'enable_pcp_embedding_lmhead_weight_sharding':lambda:True,'embedding_tp_enable':lambda:state['fine'],'lmhead_tp_enable':lambda:False}
    execute([node(p,n) for n in ['VocabParallelMode','VocabParallelPlan','_resolve_vocab_parallel_plan']],g)
    for rank in range(8):
        state['rank']=rank;plan=g['_resolve_vocab_parallel_plan'](prefix='language_model.model.embed_tokens',disable_tp=False)
        check(f'source_STANDARD_TP8_rank{rank}',[plan.mode.name,plan.shard_rank,plan.shard_world_size],['STANDARD',rank,8])
    plan=g['_resolve_vocab_parallel_plan'](prefix='markov_head.markov_w2',disable_tp=True)
    check('counterfactual_disable_tp_precedes_head_route',[plan.mode.name,plan.shard_world_size],['REPLICATED',1])
    getquant=method(path('/quantization/configs/modelslim_config.py'),None,'get_quant_type_for_layer')
    desc=quant['selected_description']
    for prefix in ['embed','mtp.0.embed','mtp.2.markov_head.embed']:
        check('descriptor_FLOAT_'+prefix,getquant(desc,prefix),None,'Original helper + current filtered descriptor; source prefix mapping reviewed separately, no tensor payload read')

    # Execute original predicates in their own AST, not a rewritten approximation.
    runner=path('/worker/model_runner_v1.py');dummy=node(runner,'_dummy_run','NPUModelRunner')
    predicate=next(n.test for n in ast.walk(dummy) if isinstance(n,ast.If) and n.lineno==4193)
    capture_self=NS(supports_mm_inputs=True,model_config=NS(is_encoder_decoder=False,requires_raw_input_tokens=True),enable_prompt_embeds=False)
    capture_embeds=eval(compile(ast.Expression(predicate),'<original-capture-predicate>','eval'),{'self':capture_self})
    check('actual_capture_uses_persistent_embeddings',capture_embeds,False)
    target=path('/models/deepseek_v41/model.py');forward=node(target,'forward','DeepseekV41Model')
    embed_pred=next(n.test for n in ast.walk(forward) if isinstance(n,ast.IfExp) and n.lineno==1519)
    check('eager_supplied_embeddings_skip_internal_lookup',eval(compile(ast.Expression(embed_pred),'','eval'),{'inputs_embeds':object()}),True)
    check('FULL_capture_None_performs_internal_lookup',eval(compile(ast.Expression(embed_pred),'','eval'),{'inputs_embeds':None}),False)
    capture=method(target,'DeepseekV41Model','can_capture_engram_producer',{'get_pp_group':lambda:NS(world_size=1)})
    c=NS(has_engram=True,_engram_overlap_enabled=True,engram_hash=NS(use_slot_cache=False),use_sequence_parallel=True)
    check('selected_SP_excludes_existing_slotless_capture_helper',capture(c),False,'Original helper eligibility only; does not exclude a new correctly synchronized producer graph')

    # Conservative SSA-free source dependency closure: retain all definitions of a name.
    kp=path('/ops/triton/spec_decode/utils.py');kn=node(kp,'copy_and_expand_dflash_and_dspark_inputs_kernel')
    params={a.arg for a in kn.args.args};defs={};stores=[];parents={c:n for n in ast.walk(kn) for c in ast.iter_child_nodes(n)}
    def names(n):return {x.id for x in ast.walk(n) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Load)}
    for n in ast.walk(kn):
        if isinstance(n,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
            targets=n.targets if isinstance(n,ast.Assign) else [n.target]
            for t in targets:
                if isinstance(t,ast.Name):defs.setdefault(t.id,set()).update(names(n.value))
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='tl' and n.func.attr=='store':stores.append(n)
    def closure(ns):
        seen=set(ns);todo=list(ns)
        while todo:
            for v in defs.get(todo.pop(),set()):
                if v not in seen:seen.add(v);todo.append(v)
        return sorted(seen&params)
    dependencies=[];physical={'context_slot_mapping_ptr','block_table_ptr','block_table_stride','block_size','DCP_SIZE','DCP_RANK','CP_INTERLEAVE_SIZE'}
    common={'out_context_positions_ptr','out_query_positions_ptr','out_input_ids_ptr','out_token_indices_ptr'}
    for n in stores:
        ns=names(n);out=next(v for v in ns if v.startswith('out_') and v.endswith('_ptr'));deps=closure(ns-{out});guards=set();ancestor=parents.get(n)
        while ancestor is not None:
            if isinstance(ancestor,(ast.If,ast.While)):guards.update(names(ancestor.test))
            ancestor=parents.get(ancestor)
        guarddeps=closure(guards)
        if out in common:assert not set(deps)&physical,(out,deps)
        dependencies.append(dict(output=out,line=n.lineno,input_dependencies=deps,lexical_control_parameters=guarddeps,physical_dependency=sorted((set(deps)|set(guarddeps))&physical),qualification='Conservative all-definitions scalar value/address closure plus lexical if/while guards; source classification only, not native memory-alias or scheduling proof'))
    check('kernel_common_fields_have_no_group_physical_dependency',sorted({d['output'] for d in dependencies if not d['physical_dependency']}),sorted(common))
    # The kernel has an anchor followed by noise rows, not five constant IDs.
    check('source_query5_anchor_then_noise',len([d for d in dependencies if d['output']=='out_input_ids_ptr']),1,'Source assignment tl.where(q_idx==0, bonus, parallel_drafting_token_id); no native output test')

    bindings=[]
    def bind(owner,line,state,reason,source):
        match=next(b for b in previous['branches'] if b['root_owner']==owner and b['line']==line)
        assert match['condition_state']=='UNRESOLVED_INPUT_OR_RUNTIME'
        bindings.append(dict(owner=owner,line=line,state=state,reason=reason,source=source,qualification='Frozen source/text load contract; no observed native branch population'))
    bind('DeepseekV41Model',1519,'SOURCE_EAGER_FULL_ENTRY_SPLIT','Normal eager accepts outer embeddings; original FULL capture selects raw IDs + None embeddings. Both entries reviewed; trace captures actual two lookups in FULL.',str(target))
    for line,state,reason in [(3617,'SOURCE_FIXED_TRUE_TARGET_MM','Original HF vision32 sets mm-prefix true; base converter preserves it with supported image limits999.'),(3628,'SOURCE_EMPTY_TEXT_MM_FEATURE_DOMAIN','Frozen pure-text workload has no MM feature iteration; not a statement about model vision capability.'),(3629,'SOURCE_UNREACHED_TEXT_MM_FEATURE_ANCESTOR','No MM features in this selected text workload.'),(3632,'SOURCE_UNREACHED_TEXT_MM_FEATURE_ANCESTOR','No MM features in this selected text workload.'),(3651,'SOURCE_FIXED_FALSE_EMPTY_TEXT_SPANS','Each text request leaves image_doc_ranges empty.'),(3884,'SOURCE_FIXED_TRUE_EMPTY_DICT_NOT_NONE','req_doc_ranges initialized{} under true mm-prefix; preserve final metadata field publication even when dictionary empty.')]:
        bind('NPUModelRunner',line,state,reason,str(runner))
    branches=copy.deepcopy(previous['branches'])
    for b in branches:
        for v in bindings:
            if (b['root_owner'],b['line'])==(v['owner'],v['line']):b.update(condition_state=v['state'],source_binding=v,review_state='BOUNDED_SOURCE_CONTRACT_AND_ENTRY_SPLIT',fusion_search_complete=False)
    pending=sum(b['condition_state']=='UNRESOLVED_INPUT_OR_RUNTIME' for b in branches)
    out=dict(status='SOURCE_EMBEDDING_DISPATCH_CAPTURE_STATIC_FRONTIER',current_distinct_sources=len(sources),sources=list(sources.values()),prior_exact_SHA_references=refs,controls=controls,kernel_output_dependencies=dependencies,new_scoped_source_bindings=bindings,original_site_count=156,scoped_source_bindings=156-pending,unresolved_input_or_control_sites=pending,branch_states=dict(Counter(b['condition_state'] for b in branches)),branches=branches,
        consumers=[dict(field='target input_ids',readers=['outer MM embedding','Engram dead/hash/history','FULL internal embedding','SP shard -> per-layer vision router'],retirement='All producer device reads plus model/draft readers; Python return/cad attribute rebinding is not a device fence'),dict(field='target QSL/physical tables/slots',readers=['global slot/logits prior to FIA overwrite','native first-owner SMLA/QLI/C2','Engram hash slot cache','DSpark input-expansion common coordinates and per-group physical mapping'],retirement='Both producer_done and consumer_done before earliest actual storage write; old alias and new draft cad are distinct'),dict(field='Engram mask/lookup banks',readers=['SP copies mask at entry','layers1/14 shard lookup only after ready event','error/skipped-consumer retirement join'],retirement='Shared persistent capacity bank, graph event wait/reset and previous-main join cannot be removed using only table-ready observation')],
        candidate='Keep larger input-frame/Engram producer delivery parent. Defer SP mask sharding/wait to first actual Engram consumer, make raw coordinates ready early and preserve DP/TP order, integer outputs and bank retirement. Existing slotless producer-capture helper is ineligible on selected SP; that flag alone is not an impossibility proof.',
        remaining=['79 original static sites remain input/control pending; source contracts are not native primitive admissions or full transitive callgraph coverage.','Actual physical draft-group multiplicity/order and final async-reader lifetime, original synthetic-DP0 group, ordinary-load whole static prices and shape/lane/rank keyed cache bitwise proof.','V1 slot-cache hashing and table communication can change fusion boundary; raw ID reader leases and producer/consumer events must precede every storage overwrite.'],
        primitive_precision='Every replacement primitive bitwise identical, including retained state/complete bytes; BF16 last-bit difference fails.',new_primitive_admissions=0,new_framework_replacement_admissions=0,whole_chain_complete=False,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in inputs],reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    assert pending==79 and len(branches)==156 and len(sources)==23
    (ROOT/'embedding-static-source-review.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'embedding-static-source-excerpts.json').write_text(json.dumps(dict(excerpts=excerpts,qualification='Original sources and methods used by bounded controls; full local source hashes included, no framework/native execution.'),ensure_ascii=False,indent=2)+'\n')
    assert 'torch' not in sys.modules
    print(json.dumps(dict(current_sources=len(sources),controls=len(controls),new_bindings=len(bindings),source_bound=156-pending,pending=pending,new_NPU_work=0)))
if __name__=='__main__':main()
