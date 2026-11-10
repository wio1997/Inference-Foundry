#!/usr/bin/env python3
# coding: utf-8
"""Derive selected roots/MRO/roles before intersecting historical source views.

No framework imports. Scalar originals and Python C3 are independent controls;
branch classifications are bounded source facts, never runtime/bit coverage.
"""
from pathlib import Path
from hashlib import sha256
from types import SimpleNamespace as NS
from dataclasses import dataclass,asdict,replace
from collections import Counter,defaultdict
import ast,copy,json,subprocess
ROOT=Path('audit/execution-search')
PRIOR_BASE='b360ca7c56f773068e91e2da7f4965b79ed3de1f'
def module(path):
    s=path.removeprefix('/vllm-workspace/').split('/')[1:];s[-1]=s[-1].removesuffix('.py')
    if s[-1]=='__init__':s.pop()
    return '.'.join(s)
def main():
    rows={}
    for name in ['whole-forward-dependencies.json','whole-forward-binding-dependencies.json']:
        for r in json.loads((ROOT/name).read_text())['sources']:rows[r['runtime_path']]=r
    units={};classes={};functions={}
    for r in rows.values():
        p=Path(r['local_path']);raw=p.read_bytes();assert sha256(raw).hexdigest()==r['runtime_sha256'];m=module(r['runtime_path']);t=ast.parse(raw);imports={}
        for n in t.body:
            if isinstance(n,ast.Import):
                for a in n.names:imports[a.asname or a.name.split('.')[0]]=a.name if a.asname else a.name.split('.')[0]
            elif isinstance(n,ast.ImportFrom):
                pkg=m if p.name=='__init__.py' else m.rsplit('.',1)[0]
                base=('.'.join(pkg.split('.')[:len(pkg.split('.'))-n.level+1])+('.'+n.module if n.module else '')) if n.level else n.module
                for a in n.names:imports[a.asname or a.name]=(base+'.'+a.name).strip('.')
        units[m]=dict(source=r,tree=t,imports=imports)
        for n in t.body:
            if isinstance(n,ast.ClassDef):classes[m+'.'+n.name]=(m,n)
            elif isinstance(n,ast.FunctionDef):functions[m+'.'+n.name]=(m,n)
    def ref(n,m):
        if isinstance(n,ast.Name):return units[m]['imports'].get(n.id,m+'.'+n.id)
        if isinstance(n,ast.Attribute):return ref(n.value,m)+'.'+n.attr
        return '<dynamic:'+ast.unparse(n)+'>'
    mros={}
    def mro(q):
        if q in mros:return mros[q]
        if q not in classes:return [q]
        m,c=classes[q];bases=[ref(n,m) for n in c.bases];seqs=[mro(b)[:] for b in bases]+[bases[:]];out=[q]
        while any(seqs):
            seqs=[s for s in seqs if s];candidates=[s[0] for s in seqs if not any(s[0] in t[1:] for t in seqs)]
            if not candidates:raise AssertionError(('unresolved C3',q,seqs))
            h=candidates[0];out.append(h)
            for s in seqs:
                if s and s[0]==h:s.pop(0)
        mros[q]=out;return out
    def lookup(q,name,seen=()):
        assert (q,name) not in seen
        for owner in mro(q):
            if owner not in classes:continue
            m,c=classes[owner]
            for n in c.body:
                if isinstance(n,ast.FunctionDef) and n.name==name:return m,n,owner,[]
                if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id==name for v in n.targets):
                    target=ref(n.value,m)
                    if '.' in target:
                        a,b=target.rsplit('.',1);v=lookup(a,b,(*seen,(q,name)))
                        if v:return v[0],v[1],v[2],[dict(owner=owner,line=n.lineno,alias=ast.unparse(n)),*v[3]]
        return None
    D='vllm_ascend.models.deepseek_v41.';C='vllm_ascend.attention.context_parallel.dsa_v41_cp.';P='vllm_ascend.spec_decode.'
    queries=[(D+'vl_model.AscendDeepseekV41ForCausalLM','forward'),(D+'model.AscendDeepseekV41LLMForCausalLM','forward'),(D+'model.DeepseekV41Model','forward'),(D+'dspark.DSparkDeepseekV41ForCausalLM','forward'),(D+'dspark.DeepseekV41DSparkModel','forward'),(D+'dspark.DeepseekV41DSparkDecoderLayer','forward'),(D+'dspark.DeepseekV41DSparkAttention','forward'),(C+'AscendDSAV41CPImpl','forward'),(C+'AscendDSAV41CPImpl','_prepare_inputs_and_caches'),(C+'AscendDSAV41CPImpl','_select_sparse_indices'),(P+'dspark_proposer.AscendDSparkProposer','_propose'),(P+'dspark_proposer.AscendDSparkProposer','set_inputs_first_pass'),(P+'dspark_proposer.AscendDSparkProposer','build_model_inputs_first_pass'),(P+'dspark_proposer.AscendDSparkProposer','_run_merged_draft'),(P+'dspark_proposer.AscendDSparkProposer','dummy_run'),('vllm_ascend.ops.fused_moe.fused_moe.AscendMoERunner','forward')]
    resolutions=[]
    for q,name in queries:
        v=lookup(q,name);assert v,(q,name);m,n,owner,aliases=v;r=units[m]['source'];resolutions.append(dict(receiver=q,method=name,resolved_owner=owner,first=n.lineno,last=n.end_lineno,runtime_path=r['runtime_path'],sha256=r['runtime_sha256'],MRO=mro(q),aliases=aliases,qualification='Source C3/aliases only, not loaded-object inspection.'))
    # Independent Python C3 oracle on the exact seven proposer class headers.
    q=P+'dspark_proposer.AscendDSparkProposer';oracle={}
    def pyclass(q):
        if q in oracle:return oracle[q]
        if q not in classes:return type(q,(),{})
        m,c=classes[q];v=type(q,tuple(pyclass(ref(b,m)) for b in c.bases),{});oracle[q]=v;return v
    python_mro=[v.__name__ for v in pyclass(q).__mro__ if v is not object];assert python_mro==mro(q)
    cfg=json.loads((ROOT/'whole-forward-frozen-config.json').read_text());assert cfg['model_config']['sha256']=='43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7';assert cfg['saved_serve']['sha256']=='c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba'
    rawcfg=cfg['model_config']['content'];text=NS(**copy.deepcopy(rawcfg['text_config']));modelmod=D+'model';util='vllm_ascend.utils';spec='vllm.config.speculative';patch='vllm_ascend.patch.platform.patch_speculative_config'
    selected=[]
    for symbol in [util+'.normalize_deepseek_v41_config',util+'.is_deepseek_v41',modelmod+'._latest_source',modelmod+'.build_layer_plan']:
        selected.append(copy.deepcopy(functions[symbol][1]))
    for name in ['DeepseekV41LayerRole','DeepseekV41Topology']:selected.append(copy.deepcopy(classes[modelmod+'.'+name][1]))
    sc=classes[spec+'.SpeculativeConfig'][1];flags=[copy.deepcopy(n) for n in sc.body if isinstance(n,ast.FunctionDef) and n.name in ['use_eagle','uses_draft_model','use_dspark']]
    selected += flags
    norm=copy.deepcopy(functions[patch+'._normalize_deepseek_dspark_draft'][1]);selected.append(norm)
    for n in selected:
        if isinstance(n,ast.FunctionDef):n.decorator_list=[]
    ns=dict(dataclass=dataclass,replace=replace)
    code=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*selected],type_ignores=[]);exec(compile(ast.fix_missing_locations(code),'<pinned-scalar-definitions>','exec'),ns)
    text=ns['normalize_deepseek_v41_config'](text);roles=[asdict(r) for r in ns['build_layer_plan'](text).layers];flagself=NS(method='dspark');flags_result={n.name:ns[n.name](flagself) for n in flags};assert flags_result==dict(use_eagle=True,use_dspark=True,uses_draft_model=False)
    class Config(NS):
        def update(self,d):self.__dict__.update(d)
    @dataclass
    class Arch:
        architectures:list
        model_type:str
        is_mm_prefix_lm:bool=True
        num_experts:int=384
        text_model_type:str='deepseek_v41_text'
        num_experts_per_token:int=6
    private=Config(**copy.deepcopy(rawcfg));private.text_config=Config(**copy.deepcopy(rawcfg['text_config']));target_before=copy.deepcopy(rawcfg)
    draft=NS(hf_config=private,model_arch_config=Arch(['DeepseekV41ForCausalLM'],'deepseek_v41'),registry=NS(inspect_model_cls=lambda architectures,config:('STATIC_REGISTRY_STUB',architectures[0])))
    ns[norm.name](draft);assert rawcfg==target_before and private.text_config.n_routed_experts==128 and private.text_config.num_experts_per_tok==3 and private.text_config.n_mtp_layers==3
    hw='vllm_ascend.device.hardware_profile';assignments={target.id:n.value for n in units[hw]['tree'].body if isinstance(n,ast.Assign) for target in n.targets if isinstance(target,ast.Name)}
    profiles=next(n.value for n in units[hw]['tree'].body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id=='_HARDWARE_PROFILES').args[0]
    a3=next(v for k,v in zip(profiles.keys,profiles.values) if ast.unparse(k)=='AscendDeviceType.A3')
    def caps(n):
        if isinstance(n,ast.Set):return {ast.unparse(e) for e in n.elts}
        if isinstance(n,ast.Name):return caps(assignments[n.id])
        if isinstance(n,ast.Call) and ast.unparse(n.func)=='frozenset':return caps(n.args[0])
        if isinstance(n,ast.BinOp) and isinstance(n.op,ast.BitOr):return caps(n.left)|caps(n.right)
        raise AssertionError(ast.unparse(n))
    a3caps=caps(next(k.value for k in a3.keywords if k.arg=='capabilities'));assert 'HardwareCapability.ENGRAM_UNROTATED_GATE' not in a3caps
    registry={}
    for n in ast.walk(units['vllm_ascend.models']['tree']):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='register_model' and len(n.args)>=2 and all(isinstance(a,ast.Constant) and isinstance(a.value,str) for a in n.args[:2]):registry[n.args[0].value]=n.args[1].value
    assert registry['DeepseekV41ForCausalLM']==D+'vl_model:AscendDeepseekV41ForCausalLM' and registry['DeepseekV41DSparkModel']==D+'dspark:DSparkDeepseekV41ForCausalLM'
    aux=json.loads((ROOT/'aux-config-call-chain.json').read_text());runner=next(r for r in rows.values() if r['runtime_path'].endswith('/worker/model_runner_v1.py'));assert aux['config_sha256']==cfg['model_config']['sha256'] and next(r for r in aux['source_methods'] if r['path']==runner['runtime_path'])['sha256']==runner['runtime_sha256']
    meta_entry=json.loads((ROOT/'common-device-entry.json').read_text());meta_ref=next(r for r in meta_entry['sources'] if r['runtime_path'].endswith('/worker/device_metadata.py'));assert sha256(Path(meta_ref['local_path']).read_bytes()).hexdigest()==meta_ref['runtime_sha256'];meta_tree=ast.parse(Path(meta_ref['local_path']).read_text());meta_cls=next(n for n in meta_tree.body if isinstance(n,ast.ClassDef) and n.name=='DeviceMetadataExecutor');meta_init=next(n for n in meta_cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__');assert [(a.arg,ast.literal_eval(v)) for a,v in zip(meta_init.args.kwonlyargs,meta_init.args.kw_defaults)]==[('capture_producers',False)]
    facts=[
      dict(id='registry',result=dict(target=registry['DeepseekV41ForCausalLM'],draft=registry['DeepseekV41DSparkModel']),proof='Current plugin literal registration38–58 + frozen architecture + original private DSpark normalization executed without framework imports.'),
      dict(id='spec_flags',result=flags_result,proof='Original current SpeculativeConfig scalar methods executed for method=dspark; successfulPP1 contract implies target raw-MTP buffer allocated despite not being trueMTP.'),
      dict(id='private_draft',result=dict(architectures=private.architectures,experts=private.text_config.n_routed_experts,topk=private.text_config.num_experts_per_tok,layers=private.text_config.n_mtp_layers,target_unchanged=True),proof='Original normalization with privateConfig/stdlibreplace and registry stub; no Transformers/fullModel init or weights.'),
      dict(id='legacy_cp_sp',result=dict(legacyCP=True,PCP=False,sequence_parallel=True),proof='Saved enable_dsa_cp=True + current guard ascend_config802–806 + saved actualCP FTcalls: successful CP requires SP; CP+PCP is rejected by get_v41_cp_classes39–40. Source/observed inference, not loaded config dump.'),
      dict(id='draft_policy',result=dict(method='dspark',parallel_drafting=True,query_positions=5,eager=True,confidence_dynamic_method=None),proof='Savedscript K5/enforce_eager, SpecConfig1461–1462, AscendDSpark31–75 and DynamicSpecConfig1307–1308 defaults (no dynamic config argument).'),
      dict(id='A3_metadata',result=dict(scope='FrozenV1/A3 only',device_adaptor='STANDARD/BaseDeviceAdaptor',packed_cache_ops=None,metadata_in_graph=False,executor_capture_producers=False),proof='SavedV2runner=0; currentV1runner6136 constructs default DeviceMetadataExecutor(capture_producers=False), submits3902 before model forward and requiresExternalEvents3268. SameSHAmetadata executor reference from prior17pin acquisition is validated locally. Basepackedcacheops52–54=None; currentbuilder rejects metadata_in_graph on nonA5. Do not import V2TargetDeviceMetadata(capture_producers=True) policy into this V1 path.'),
      dict(id='mc2_mode',result=dict(saved_mode=1,megamoe_supported=False,shape_comm_selector='UNRESOLVED_PER_CASE'),proof='Current AscendConfig671–673 explicitly rolls mode1 back to own dispatch_ffn_combine. Do not equate flag1 with CANNMegaMoe or make every dynamic transport branch constant.'),
      dict(id='target_roles',result=dict(layers=len(roles),ratios=dict(Counter(r['compress_ratio'] for r in roles)),KV_sources=[r['layer_idx'] for r in roles if r['is_kv_source']],index_sources=[r['layer_idx'] for r in roles if r['is_index_source']],Engram=[r['layer_idx'] for r in roles if r['engram_slot'] is not None],hash_layers=0),proof='Original current normalize/build_layer_plan pure definitions on frozen text config; no nominal model-name assumptions.'),
      dict(id='aux_entries',result=aux['model_entry_layers_zero_based'],proof='Prior pure virtual selector proof only reused after configSHA/currentNPUmethodSHA match; upstream parent source SHA remains archived proof, not newly loaded-object evidence.'),
      dict(id='raw_MTP_bank',result=dict(allocated=True,raw_getter_called=True,raw_numeric_rebind_for_dspark=False,latency_reopen=False),proof='Original spec_flags/use_eagleTrue + model976–988/1595–1604 + runner2056–2060 condition method==mtp. Existing terminal same-scope costs intersected only after derivation, not a new latency opportunity.')]
    facts += [dict(id='A3_capabilities',result=dict(ENGRAM_UNROTATED_GATE=False,RMS_NORM_CAST='HardwareCapability.RMS_NORM_CAST' in a3caps),proof='Pure current HardwareProfile A3 set-union evaluation, not model-name/device-runtime guessing.'),dict(id='selected_V41_cache_builders',result=dict(V41_CP=True,LegacyDSA=False,GDN=False,proposer='AscendDSparkProposer',ubid_at_literal_call=None),proof='Registered target/draft cache backend get_v41_cp_classes returns CP subclass of V41 builder. Literal _build_attn_group_metadata call3876 leaves ubid defaultNone; AscendDSpark C3 proves Eagle/DFlash subtype. Source bindings limited to these frozen text cache-group calls.'),dict(id='decoder_moe_contract',result=dict(fp32_router_input_provided=True,already_sequence_parallel=True,internal_router=True,forward_return='Tensor'),proof='Decoder912–918 always supplies rounded/FP32 outputs plus SP flag; MoE ctor passes nonNonegate, current AscendMoERunner249 property and outer forward215–246 returnTensor after combine/reduction. Source API/type contract, not native bitwise/object dump.'),dict(id='V41_attention_entry',result=dict(output_buffer_provided=True,legacy_CP=True),proof='Selected Attention.forward and dsa_v41_forward56–64 pass explicitoutput; CPimpl inherits base forward. Direct standalone calls are outside this binding.')]
    primary=dict(status='BOUNDED_CURRENT_REGISTRY_MRO_SCALAR_FLAGS_AND_ROLE_BINDINGS',source_count=len(rows),sources=[dict(runtime_path=r['runtime_path'],local_path=r['local_path'],sha256=r['runtime_sha256']) for r in rows.values()],prior_exact_SHA_metadata_reference=dict(runtime_path=meta_ref['runtime_path'],local_path=meta_ref['local_path'],sha256=meta_ref['runtime_sha256'],acquisition='common-device-entry.json',not_reacquired_this_turn=True),registry={k:registry[k] for k in ['DeepseekV41ForCausalLM','DeepseekV41DSparkModel']},resolutions=resolutions,python_C3_oracle=python_mro,scalar_flags=flags_result,target_roles=roles,draft_layers=[dict(layer=i,experts=128,topk=3,Engram=None,compress_ratio=0) for i in range(3)],facts=facts,static_entry_corrections=['DSparkAttention.forward is an assignment alias, not a FunctionDef.','DSparkDecoder.forward is inherited.','Actualrunner2098 calls drafter._propose, resolved through sourceC3 to AscendSpecDecodeBaseProposer1013, not upstreamSpecDecodeBaseProposer.propose.','_runnable binds _run_merged_draft300; DSpark parallel_drafting ends after a single three-body pass plus K5 serial Markov decisions.','LegacyCPImpl.forward is inherited from base; the separately-definedPCP.forward in old18staticroots is outside this frozenmode.'],precision_contract='EVERY_REPLACEMENT_PRIMITIVE_BITWISE_IDENTICAL',new_primitive_admissions=0,new_framework_replacement_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],whole_chain_complete=False,reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'whole-forward-dispatch.json').write_text(json.dumps(primary,ensure_ascii=False,indent=2)+'\n')
    # Only now read historical source-view ranges/cost records for intersections.
    prior_raw=subprocess.check_output(['git','show',PRIOR_BASE+':audit/execution-search/source-frontier.json']);historical=json.loads(prior_raw);range_map=defaultdict(list);schema_counts=Counter();sha_paths=defaultdict(list)
    for r in rows.values():sha_paths[r['runtime_sha256']].append(r)
    def collect(v):
        if isinstance(v,dict):
            path=v.get('runtime_path');h=v.get('sha256',v.get('runtime_sha256'));ranges=v.get('reviewed_ranges',[])
            if not path and isinstance(v.get('path'),str):
                p=v['path']
                if p.startswith('/vllm-workspace/'):path=p
                elif h in sha_paths:
                    candidates=[s for s in sha_paths[h] if Path(s['local_path']).name==Path(p).name]
                    if len(candidates)==1:path=candidates[0]['runtime_path']
            if path and h:
                for r in ranges:
                    if isinstance(r,list) and len(r)==2 and all(isinstance(a,int) for a in r):range_map[(path,h)].append(r);schema_counts['explicit_exact_SHA_range_records']+=1
            elif ranges:schema_counts['unresolved_range_container']+=1
            for x in v.values():collect(x)
        elif isinstance(v,list):
            for x in v:collect(x)
    collect(historical);old=json.loads((ROOT/'whole-forward-entry.json').read_text());branches=[];root_ranges=[]
    for root in old['roots']:
        key=(root['runtime_path'],root['sha256']);intervals=range_map[key];a,z=root['first_line'],root['last_line'];covered=sorted({i for lo,hi in intervals for i in range(max(a,lo),min(z,hi)+1)})
        reach='UNRESOLVED_DYNAMIC_CALLER'
        if root['owner']=='AscendDSAV41PCPImpl':reach='EXCLUDED_BY_FROZEN_LEGACY_CP'
        elif root['owner']!='DeepseekV41MLP':reach='SOURCE_SELECTED_ROOT_OR_DELEGATED_BODY_NOT_ALL_BRANCHES_OBSERVED'
        root_ranges.append(dict(owner=root['owner'],method=root['method'],runtime_path=key[0],sha256=key[1],first=a,last=z,prior_viewed_lines=len(covered),body_lines=z-a+1,unseen_line_count=z-a+1-len(covered),reachability=reach,qualification='Line-range intersection only, not branch/reader/value/native coverage.'))
        for b in root['branch_sites']:
            r=dict(root_owner=root['owner'],root_method=root['method'],runtime_path=key[0],sha256=key[1],**b,prior_source_range_intersects=any(lo<=b['line']<=hi for lo,hi in intervals),selection_state=reach,condition_state='UNRESOLVED_INPUT_OR_RUNTIME',evidence=[],observed_outcomes=[])
            if reach.startswith('EXCLUDED'):r.update(condition_state='NOT_SELECTED_FROZEN_MODE',evidence=['legacy_cp_sp'])
            if root['owner']=='DeepseekV41Model' and root['method']=='forward':
                if b['line'] in [1541,1562,1582,1596]:r.update(condition_state='SOURCE_FIXED_TRUE_WHEN_REACHED',evidence=['legacy_cp_sp'])
                if b['line']==1595:r.update(condition_state='SOURCE_FIXED_TRUE_WHEN_REACHED',evidence=['spec_flags','raw_MTP_bank'])
                if b['line']==1603:r.update(condition_state='SOURCE_FIXED_FALSE_WHEN_REACHED',evidence=['legacy_cp_sp','raw_MTP_bank'])
                if b['line']==1556:r.update(condition_state='SOURCE_FINITE_DOMAIN_40_ROLES',evidence=['target_roles'])
                if b['line']==1560:r.update(condition_state='SOURCE_ROLE_SPLIT_3_TRUE_37_FALSE_WHEN_REACHED',evidence=['target_roles','aux_entries'])
                if b['line']==1565:r.update(condition_state='ROLE_EXCLUDES_38_LAYERS_2_REQUIRE_MASK_NUMEL',evidence=['target_roles'],remaining='token_mask.numel population and producer/event context for layers1/14; all-false mask contents do not imply numel==0.')
            if root['owner']=='DeepseekV41DecoderLayer' and b['line'] in [897,900]:r.update(condition_state='SOURCE_FIXED_TRUE_WHEN_REACHED',evidence=['legacy_cp_sp'])
            if root['owner']=='DeepseekV41DSparkModel' and root['method']=='forward':
                if b['line'] in [251,274]:r.update(condition_state='SOURCE_FIXED_TRUE_WHEN_REACHED',evidence=['legacy_cp_sp','private_draft'])
                if b['line']==263:r.update(condition_state='SOURCE_FINITE_DOMAIN_3_DRAFT_BLOCKS',evidence=['private_draft'])
            if root['owner']=='DeepseekV41MoE' and b['line'] in [277,320]:r.update(condition_state='SOURCE_FIXED_FALSE_FOR_DECODER_CALL',evidence=['legacy_cp_sp'],remaining='Original decoder913–918 passes already_sequence_parallel=True; direct outside-decoder callers are not covered by this binding.')
            if root['owner']=='AscendEngram' and b['line']==42:r.update(condition_state='SOURCE_FIXED_FALSE_WHEN_REACHED',evidence=['A3_capabilities'])
            if root['owner']=='AscendDSAV41Impl' and b['line'] in [694,714]:r.update(condition_state='SOURCE_FIXED_FALSE_FOR_EXPLICIT_OUTPUT_ENTRY' if b['line']==694 else 'SOURCE_FIXED_TRUE_WHEN_REACHED',evidence=['V41_attention_entry'])
            if root['owner']=='DeepseekV41MoE' and b['line'] in [271,279,282,284,292,301,306,323]:
                truth=b['line'] in [271,279,282,306];r.update(condition_state='SOURCE_FIXED_TRUE_FOR_DECODER_API' if truth else 'SOURCE_FIXED_FALSE_FOR_DECODER_API',evidence=['decoder_moe_contract'],remaining='Source API/caller type proof only; no native numerical acceptance.')
            if root['owner']=='NPUModelRunner' and root['method']=='_build_attention_metadata':
                known={3729:False,3765:False,3774:False,3782:True,3790:False,3807:False,3810:True,3843:False,3846:False,3854:True,3860:False,3863:True,3870:False}
                if b['line'] in known:r.update(condition_state='SOURCE_FIXED_TRUE_FOR_V41_CACHE_CALL' if known[b['line']] else 'SOURCE_FIXED_FALSE_FOR_V41_CACHE_CALL',evidence=['selected_V41_cache_builders'])
            branches.append(r)
    # Keep predicate facts separate from branches unreachable under a fixed parent.
    by_root={};by_key={(b['root_owner'],b['root_method'],b['line'],b['kind']):b for b in branches}
    for root in old['roots']:
        mod=module(root['runtime_path']);_,cls=classes[mod+'.'+root['owner']];method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==root['method']);parents={}
        for n in ast.walk(method):
            for child in ast.iter_child_nodes(n):parents[child]=n
        for n in ast.walk(method):
            key=(root['owner'],root['method'],getattr(n,'lineno',None),type(n).__name__);row=by_key.get(key)
            if row is None:continue
            current=n;ancestor=parents.get(current)
            while ancestor is not None:
                if isinstance(ancestor,ast.If):
                    p=by_key.get((root['owner'],root['method'],ancestor.lineno,'If'))
                    if p:
                        state=p.get('predicate_state_before_reachability',p['condition_state']);true=state.startswith('SOURCE_FIXED_TRUE');false=state.startswith('SOURCE_FIXED_FALSE')
                        if (false and current in ancestor.body) or (true and current in ancestor.orelse):row.update(predicate_state_before_reachability=row['condition_state'],condition_state='SOURCE_UNREACHED_BY_FIXED_ANCESTOR',fixed_ancestor=dict(line=ancestor.lineno,state=state));break
                current=ancestor;ancestor=parents.get(current)
    assert len(branches)==156
    for b in branches:
        b['review_state']=('AST_GUARD_INDEX_ONLY_BRANCH_AND_DATAFLOW_NOT_CLOSED' if b['condition_state']=='UNRESOLVED_INPUT_OR_RUNTIME' else 'BOUNDED_SOURCE_BINDING_NOT_NATIVE_OR_FULL_CONSUMER_PROOF')
        b['fusion_search_complete']=False
    selected_gaps=[dict(delivery='Larger A3 common CP frame: rawCPU+scalars+canonicalQSL/device lengths -> originalbase/native producer -> all consumers',priority_basis='Existing same-load inputprepareFree477.800ms/1025.503ms and disjoint CPself/helper/base ledger, not static call counts.',new_scope='FrozenV1 uses executor capture_producers=False and A3 builder metadata_in_graph=False; producer-input addresses and model-graph input addresses must be audited separately. Retaining every unused per-builder QSL is not established as a universal requirement. Consider one existing canonical QSL with original persistent native packet owners; include all stage/key readers, CPU immutable lease and producer/consumer retire/acquire before next write.',missing=['Actual first-owner/all-reader and native/captured model pointer set for canonical QSL, fallback cohorts and exact primitive bytes.','More readers can extend canonical QSL last-read; must preserve prior producer_done + consumer_done + CPU source lease, not release-only/devicewait-only.','Whole paid parent/unprofiled request transfer; previous device-only2.098217% is a subset, not a cap.'],status='待确认，值得深入')]
    terminal=historical['terminal_review']['costs'];delta=dict(status='CURRENT_BINDINGS_THEN_EXACT_SHA_PRIOR_RANGE_INTERSECTION_ONLY',original_static_roots=18,original_callsite_count=295,original_branch_sites=156,branch_states=dict(Counter(b['condition_state'] for b in branches)),branches=branches,root_ranges=root_ranges,remaining_selection_gaps=['Original156site ledger is not transitive callgraph closure: inheritedaliases/proposerhelpers now named, remaining effectivecallee bodies and every Tensor/resource boundary need recursive review.','Many worker batch/metadata/capture/empty/context conditions retain explicit unresolved input/runtime state; source-viewed lines never upgraded to executed/complete.','Visiontower is not in pure-text request execution; token-dependent vision-router bias cannot be excluded merely because prompts are text.','Current saved launch/config do not prove loaded class/env/capture object state; retain plugin/binary/runtime/ordinary-case gaps.'],reused_terminal_budget=dict(source='terminal-cost-replay.json',whole_region=terminal['C8'],ordinary_all_free_max_pct=terminal['ordinary_whole_post_to_head_free_max_pct'],decision='Same exact small latency scope remains stopped; no new NPU repeat. Capacity/larger row-demand parents remain separate.'),next_deliveries=selected_gaps,whole_chain_complete=False,new_framework_replacement_admissions=0,new_primitive_admissions=0,inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in [ROOT/'whole-forward-entry.json',ROOT/'whole-forward-dispatch.json',ROOT/'aux-config-call-chain.json']],qualification='Current primary derivation written first. Historical exactSHA line ranges only intersect afterwards. Static source bindings are not actual branch population/native-bit validation or a guarantee all fusions are found.')
    delta['prior_range_schema_counts']=dict(schema_counts)
    delta['inputs'].append(dict(path='audit/execution-search/source-frontier.json',commit=PRIOR_BASE,sha256=sha256(prior_raw).hexdigest()))
    (ROOT/'whole-forward-coverage-delta.json').write_text(json.dumps(delta,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(sources=len(rows),resolutions=len(resolutions),python_C3=True,roles=len(roles),draft_experts=128,draft_topk=3,static_branches=len(branches),states=delta['branch_states'],rawMTPbank_allocated=True,new_framework_admissions=0,whole_chain_complete=False)))
if __name__=='__main__':main()
