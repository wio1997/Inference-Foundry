#!/usr/bin/env python3
"""Bind selected V1 caller constants and Engram handoff, preserving unknowns."""
from pathlib import Path
from hashlib import sha256
from types import SimpleNamespace as NS
from contextlib import nullcontext
from functools import partial
from enum import Enum
from collections import Counter
import ast,json
ROOT=Path('audit/execution-search')
class Mode(Enum): NONE=0; FULL=1
class Tensor:
    shape=(24,5120);device=NS(type='npu')
    def record_stream(self,s):pass
    def new_empty(self,*a,**k):return Tensor()
def main():
    dep=json.loads((ROOT/'v1-input-dependencies.json').read_text());sources={};trees={};lines={}
    for s in dep['sources']:
        p=Path(s['local_path']);raw=p.read_bytes();assert sha256(raw).hexdigest()==s['runtime_sha256']
        sources[s['runtime_path']]=dict(runtime_path=s['runtime_path'],local_path=str(p),sha256=s['runtime_sha256'])
        trees[s['runtime_path']]=ast.parse(raw);lines[s['runtime_path']]=raw.decode().splitlines()
    def path(suf):return next(k for k in sources if k.endswith(suf))
    def evidence(suf,a,z):return dict(runtime_path=path(suf),sha256=sources[path(suf)]['sha256'],first=a,last=z)
    model=path('/models/deepseek_v41/model.py');cls=next(n for n in trees[model].body if isinstance(n,ast.ClassDef) and n.name=='DeepseekV41Model')
    names=['prepare_engram_inputs','prepare_engram_graph_inputs','_can_overlap_engram_preparation']
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(methods)==3
    ctx=NS(cudagraph_runtime_mode=Mode.NONE,batch_descriptor=(24,4))
    stream=lambda:NS(wait_stream=lambda x:None)
    fake=NS(bool=bool,Tensor=Tensor,npu=NS(current_stream=stream,Stream=lambda **k:stream(),Event=lambda:object(),stream=lambda s:nullcontext()))
    ns=dict(torch=fake,partial=partial,CUDAGraphMode=Mode,is_forward_context_available=lambda:True,get_forward_context=lambda:ctx)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),ast.ClassDef(name='Bound',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),model,'exec'),ns)
    controls=[]
    for mode in Mode:
      for enabled in [False,True]:
       for has in [False,True]:
        ctx.cudagraph_runtime_mode=mode;b=ns['Bound']();b.has_engram=has;b._engram_overlap_enabled=enabled;b.engram_rotation=Tensor();b._engram_max_tokens=8192;b._engram_prepare_stream=None;b.config=NS(engram_layer_ids=[1,14])
        b._get_engram_input_buffers=lambda:({1:Tensor(),14:Tensor()},Tensor())
        b._get_engram_external_events=lambda key,prime:(object(),{1:object(),14:object()})
        b.prepare_engram=lambda *a,**k:None
        b.retire_engram_lookups=lambda **k:None
        eager=b.prepare_engram_inputs(Tensor(),Tensor(),24,Tensor())
        capture=b.prepare_engram_graph_inputs(24)
        for entry,result in [('normal_forward',eager),('capture_input_setup',capture)]:
            assert result['engram_lookups'] is not None and 'engram_query_start_loc' not in result
            graph=result.get('engram_graph_events',False);assert not (graph and mode!=Mode.FULL)
            # Original forward's SP wait clears the mask event before the loop.
            event=result.get('engram_mask_ready_event');after_sp=None if event is not None else event
            assert after_sp is None
            controls.append(dict(entry=entry,mode=mode.name,overlap_enabled=enabled,has_engram=has,lookups_are_nonnull=True,
                graph_events=graph,pending_events=bool(result.get('engram_pending')),cleanup_guard_1529=False,mask_event_at_layer_1569=False,query_start_kwarg_emitted=False))
    bindings=[]
    def bind(owner,line,state,why,ev):bindings.append(dict(owner=owner,line=line,state=state,reason=why,evidence=ev,qualification='Selected V1 source/config contract only; no live object, native bit or executed branch population proof.'))
    W='NPUModelRunner.execute_model';M='NPUModelRunner._build_attention_metadata';F='DeepseekV41Model.forward';L='AscendDeepseekV41LLMForCausalLM.forward'
    bind(W,2246,'SOURCE_FIXED_FALSE_SELECTED_DSPARK','Frozen method dspark excludes use_ngram_gpu; no scheduler-output copy for ngram.',[evidence('/worker/model_runner_v1.py',2243,2258)])
    for line,state,why in [(2267,'SOURCE_FIXED_FALSE_PP1','Frozen PP1 excludes nonlast-PP handoff.'),(2270,'SOURCE_UNREACHED_PP1_ANCESTOR','Inside the excluded nonlast-PP branch.'),(2612,'SOURCE_FIXED_TRUE_DSPARK_AUX','DSpark setup sets use_aux_hidden_state_outputs=True.'),(2614,'SOURCE_FIXED_TRUE_PP1','Parent broadcast_pp_output requires PP world size >1; frozen PP1 is false.'),(2616,'SOURCE_FIXED_FALSE_PP1','PP1 is last rank.'),(2640,'SOURCE_UNREACHED_PP1_BROADCAST_ELSE','Else of broadcast_pp_output, excluded by PP1.'),(2649,'SOURCE_UNREACHED_PP1_BROADCAST_ELSE','Else of broadcast_pp_output, excluded by PP1.')]:
        bind(W,line,state,why,[evidence('/worker/model_runner_v1.py',750,764),evidence('/v1/worker/gpu_model_runner.py',557,560),evidence('/worker/model_runner_v1.py',2610,2655)])
    for line,state,why in [(3528,'SOURCE_FIXED_FALSE_EXECUTE_DEFAULT','Actual execute_model call leaves for_cudagraph_capture at False.'),(3540,'SOURCE_FIXED_TRUE_DCP1','Frozen decode_context_parallel_size1, property use_dcp=dcp_size>1; legacy DSA CP is distinct.'),(3544,'SOURCE_UNREACHED_DCP1_ANCESTOR','After early return in _get_dcp_metadata.'),(3546,'SOURCE_UNREACHED_DCP1_ANCESTOR','After early return in _get_dcp_metadata.'),(3589,'SOURCE_FIXED_FALSE_EXECUTE_DEFAULT','Actual execute_model call leaves dcp_dummy_metadata=None.'),(3611,'SOURCE_FIXED_TRUE_ASYNC_SPEC','Parent source derives use_async_spec_decode = use_async_scheduling and num_spec_tokens>0; frozen True/5.'),(3707,'SOURCE_FIXED_FALSE_DCP1','Frozen DCP1 excludes DCP common metadata rewrite.')]:
        bind(M,line,state,why,[evidence('/worker/model_runner_v1.py',2495,2509),evidence('/worker/model_runner_v1.py',3454,3467),evidence('/worker/model_runner_v1.py',708,709),evidence('/v1/worker/gpu_model_runner.py',716,718),evidence('/config/parallel.py',353,355)])
    for line,state,why in [(1520,'SOURCE_FIXED_FALSE_V1_PREPARED_LOOKUPS','_model_forward always overwrites model_inputs with nonnull graph/normal Engram lookup dictionary, including has_engramFalse control.'),(1529,'SOURCE_FIXED_FALSE_V1_SAME_CONTEXT','V1 normal/capture preparation emits graph_eventsTrue only for same FULL context; cleanup mismatch guard cannot be true.'),(1569,'SOURCE_FIXED_FALSE_SELECTED_SP_AFTER_ENTRY_WAIT','Selected SP clears any mask-ready event at1535–1537; no write reintroduces it before layer wait.'),(1606,'SOURCE_FIXED_TRUE_FROZEN_AUX_ROLE_SET','Frozen auxiliary checkpoints append three entering-layer states, including empty token tensors; list length remains three.')]:
        bind(F,line,state,why,[evidence('/worker/model_runner_v1.py',3192,3242),evidence('/models/deepseek_v41/model.py',1392,1503),evidence('/models/deepseek_v41/model.py',1518,1607)])
    bind(L,1711,'SOURCE_FIXED_FALSE_SELECTED_V1_TEXT_KWARGS','V1 _init_model_kwargs emits prompt lookback only for generate; text requests have no multimodal kwargs. _model_forward/Engram preparation never emit engram_query_start_loc; V2 captured-producer branch remains separate.',[evidence('/v1/worker/gpu_model_runner.py',1090,1101),evidence('/v1/worker/gpu_model_runner.py',1712,1734),evidence('/worker/model_runner_v1.py',3192,3242),evidence('/models/deepseek_v41/model.py',1696,1724)])
    bind(L,1712,'SOURCE_UNREACHED_V1_QUERY_KWARG_ANCESTOR','Context body excluded only for selected V1 text entry; do not delete this supported alternative implementation.',[evidence('/models/deepseek_v41/model.py',1711,1724)])
    prior=json.loads((ROOT/'whole-forward-dispatch.json').read_text());spec=next(s for s in prior['sources'] if s['runtime_path'].endswith('/config/speculative.py'))
    assert sha256(Path(spec['local_path']).read_bytes()).hexdigest()==spec['sha256']
    use_ngram=next(n for n in ast.walk(ast.parse(Path(spec['local_path']).read_text())) if isinstance(n,ast.FunctionDef) and n.name=='use_ngram_gpu')
    assert ast.unparse(use_ngram.body[0].value)=="self.method == 'ngram_gpu'"
    bindings[0]['evidence'].append(dict(runtime_path=spec['runtime_path'],sha256=spec['sha256'],first=use_ngram.lineno,last=use_ngram.end_lineno,provenance='Prior exactSHA reference, not newly acquired in16source read'))
    old=json.loads((ROOT/'whole-forward-coverage-delta.json').read_text());branches=old['branches'];lookup={(b['root_owner']+'.'+b['root_method'],b['line']):b for b in branches}
    for r in bindings:
        b=lookup[(r['owner'],r['line'])];assert b['condition_state']=='UNRESOLVED_INPUT_OR_RUNTIME';b['previous_condition_state']=b['condition_state'];b['condition_state']=r['state'];b['V1_binding']=r
    counts=Counter(b['condition_state'] for b in branches);assert len(bindings)==21 and counts['UNRESOLVED_INPUT_OR_RUNTIME']==86
    result=dict(status='BOUNDED_SELECTED_V1_INPUT_AND_SHARED_QSL_OWNER_SOURCE',sources=list(sources.values()),source_count=16,prior_exact_SHA_references=[spec],bindings=bindings,engram_handoff_controls=controls,
        original_site_count=156,prior_source_bindings=49,new_scoped_source_bindings=21,scoped_source_bindings=70,unresolved_input_or_control_sites=86,branches=branches,branch_states=dict(counts),
        input_origin_facts=[
            'Worker1435–1437 publishes unpadded CPU prefix QSL; compute_slot_mapping1622–1626 and logits1645 read its device prefix before later FIA padding1105 publication. First H2D is live.',
            'Worker2495 passes request/global device seq_lens, padded QSL and physical pergroup slot/table through shallow cm copies; CP derives localQSL and authoritative-device suffix lengths; CPU mirrors keep original public/internal differences.',
            'Base batch coordinates expose canonical first QSL, but SMLA1180 and QLI1260 closures capture incoming common QSL, while C2 physical slot1013 and global ring1325–1338 also read it. Owner tests must cover publication and physical-group mapping, not only cached batch.',
            'V1 producer inputs are graph-external and model-graph inputs include canonical coordinate QSL and persistent packet/group IDs. Rebinding incoming commonQSL to canonical may remove nonfirst pin/H2D, but extends canonical producer-reader lifetime.',
            'GPU input-prep CPU event fences previous CPU transfer before next host write; DeviceMetadataExecutor.release is not a join of unconsumed QLI. Shared canonical input requires producer_done and consumer_done/acquire before earliest input write.',
            'Parent _preprocess supports_mm_inputs branch can execute embedding outside model even for text, then copies to persistent input embeddings and passes both embeddings and raw IDs. Model1519 is a caller/config branch, not proof pure text performs embedding inside body.'
        ],
        candidate=dict(classification='待确认，值得深入',scope='Whole canonical rawCPU/scalar/device-QSL and lengths through original physical metadata owners, native packets and model readers',
            mechanism='Share one original local QSL by explicitly rebinding all eligible incoming common.query_start_loc views, retain physical packet and mapping owners; compare larger device-derived QSL/suffix frame only after raw global inputs and earliest writes are audited.',
            missing=['Exact first-owner families and all eager/deferred/captured readers, plus stable source epoch/CPU lease.','Native output bytes, addresses/events, idle/empty/tail/errors and canonical input acquire-before-write.','Paid no-probe whole-parent cost and request transfer; existing independent device-only result is not parent cap.']),
        stopped=[dict(scope='Delete initial global QSL H2D as duplicate of later padding copy',reason='Existing slot mapping and logits consumers precede second copy.'),dict(scope='Delete all nonfirst builder QSL copies without rebinding native incoming common',reason='Native first-key and eager C2 physical-slot readers can retain a noncanonical address.')],
        remaining=['86 original static sites still unclosed; transitive helper branches and resource readers are additional, not counted as86 operators.','supports_mm_inputs/is_mm_prefix_lm, exact physical group order and runtime env/state still require binding; defaults or config stubs are not live objects.','MM embedding-to-persistent-copy and early-input/canonical frame are separate largest deliveries to price, not silently excluded on pure-text label.'],
        qualification='Original three Engram handoff methods with explicit tensor/stream/producer stubs: proves dictionary/control propagation only. No actual hashing, numerical tensors, event safety or Transformers object validation.',
        inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in [ROOT/'v1-input-dependencies.json',ROOT/'whole-forward-coverage-delta.json',ROOT/'whole-forward-frozen-config.json']],
        new_primitive_admissions=0,new_framework_replacement_admissions=0,whole_chain_complete=False,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'v1-input-source-review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(sources=16,new_bindings=len(bindings),remaining=86,engram_controls=len(controls),new_framework_admissions=0)))
if __name__=='__main__':main()
