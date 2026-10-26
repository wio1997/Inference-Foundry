#!/usr/bin/env python3
"""Exact draft methods with CPU tensors; explicit model/projection boundaries.
Historical reference is three steps, not one. No native or output-model proof.
"""
from __future__ import annotations
import ast,contextlib,copy,hashlib,itertools,json,pathlib,sys,types
from typing import Optional,Iterator
from unittest.mock import patch
import torch

R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L029-R003-hidden-cpu'
if '--declarations-only' not in sys.argv:O.mkdir(parents=True,exist_ok=False)
protocol=json.loads((R/'specs/L029-hidden-cpu.json').read_text())
for n,h in protocol['source_hashes'].items():assert hashlib.sha256((R/'sources/installed'/n).read_bytes()).hexdigest()==h
assert torch.__version__=='2.8.0'
g={'torch':torch,'Optional':Optional,'Iterator':Iterator,'contextmanager':contextlib.contextmanager,
   'contextlib':contextlib,'ExitStack':contextlib.ExitStack,'_is_npu':True,'_is_cuda':False,
   '_is_hip':False,'_is_xpu':False,'_DEBUG_ASSERT':False}
trace=[];bodies=[];cases=[];checks=0
def check(condition,label):
    global checks
    checks+=1;assert condition,label
def load(name,method,owner=None,target_name=None):
    p=R/'sources/installed'/name;text=p.read_text();tree=ast.parse(text)
    scope=tree if owner is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==owner)
    node=next(n for n in scope.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==method)
    bodies.append({'path':name,'owner':owner,'name':method,'line':node.lineno,'end_line':node.end_lineno,'complete_body_sha256':hashlib.sha256(ast.get_source_segment(text,node).encode()).hexdigest()})
    m=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),copy.deepcopy(node)],type_ignores=[]);ast.fix_missing_locations(m);exec(compile(m,str(p),'exec'),g)
    if target_name:g[target_name]=g[method]
load('python/sglang/srt/speculative/base_spec_worker.py','_rebuild_topk1_chain_buffers','EagleDraftWorkerBase')
for name in ['_select_top_k_tokens_first','_select_top_k_tokens_later','select_top_k_tokens']:load('python/sglang/srt/speculative/spec_utils.py',name)
load('python/sglang/srt/speculative/eagle_utils.py','per_step_draft_out_cache_loc')
load('python/sglang/srt/layers/attention/index_topk_share.py','IndexTopKShareState')
load('python/sglang/srt/speculative/eagle_worker_v2.py','_draft_forward_idle','EagleDraftWorker')
load('python/sglang/srt/speculative/eagle_worker_v2.py','draft_forward','EagleDraftWorker')
load('python/sglang/srt/models/deepseek_nextn.py','forward','DeepseekModelNextN','nextn_forward')
load('python/sglang/srt/managers/overlap_utils.py','_resolve_spec_extras','FutureMap')
if '--declarations-only' in sys.argv:
    print(json.dumps({'declarations':'PASS','bodies':len(bodies),'semantic_checks':0}));sys.exit(0)

g['get_exec']=lambda:types.SimpleNamespace(graph=types.SimpleNamespace(cuda_graph_config=types.SimpleNamespace(decode=types.SimpleNamespace(max_bs=8))))
g['get_schedule']=lambda:types.SimpleNamespace(max_running_requests=8)
config=types.SimpleNamespace(speculative_use_rejection_sampling=False)
g['get_spec']=lambda:config
g['maybe_detect_nan']=lambda *a,**kw:None
g['maybe_detect_inf']=lambda *a,**kw:None
g['maybe_detect_oob']=lambda *a,**kw:None
g['ForwardContext']=lambda **kw:types.SimpleNamespace(**kw)
g['forward_context']=lambda ctx:contextlib.nullcontext()
class ModelBoundary(Exception):pass
class ProjectionBoundary(Exception):pass
class Mode:
    def __init__(self,idle=False):self.idle=idle
    def is_idle(self):return self.idle
    def is_extend(self,**kw):return False
    def is_decode(self):return True
def worker(steps,tokens):
    s=types.SimpleNamespace(topk=1,speculative_num_steps=steps,speculative_num_draft_tokens=tokens,
        server_args=types.SimpleNamespace(),device='cpu',hot_token_id=None,
        index_share_for_mtp_iteration=True,seed_dsa_topk_from_draft_extend=True,
        draft_attn_backend=types.SimpleNamespace(attn_backends=[object()]*max(steps-1,1)))
    s._draft_forward_idle=types.MethodType(g['_draft_forward_idle'],s)
    def model_boundary(fb):
        trace.append({'model_boundary':True,'hidden_none':fb.spec_info.hidden_states is None,
                      'hidden_shape':None if fb.spec_info.hidden_states is None else list(fb.spec_info.hidden_states.shape)})
        raise ModelBoundary('actual model not executed')
    s.draft_runner=types.SimpleNamespace(forward=model_boundary,canary_manager=None,
        model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(architectures=['DeepseekV3ForCausalLMNextN']),model_is_mrope=False))
    g['_rebuild_topk1_chain_buffers'](s);return s
def batch(bs,steps,hidden):
    di=types.SimpleNamespace(topk_p=torch.ones((bs,1)),topk_index=torch.arange(bs).view(bs,1)+3,
        hidden_states=hidden,draft_probs=torch.ones((bs,4))/4,dsa_topk_indices=torch.ones((bs,2048),dtype=torch.int32))
    return types.SimpleNamespace(spec_info=di,forward_mode=Mode(bs==0),batch_size=bs,
        input_ids=torch.arange(bs),out_cache_loc=torch.arange(bs*steps),positions=torch.arange(bs),
        reuse_dsa_topk_indices=False)

for steps,tokens in [(3,4),(1,2)]:
    s=worker(steps,tokens)
    check(s._topk1_parents_prealloc.shape==(8,steps if steps>1 else 0),'actual chain buffer dimensions')
    check(s._topk1_score_indices_prealloc.shape==(8,steps),'actual score dimensions')
    cases.append({'kind':'VALID_CHAIN','steps':steps,'draft_tokens':tokens})
try:worker(1,4)
except AssertionError:check(True,'one step cannot retain reference draft4 under topk1 chain')
else:raise AssertionError('invalid chain metadata accepted')
cases.append({'kind':'INVALID_CHAIN','steps':1,'draft_tokens':4,'source_rejects':True})

first=g['_select_top_k_tokens_first']
def traced_first(p,index,hidden,k):
    out=first(p,index,hidden,k)
    trace.append({'CPU_hidden_repeat':hidden is not None,
                  'new_CPU_storage':False if hidden is None else out[1].data_ptr()!=hidden.data_ptr()})
    return out
g['_select_top_k_tokens_first']=traced_first
def outputs_equal(a,b):
    return all(x is None and y is None or isinstance(x,torch.Tensor) and isinstance(y,torch.Tensor) and torch.equal(x,y) for x,y in zip(a,b))
for bs,rejection,hot in itertools.product([2,3],[False,True],[False,True]):
    config.speculative_use_rejection_sampling=rejection;outputs=[];records=[]
    for present in [True,False]:
        trace.clear();s=worker(1,2);s.hot_token_id=torch.arange(32).flip(0) if hot else None
        hidden=torch.arange(bs*6144,dtype=torch.float32).view(bs,6144) if present else None
        fb=batch(bs,1,hidden);out=g['draft_forward'](s,fb);outputs.append(out);records.append(list(trace))
        check(not any(isinstance(t,dict) and t.get('model_boundary') for t in trace),'step1 makes no model call in this exact method')
        expected=torch.arange(bs).view(bs,1)+3
        if hot:expected=s.hot_token_id[expected]
        check(torch.equal(out[2],expected),'actual one-step token/topology construction')
        check(out[0].shape==(bs,0) and out[1].shape==(bs,1),'one-step chain shape')
        check((out[3] is not None)==rejection,'rejection proposal presence')
        check(fb.reuse_dsa_topk_indices is False and fb.spec_info.dsa_topk_indices is None,'real index share context cleanup')
    check(outputs_equal(*outputs),'conditional single-step outputs independent of hidden input')
    check(records[0][0]['CPU_hidden_repeat'] and records[0][0]['new_CPU_storage'] and not records[1][0]['CPU_hidden_repeat'],'CPU repeat copy removed only for None input')
    cases.append({'kind':'CONDITIONAL_STEP1','bs':bs,'rejection_fixture':rejection,'hot_map_fixture':hot,
                  'steps':1,'draft_tokens':2,'hidden_None_outputs_equal':True,'reference_configuration':False,
                  'prepare_ForwardBatch_graph_not_executed':True})
config.speculative_use_rejection_sampling=False
for present in [True,False]:
    fb=batch(0,1,torch.empty((0,6144)) if present else None);out=g['draft_forward'](worker(1,2),fb)
    check(out==(None,None,None,None),'one-step idle body returns empty topology')
cases.append({'kind':'STEP1_IDLE','outputs_empty':True})

for bs,present in itertools.product([2,3],[True,False]):
    trace.clear();fb=batch(bs,3,torch.ones((bs,6144)) if present else None)
    try:g['draft_forward'](worker(3,4),fb)
    except ModelBoundary:check(True,'reference step3 reaches model input boundary')
    else:raise AssertionError('expected model boundary')
    entry=next(t for t in trace if isinstance(t,dict) and t.get('model_boundary'))
    check(entry['hidden_none']==(not present),'exact worker forwards hidden argument to model')
    check(not fb.reuse_dsa_topk_indices,'index state restored after model boundary exception')
    cases.append({'kind':'REFERENCE_STEP3_MODEL_BOUNDARY','bs':bs,'hidden_present':present,
                  'entry':entry,'is_native_success':False,'full3step_outputs_not_tested':True})

g['ReplicatedLinear']=type('ReplicatedLinearBoundary',(),{})
g['BumpAllocator']=lambda **kw:types.SimpleNamespace()
for bs,present in itertools.product([2,3],[True,False]):
    projection_inputs=[]
    def projection(x):projection_inputs.append(x.clone());raise ProjectionBoundary('native projection not executed')
    model=types.SimpleNamespace(quant_config=object(),rot_weight=None,
        enorm=torch.nn.RMSNorm(6144,eps=1e-6,elementwise_affine=False),
        hnorm=torch.nn.RMSNorm(6144,eps=1e-6,elementwise_affine=False),eh_proj=projection)
    fb=batch(bs,3,torch.ones((bs,6144)) if present else None)
    try:g['nextn_forward'](model,fb.input_ids,fb.positions,fb,input_embeds=torch.ones((bs,6144)))
    except ProjectionBoundary:
        check(present and projection_inputs[0].shape==(bs,12288),'real source cat has hidden input before projection boundary')
    except TypeError:
        check(not present and not projection_inputs,'None rejected by explicit CPU RMSNorm boundary before projection')
    else:raise AssertionError('expected projection or missing tensor boundary')
    cases.append({'kind':'NEXTN_SOURCE_INPUT','bs':bs,'hidden_present':present,
                  'CPU_RMSNorm_boundary':True,'native_projection_not_executed':True})

# Counterexample to assuming the gate alone clears a preexisting hidden field.
g['gather_spec_extras']=lambda idx,p,index,bonus,hidden:(p[idx],index[idx],bonus[idx],None)
raw_hidden=torch.ones((3,6144));di=types.SimpleNamespace(future_indices=torch.tensor([3,1]),draft_probs=None,
    hidden_states=raw_hidden,future_dsa_topk_indices_available=False)
f=types.SimpleNamespace(spec_algo=types.SimpleNamespace(is_ngram=lambda:False),device='cpu',need_topk=True,
    need_hidden_states=False,topk_p_buf=torch.ones((8,1)),topk_index_buf=torch.ones((8,1),dtype=torch.long),
    output_tokens_buf=torch.ones(8,dtype=torch.long),draft_probs_buf=None,dsa_topk_indices_buf=None)
with patch.object(torch,'get_device_module',return_value=types.SimpleNamespace(current_stream=lambda:object())),patch.object(torch.Tensor,'record_stream',lambda *a,**kw:None):
    g['_resolve_spec_extras'](f,types.SimpleNamespace(spec_info=di))
check(di.hidden_states is raw_hidden and di.hidden_states.shape[0]==3 and di.topk_p.shape[0]==2,'disabling relay gate alone retains raw preexisting hidden field')
cases.append({'kind':'GATE_FALSE_RETAINS_RAW_HIDDEN','new_topk_rows':2,'raw_hidden_rows':3,
              'native_gather_replaced_by_explicit_CPU_boundary':True,'does_not_prove_current_worker_bug':True})

result={'status':'PASS','semantic_checks':checks,'case_count':len(cases),'cases':cases,'source_bodies':bodies,
    'source_commit':protocol['source_commit'],'source_hashes':protocol['source_hashes'],'torch':torch.__version__,
    'historical_reference_steps':3,'one_step_is_reference_optimization':False,
    'source_model_and_projection_boundaries':True,'whole_modules_imported':False,'native_executed':False,
    'candidate_accepted':False,'accepted_baseline':None,'performance':'NOT_MEASURED',
    'limitations':['Step1 numerical equality only covers exact draft_forward/selected helpers, not prepare/ForwardBatch/graph/PD/full model.',
                   'Step3 stops at the first model boundary; no claim of actual native model count or full3step outputs.',
                   'NextN source uses explicit CPU RMSNorm and projection boundary, not actual trained norms/weights or NPU projection.',
                   'Historical three-step snapshot is not a fresh task-owned runtime baseline.']}
(O/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','semantic_checks','case_count','native_executed']}))
