#!/usr/bin/env python3
"""Original relay constructors/stash/producer with real CPU tensors.
Native gather, stream/event and worker outputs are explicit boundaries.
"""
from __future__ import annotations
import ast,copy,dataclasses,enum,hashlib,itertools,json,pathlib,sys,types
from typing import Any,Optional,Sequence
from unittest.mock import patch
import msgspec
import torch

R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L028-R003-payload-cpu';O.mkdir(parents=True,exist_ok=False)
protocol=json.loads((R/'specs/L028-payload-cpu.json').read_text())
for n,h in protocol['source_hashes'].items():assert hashlib.sha256((R/'sources/installed'/n).read_bytes()).hexdigest()==h,n
assert torch.__version__=='2.8.0' and msgspec.__version__=='0.19.0'
module=types.ModuleType('L028_exact_relay');sys.modules[module.__name__]=module;g=module.__dict__
trace=[];bodies=[];cases=[];checks=0
g.update(torch=torch,msgspec=msgspec,dataclass=dataclasses.dataclass,Optional=Optional,Sequence=Sequence,Any=Any,Enum=enum.Enum,auto=enum.auto,_is_npu=True,_is_cuda=False,_is_hip=False,_DEBUG_ASSERT=False)
def check(value,label):
    global checks
    checks+=1;assert value,label
def load(name,wanted,owner=None):
    p=R/'sources/installed'/name;text=p.read_text();tree=ast.parse(text)
    scope=tree if owner is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==owner)
    nodes=[]
    for key in wanted:
        node=next(n for n in scope.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==key)
        bodies.append({'path':name,'owner':owner,'name':key,'line':node.lineno,'end_line':node.end_lineno,'complete_AST_body':True,'body_sha256':hashlib.sha256(ast.get_source_segment(text,node).encode()).hexdigest()});nodes.append(copy.deepcopy(node))
    m=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[]);ast.fix_missing_locations(m);exec(compile(m,str(p),'exec'),g)

load('python/sglang/srt/speculative/spec_info.py',['SpeculativeAlgorithm'])
g['get_spec']=lambda:types.SimpleNamespace(speculative_algorithm='EAGLE3',enable_multi_layer_eagle=False)
load('python/sglang/srt/speculative/spec_utils.py',['spec_need_hidden_states'])
spec_boundary=types.ModuleType('sglang.srt.speculative.spec_utils');spec_boundary.spec_need_hidden_states=g['spec_need_hidden_states']
for name in ['sglang','sglang.srt','sglang.srt.speculative']:
    if name not in sys.modules:
        pkg=types.ModuleType(name);pkg.__path__=[];sys.modules[name]=pkg
sys.modules[spec_boundary.__name__]=spec_boundary
load('python/sglang/srt/managers/overlap_utils.py',['ResolvedConfidence','RelayPayload','ConfidenceRelay','FutureMap','resolve_forward_inputs'])
load('python/sglang/srt/managers/scheduler.py',['_relay_forward_payload'],owner='Scheduler')
FutureMap,Payload,Algo=g['FutureMap'],g['RelayPayload'],g['SpeculativeAlgorithm'];algo=Algo.EAGLE3
check(algo.need_topk() and g['spec_need_hidden_states'](),'actual EAGLE3 policy needs topk and hidden')

class EventBoundary:
    def record(self,*a,**kw):trace.append('event.record')
    def wait(self,*a,**kw):trace.append('event.wait')
class DeviceBoundary:
    Event=EventBoundary
    @staticmethod
    def current_stream():return object()
class NativeGatherBoundary(Exception):pass
def stop_gather(*args):raise NativeGatherBoundary('native Triton gather not executed')
def gather_oracle(indices,p,index,bonus,hidden):
    trace.append('CPU_GATHER_ORACLE_NOT_NATIVE')
    return p[indices],index[indices],bonus[indices],None if hidden is None else hidden[indices]
g['gather_spec_extras']=gather_oracle
def new_map():
    pool=types.SimpleNamespace(req_to_token=torch.zeros((8,64),dtype=torch.int32))
    return FutureMap(torch.device('cpu'),algo,pool,needs_cpu_seq_lens=True,needs_confidence_relay=False)
def draft(slots,epoch,dsa=True):
    rows=torch.tensor(slots,dtype=torch.int64)
    bonus=(rows+epoch*100).to(torch.int32)
    hidden=(rows[:,None]*10+epoch+torch.arange(6144)[None,:]%8).float()
    return types.SimpleNamespace(bonus_tokens=bonus,topk_p=(rows[:,None]+epoch).float()/16,
        topk_index=rows[:,None]+epoch*10,hidden_states=hidden,
        draft_probs=(rows[:,None]+torch.arange(4)[None,:]+epoch).float(),
        dsa_topk_indices=(rows[:,None]*100+torch.arange(2048)[None,:]+epoch).to(torch.int32) if dsa else None)
def relay(f,di,slots):
    owner=types.SimpleNamespace(spec_algorithm=algo,future_map=f,chunked_req=None,
        beam_coordinator=types.SimpleNamespace(maybe_select_and_relay=lambda *a,**kw:trace.append('beam-boundary')))
    batch=types.SimpleNamespace(beam_tail=None)
    g['_relay_forward_payload'](owner,batch,torch.tensor(slots),types.SimpleNamespace(next_draft_input=di))
def receiver(f,slots,available=True,prob=True):
    interleaved=torch.empty(len(slots)*2,dtype=torch.int64);interleaved[::2]=torch.tensor(slots);interleaved[1::2]=777
    indices=interleaved[::2]
    di=types.SimpleNamespace(future_indices=indices,future_dsa_topk_indices_available=available,draft_probs=torch.empty(0) if prob else None)
    return types.SimpleNamespace(prefill_input_ids_cpu=None,input_ids=None,spec_info=di,
        enable_overlap=True,spec_algorithm=algo,device='cpu',req_pool_indices=indices)
def equal(actual,expected,label):check(torch.equal(actual,expected.to(actual.dtype)),label)
fields=['bonus_tokens','topk_p','topk_index','hidden_states','draft_probs','dsa_topk_indices']
def snapshot(di):return {n:None if getattr(di,n) is None else getattr(di,n).clone() for n in fields}
bufs={'bonus_tokens':'output_tokens_buf','topk_p':'topk_p_buf','topk_index':'topk_index_buf','hidden_states':'hidden_states_buf','draft_probs':'draft_probs_buf','dsa_topk_indices':'dsa_topk_indices_buf'}

with patch.object(torch,'get_device_module',return_value=DeviceBoundary),patch.object(torch.Tensor,'record_stream',lambda self,stream:trace.append('record_stream-boundary')):
    for perm,keep_optional in itertools.product(itertools.permutations([1,2,3]),[False,True]):
        trace.clear();f=new_map()
        check(isinstance(f.confidence_relay,g['ConfidenceRelay']),'real msgspec Struct constructed')
        cold=types.SimpleNamespace(bonus_tokens=torch.tensor([11,12,13]),topk_p=None,topk_index=None,hidden_states=None,draft_probs=None,dsa_topk_indices=None)
        relay(f,cold,[1,2,3]);check(not f.need_topk and not f.need_hidden_states,'cold bonus-only does not initialize extras')
        di=draft([1,2,3],1);want=snapshot(di)
        f.publish(torch.tensor([1,2,3]),torch.tensor([101,202,303],dtype=torch.int32))
        equal(f.output_tokens_buf[[1,2,3]],cold.bonus_tokens,'early length publish did not write final bonus')
        check(f.hidden_states_buf is None,'early publish lacks final hidden payload')
        relay(f,di,[1,2,3])
        for n in fields:equal(getattr(f,bufs[n])[[1,2,3]],want[n],'stash '+n)
        for n in fields:getattr(di,n).fill_(-9)
        for n in fields:equal(getattr(f,bufs[n])[[1,2,3]],want[n],'stash snapshot survives producer mutation '+n)
        batch=receiver(f,perm);g['resolve_forward_inputs'](batch,f)
        selection=torch.tensor([slot-1 for slot in perm])
        for n in fields:equal(getattr(batch.spec_info,n),want[n][selection],'reordered consumer '+n)
        # Full cancellation scheduling is not executed: survivors are a boundary input.
        survivors=receiver(f,[3,1]);g['resolve_forward_inputs'](survivors,f)
        equal(survivors.spec_info.hidden_states,want['hidden_states'][[2,0]],'survivor hidden rows')
        old_slot_seed=f.dsa_topk_indices_buf[2].clone()
        reused=draft([2],2,dsa=keep_optional);new=snapshot(reused);relay(f,reused,[2])
        batch2=receiver(f,[2,3,1],available=keep_optional,prob=keep_optional);g['resolve_forward_inputs'](batch2,f)
        equal(batch2.spec_info.bonus_tokens,torch.tensor([202,103,101]),'reused slot bonus refreshed')
        equal(batch2.spec_info.hidden_states[0],new['hidden_states'][0],'reused slot hidden refreshed')
        if keep_optional:
            equal(batch2.spec_info.dsa_topk_indices[0],new['dsa_topk_indices'][0],'new seed reaches reused slot')
            equal(batch2.spec_info.draft_probs[0],new['draft_probs'][0],'draft probs requested')
        else:
            check(batch2.spec_info.dsa_topk_indices is None and batch2.spec_info.draft_probs is None,'consumer absence remains absent')
            equal(f.dsa_topk_indices_buf[2],old_slot_seed,'old seed remains in pool but availability suppresses it')
        check(any(x=='CPU_GATHER_ORACLE_NOT_NATIVE' for x in trace),'explicit gather reference boundary reached')
        cases.append({'kind':'PAYLOAD_CPU','permutation':list(perm),'optional_on_reuse':keep_optional,
                      'hidden_width':6144,'seed_width':2048,'topk_width':1,
                      'CPU_tensor_producer_and_stash':True,'native_consumer_gather':False,'full_cancel_scheduler':False})

    f=new_map();di=draft([1],1);relay(f,di,[1]);g['gather_spec_extras']=stop_gather
    try:g['resolve_forward_inputs'](receiver(f,[1]),f)
    except NativeGatherBoundary:check(True,'unreplaced consumer stops at native gather boundary')
    else:raise AssertionError('expected native boundary')
    cases.append({'kind':'NATIVE_GATHER_BOUNDARY','native_executed':False,'is_native_success':False})
    g['gather_spec_extras']=gather_oracle
    before={n:getattr(f,b)[1].clone() for n,b in bufs.items()}
    f.stash(torch.empty(0,dtype=torch.int64),Payload(bonus_tokens=None))
    for n,b in bufs.items():equal(getattr(f,b)[1],before[n],'empty DP payload leaves written row '+n)
    cases.append({'kind':'DP_EMPTY_STASH','empty_payload_read':False})
    try:f.stash(torch.tensor([1]),Payload(bonus_tokens=torch.tensor([77])))
    except AttributeError:check(True,'after lazy init caller must supply required topk/hidden fields')
    else:raise AssertionError('expected incomplete warm payload contract rejection')
    cases.append({'kind':'INCOMPLETE_WARM_PAYLOAD','AttributeError_expected':True,
                  'actual_worker_emits_this_payload_proven':False,'transactional_rollback_tested':False})

result={'status':'PASS','semantic_checks':checks,'case_count':len(cases),'cases':cases,
    'source_commit':protocol['source_commit'],'source_hashes':protocol['source_hashes'],'source_bodies':bodies,
    'torch':torch.__version__,'msgspec':msgspec.__version__,'CPU_real_constructors_and_tensors':True,
    'whole_modules_imported':False,'gather_reference_is_explicit_boundary':True,
    'native_gather_boundary_stops':1,'native_boundary_is_success':False,'native_executed':False,
    'candidate_accepted':False,'accepted_baseline':None,'performance':'NOT_MEASURED',
    'limitations':['Consumer fused Triton gather is not executed; CPU gather is an independent mapping oracle boundary.',
                   'Device event/record_stream, get_spec configuration, beam coordinator and worker output are explicit boundaries.',
                   'Full request cancellation/slot generation, KV allocation/free, PD transfer, NPU streams and native timing not executed.',
                   'Cold/warm optional payload contract cases do not prove actual worker emits incomplete payload.',
                   'L002 length-only324cases not repeated or re-counted.']}
(O/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','semantic_checks','case_count','native_executed']}))
