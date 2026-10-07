"""Execute actual instrumented wrapper AST with CPU tensors and capture doubles."""
import ast,json,sys,tempfile,types,enum,os
from pathlib import Path
import torch
root=Path(sys.argv[1]);destination=Path(sys.argv[2]);source=root/'shim/breakable_aclgraph.py';tree=ast.parse(source.read_text())
class E(enum.Enum):NONE=0;FULL=1
class Extra: is_draft_model=False
extra=Extra();fc=None;sequence=[]
class FakeCapture:
    num_graphs=1;num_eager_breaks=0
    def replay(self):sequence.append('actual_replay')
class Base:
    def __init__(self,runnable,vllm_config):self.runnable=runnable;self.vllm_config=vllm_config;self.entries={}
    def __call__(self,*args,**kwargs):
        if fc.cudagraph_runtime_mode==E.NONE:return self.runnable(*args,**kwargs)
        entry=self.entries.setdefault(fc.batch_descriptor,types.SimpleNamespace(batch_descriptor=fc.batch_descriptor,capture=None,input_addresses=[]))
        return self._capture(entry,args,kwargs) if entry.capture is None else self._replay(entry,args,kwargs)
    def _capture(self,entry,args,kwargs):sequence.append('capture');out=self.runnable(*args,**kwargs);entry.capture=FakeCapture();entry.output=out;return out
    def _replay(self,entry,args,kwargs):entry.capture.replay();return entry.output
class Model:
    def __call__(self,**kw):sequence.append('model');return kw.get('input_ids')
config=types.SimpleNamespace(compilation_config=types.SimpleNamespace(mode=0,cudagraph_mode='FULL_DECODE_ONLY',cudagraph_capture_sizes=[2],max_cudagraph_capture_size=2),additional_config={},model_config=types.SimpleNamespace(enforce_eager=False),speculative_config=types.SimpleNamespace(enforce_eager=True))
# Install only runtime CPU doubles for imports used by the observer.
fdmod=types.ModuleType('vllm.forward_context');fdmod.is_forward_context_available=lambda:True
mod=types.ModuleType('vllm.distributed');mod.get_tp_group=lambda:types.SimpleNamespace(rank_in_group=0)
sys.modules['vllm.forward_context']=fdmod;sys.modules['vllm.distributed']=mod
stream=types.SimpleNamespace(synchronize=lambda:sequence.append('stream_synchronize'))
original_npu=getattr(torch,'npu',None);torch.npu=types.SimpleNamespace(current_stream=lambda:stream)
ns={'Any':object,'Callable':object,'VllmConfig':object,'torch':torch,'CUDAGraphMode':E,'BreakableCUDAGraphWrapper':Base,'get_forward_context':lambda:fc,'_EXTRA_CTX':extra,'weak_ref_workspaces':lambda x:sequence.append('weakref'),'get_graph_params':lambda:None,'get_draft_graph_params':lambda:None,'get_draft_graph_prefill_params':lambda:None,'__name__':'cpu_actual_wrapper'}
filtered=ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) or (isinstance(n,ast.Import) and any(a.name in ('json','os') for a in n.names)) or (isinstance(n,ast.ImportFrom) and n.module=='pathlib')],type_ignores=[])
exec(compile('from __future__ import annotations\n'+ast.unparse(filtered),str(source),'exec'),ns)
class Desc:
    num_tokens=2;num_reqs=1;uniform=True;has_lora=False
bd=Desc();seq=torch.tensor([2],dtype=torch.int32);slot=torch.tensor([1,2],dtype=torch.int32);block=torch.tensor([[5,6]],dtype=torch.int32);local=torch.tensor([2],dtype=torch.int32)
md=types.SimpleNamespace(seq_lens=seq,seq_lens_cpu=seq,cum_query_lens=torch.tensor([2],dtype=torch.int32),slot_mapping=slot,block_table=block,sin=torch.zeros(2,1),cos=torch.ones(2,1),dcp_context=types.SimpleNamespace(seq_lens=local,slot_mapping=slot,block_table=block),attn_state='SpecDecoding',num_input_tokens=2,num_actual_tokens=2,num_decodes=1,num_prefills=0,block_size=128)
fc=types.SimpleNamespace(cudagraph_runtime_mode=E.FULL,batch_descriptor=bd,attn_metadata={'layer':md},capturing=False)
kw={'input_ids':torch.tensor([1,2]),'positions':torch.tensor([0,1])};cases=[]
with tempfile.TemporaryDirectory() as td:
    diag=Path(td);(diag/'witnesses').mkdir();(diag/'scope.txt').write_text('startup\n');os.environ['GLM_GRAPH_DIAGNOSTIC_ROOT']=td
    obj=ns['BreakableACLGraphWrapper'](Model(),config);obj(**kw);assert sequence[:2]==['capture','model'];assert fc.capturing;cases.append('capture invokes model once')
    (diag/'scope.txt').write_text('graph_complete\n');seq.fill_(60);local.fill_(60);slot+=58;sequence.clear();obj(**kw);assert sequence==['stream_synchronize','actual_replay'];cases.append('replay preserves stream sync and skips model')
    seq+=2;local+=2;slot+=2;kw['positions']+=2;sequence.clear();obj(**kw);assert sequence==['stream_synchronize','actual_replay'];cases.append('second dynamic metadata replay')
    row=json.loads((diag/'witnesses/target_rank0.json').read_text());xs=[z for z in row['snapshots'] if z['scope']=='graph_complete'];assert len(xs)==2;assert xs[0]['metadata'][0]['fields']['seq_lens']['first32']==[60];assert xs[1]['metadata'][0]['fields']['seq_lens']['first32']==[62];assert xs[0]['metadata'][0]['fields']['seq_lens']['ptr']==xs[1]['metadata'][0]['fields']['seq_lens']['ptr'];cases.append('snapshot copies mutable values but preserves ptr')
    extra.is_draft_model=True;fc.cudagraph_runtime_mode=E.NONE;draft=ns['BreakableACLGraphWrapper'](Model(),config);sequence.clear();draft(**kw);assert sequence==['model'] and not draft.entries;cases.append('draft NONE eager fallback')
    os.environ.pop('GLM_GRAPH_DIAGNOSTIC_ROOT');sequence.clear();draft(**kw);assert sequence==['model'];cases.append('environment disabled passthrough')
if original_npu is None:del torch.npu
else:torch.npu=original_npu
result={'passed':True,'cases':cases,'case_count':len(cases),'CPU_only':True,'NPU_initialized':False,'actual_AST':str(source)};destination.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
