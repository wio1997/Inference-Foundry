"""Exact-source native query-row budget experiment, no weights/model serving."""
from pathlib import Path
import ast,datetime,hashlib,json,os,traceback,types
import torch,torch_npu,msgspec
rank=int(os.environ['LOCAL_RANK']);torch.npu.set_device(rank);os.sched_setaffinity(0,{168+rank});torch.set_num_threads(1);torch.set_grad_enabled(False)
R=Path(__file__).resolve().parent;OUT=R/'evidence';OUT.mkdir(exist_ok=True)
record={'rank':rank,'pid':os.getpid(),'phase':'initializing','cases':[],'scope':'Native-source score/chunk tensor fixture only; no full model, cross-host PD, full quality, graph or performance claim'}
def save():
 p=OUT/f'rank{rank}.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(record,indent=2)+'\n');q.replace(p)
try:
 save();m=json.loads((R/'mechanism.json').read_text())
 for rel,expected in m['source_sha256'].items():assert hashlib.sha256((R/rel).read_bytes()).hexdigest()==expected
 record.update(source_sha256=m['source_sha256'],architecture=os.uname().machine,torch_version=torch.__version__,torch_npu_version=getattr(torch_npu,'__version__','unknown'))
 scoring=ast.parse((R/'native/dsv4/v41_indexer/scoring.py').read_text());sparse=ast.parse((R/'native/dsv4/dsv41_sparse.py').read_text());cls=next(n for n in sparse.body if isinstance(n,ast.ClassDef) and n.name=='DeepseekV41Indexer');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='scores')
 nodes=[n for n in scoring.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ['RequestScores','ChunkScores','_score_chunks']];assert len(nodes)==3
 text='from __future__ import annotations\n'+ast.unparse(ast.Module(body=[method,*nodes],type_ignores=[]))+'\n';(R/f'extracted_native.pid{os.getpid()}.py').write_text(text)
 contexts={}
 for name,budget in [('original',m['original_budget_bytes']),('candidate',m['candidate_budget_bytes'])]:
  ns={'torch':torch,'msgspec':msgspec,'_TORCH_SCORE_BUDGET_BYTES':budget};exec(compile(text,str(R/f'extracted_native.pid{os.getpid()}.py'),'exec'),ns);contexts[name]=ns
 indexer=types.SimpleNamespace();indexer.scores=types.MethodType(contexts['original']['scores'],indexer)
 # Same native tensor operations. No custom attention/selection implementation.
 cases=[(1,127,'bf16'),(16,128,'float32'),(96,129,'bf16'),(127,513,'float32'),(513,4097,'bf16'),(8191,4095,'bf16'),(8193,4097,'float32'),(512,131072,'bf16')]
 heads=m['index_n_heads'];dim=m['index_head_dim'];device=f'npu:{rank}'
 for case,(rows,lc,kind) in enumerate(cases):
  gen=torch.Generator().manual_seed(56300+100*rank+case)
  q=(torch.randn(rows,heads,dim,generator=gen)*.125).bfloat16().to(device);k=(torch.randn(lc,dim,generator=gen)*.125).bfloat16().to(device)
  w=torch.rand(rows,heads,generator=gen).bfloat16().to(device)
  if kind=='float32':w=w.float()
  q0=q.clone();k0=k.clone();w0=w.clone();topk=min(512,lc)
  columns=torch.arange(lc,device=device);tok=torch.arange(rows,device=device);lens=torch.full((rows,),lc,device=device,dtype=torch.int64)
  if lc>512:lens=512+(torch.arange(rows,device=device)%(lc-511))
  outputs={};alloc={};chunkcounts={}
  for name,ns in contexts.items():
   scores=torch.empty((rows,lc),device=device,dtype=torch.float32);indices=torch.empty((rows,topk),device=device,dtype=torch.int64)
   req=ns['RequestScores'](lc=lc,k=topk,columns=columns,tok=tok,lens=lens,slots=columns)
   torch.npu.synchronize();start=int(torch.npu.memory_allocated());torch.npu.reset_peak_memory_stats();count=0
   for chunk in ns['_score_chunks'](indexer=indexer,q=q,weights=w,index_k=k,request=req):
    scores[chunk.rows]=chunk.scores
    indices[chunk.rows]=chunk.scores.topk(req.k,dim=-1,sorted=False).indices
    count+=1
   del chunk
   torch.npu.synchronize();alloc[name]=int(torch.npu.max_memory_allocated())-start;outputs[name]=(scores,indices);chunkcounts[name]=count
  assert torch.equal(outputs['original'][0].view(torch.int32),outputs['candidate'][0].view(torch.int32)),'Native score FP32 bit pattern differs; preserve failure without tolerance changes'
  assert torch.equal(outputs['original'][1],outputs['candidate'][1]),'Native unsortedtopk indices differ; preserve failure'
  assert torch.equal(q,q0) and torch.equal(k,k0) and torch.equal(w,w0),'Inputs mutated'
  large=case in [5,6,7]
  if large:
   assert alloc['candidate']<768*1024**2,('Workingpeakcandidateexceededpreregistered768MiB',alloc)
   assert alloc['original']-alloc['candidate']>1024**3,('No>1GiBworkingpeakreduction',alloc)
   assert chunkcounts['candidate']>chunkcounts['original']
  record['cases'].append({'rows':rows,'lc':lc,'heads':heads,'dim':dim,'q_k_dtype':'bfloat16','weight_dtype':str(w.dtype),'score_FP32_int32_bit_patterns_equal':True,'native_unsorted_topk_indices_equal':True,'inputs_unchanged':True,'working_peak_delta_bytes':alloc,'native_chunk_counts':chunkcounts,'large_peak_gate':large,'scope':'PyTorch activeallocation only; no modelHBM or E2Egain'});save()
  del outputs,q,k,w,q0,k0,w0,scores,indices,req,columns,tok,lens;torch.npu.synchronize()
 record.update(phase='passed',at=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
except BaseException as exc:
 record.update(phase='failed',error=repr(exc),traceback=traceback.format_exc());save();raise
