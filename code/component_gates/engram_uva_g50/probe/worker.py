"""Real NPU/UVA component checks; no serving/performance acceptance."""
from pathlib import Path
import argparse,os,json,hashlib,time,traceback,ctypes
p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);a=p.parse_args();os.sched_setaffinity(0,{168+a.rank});os.environ['OMP_NUM_THREADS']='1'
import torch,torch_npu
torch.set_num_threads(1);torch.npu.set_device(a.rank)
R=Path(__file__).resolve().parents[1];framework=os.environ['PROBE_FRAMEWORK'];started=time.monotonic();cases=[];graphs=[];failure=None
if framework=='sglang':
 import source_uva as module
 def lookup(codes,scales,ids,output,head_start,heads,pad,vocab_start,vocab_end):
  rows=ids.shape[0]*heads
  if not rows:return output
  module._engram_host_uva_gather_dequant_kernel[(rows,)](codes.ptrs,scales.ptrs,ids,output,rows,vocab_start,vocab_end,ids.stride(0),CHUNK=module.CHUNK_ROWS,WIDTH=256,GROUP=32,HEAD_START=head_start,LOCAL_HEADS=heads,PAD_HEADS=pad,QUANTIZED=True,MXFP8=False,num_warps=4)
  return output
else:
 import vllm_ascend.ops  # Match framework custom-op package initialization before device_op.
 from vllm_ascend.models.deepseek_v41.engram import npu as module
 def lookup(codes,scales,ids,output,head_start,heads,pad,vocab_start,vocab_end):
  return module.gather_dequantize_host_uva(codes,scales,ids,head_start=head_start,local_heads=heads,pad_heads=pad,output=output,vocab_start=vocab_start,vocab_end=vocab_end)
host_malloc_arity=None
if framework=='sglang':
 import re
 candidates=[Path('/usr/local/Ascend/ascend-toolkit/latest/include/acl/acl_rt.h'),Path('/usr/local/Ascend/cann-9.1.0/aarch64-linux/include/acl/acl_rt.h')]
 header=next((p for p in candidates if p.is_file()),None);assert header is not None,'ActualCANNheaderrequiredforHostABIcheck'
 match=re.search(r'aclrtMallocHost\s*\(([^;]+?)\)\s*;',header.read_text());assert match,'HostMallocdeclarationnotfound';host_malloc_arity=match[1].count(',')+1;assert host_malloc_arity in [2,3]
 if host_malloc_arity==2:
  lib=module._host_library();raw_malloc=lib.aclrtMallocHost;raw_malloc.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_size_t]
  def compat_malloc(pointer,size,reserved):
   assert reserved==0
   return raw_malloc(pointer,size)
  lib.aclrtMallocHost=compat_malloc
OUT=R/'evidence';OUT.mkdir(exist_ok=True)
def save():
 d={'rank':a.rank,'framework':framework,'device_name':torch.npu.get_device_name(a.rank),'torch':torch.__version__,'torch_npu':torch_npu.__version__,'native_flags':module.DeviceOperator.host_register_flags(),'HostMalloc_header_arity':host_malloc_arity,'source_module':module.__file__,'source_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'cases':cases,'graphs':graphs,'passed':failure is None and len(cases)==10 and len(graphs)==2,'failure':failure,'elapsed_s':time.monotonic()-started,'scope':'SyntheticINT8group32HostUVAlookupandgraphonly;noHCCL/fullmodel/PD/performanceclaim'};p=OUT/f'rank_{a.rank}.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)

def oracle(codes,scales,ids,head_start,heads,pad,start,end):
 out=torch.full((ids.shape[0]*pad,256),17,dtype=torch.bfloat16)
 for token in range(ids.shape[0]):
  for head in range(heads):
   row=int(ids[token,head_start+head]);dest=token*pad+head
   if row<start or row>=end:out[dest].zero_()
   else:
    # Independent float64 product of signed codes and exactly decoded powers of two.
    out[dest]=(codes[row-start].double().reshape(8,32)*scales[row-start].double()[:,None]).reshape(256).bfloat16()
 return out
try:
 for chunk in [1<<22,128]:
  module.CHUNK_ROWS=chunk;codes=None;scales=None
  try:
   codes=module.HostUvaBuffer((2048,256),torch.int8,torch.device('npu',a.rank));scales=module.HostUvaBuffer((2048,8),torch.float32,torch.device('npu',a.rank));g=torch.Generator().manual_seed(1234+a.rank)
   codes.tensor.copy_(torch.randint(-127,128,(2048,256),dtype=torch.int8,generator=g));raw=torch.randint(120,135,(2048,8),dtype=torch.uint8,generator=g)
   # Match checkpoint F8_E8M0 scale semantics without reinterpreting codes as FP8.
   decoded=torch.ldexp(torch.ones((2048,8),dtype=torch.float32),raw.int()-127);scales.tensor.copy_(decoded)
   start=384000000;end=start+2048;head_start=1;heads=3;pad=4
   for tokens in [0,1,3,16,37]:
    hostids=torch.randint(start,end,(max(tokens,1)*2,8),dtype=torch.int64,generator=g)[::2][:tokens]
    if tokens:hostids[0,head_start]=-1;hostids[-1,head_start+heads-1]=end
    ids=torch.empty_strided(hostids.shape,(16,1),dtype=torch.int64,device='npu').copy_(hostids);out=torch.full((tokens*pad,256),17,dtype=torch.bfloat16,device='npu');lookup(codes,scales,ids,out,head_start,heads,pad,start,end);torch.npu.synchronize();expected=oracle(codes.tensor,scales.tensor,hostids,head_start,heads,pad,start,end);actual=out.cpu();assert torch.equal(actual,expected),(chunk,tokens,(actual.float()-expected.float()).abs().max().item())
    cases.append({'chunk_rows':chunk,'tokens':tokens,'ids_stride':list(ids.stride()),'global_vocab_start':start,'owned_heads':heads,'pad_heads':pad,'bitwise_bf16_match':True,'raw_E8M0_sha256':hashlib.sha256(raw.numpy().tobytes()).hexdigest()})
   ids=ids[:16];hostids=hostids[:16];out=torch.full((16*pad,256),17,dtype=torch.bfloat16,device='npu')
   for _ in range(2):lookup(codes,scales,ids,out,head_start,heads,pad,start,end)
   torch.npu.synchronize();graph=torch.npu.NPUGraph()
   with torch.npu.graph(graph):lookup(codes,scales,ids,out,head_start,heads,pad,start,end)
   changed=[]
   for replay in range(3):
    codes.tensor.add_(1);hostids.copy_(torch.randint(start,end,hostids.shape,dtype=torch.int64,generator=g));ids.copy_(hostids);graph.replay();torch.npu.synchronize();expected=oracle(codes.tensor,scales.tensor,hostids,head_start,heads,pad,start,end);actual=out.cpu();assert torch.equal(actual,expected);changed.append(hashlib.sha256(actual.view(torch.uint8).numpy().tobytes()).hexdigest())
   assert len(set(changed))==3;graphs.append({'chunk_rows':chunk,'changed_host_weights_and_NPU_ids_replays':3,'output_sha256':changed,'bitwise_bf16_match':True});del graph;torch.npu.synchronize()
  finally:
   if scales is not None:scales.close()
   if codes is not None:codes.close()
except BaseException as e:failure={'error':repr(e),'traceback':traceback.format_exc()}
save();print(json.dumps({'rank':a.rank,'cases':len(cases),'graphs':len(graphs),'failure':failure}),flush=True);raise SystemExit(failure is not None)
