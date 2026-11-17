"""Prime finite native allocator specializations via Triton warmup; no kernel launch or tensor allocation."""
import os,json,time,datetime,hashlib,itertools
from pathlib import Path
os.sched_setaffinity(0,{188,189});os.nice(10)
import torch
import triton
from triton.runtime.jit import MockTensor
from sgl_kernel_npu.mem_cache.allocator import alloc_extend_kernel
R=Path('/task/allocator_compile_prime_g40');C=Path('/task/sg_pd_g33/cache/triton');assert os.environ['TRITON_CACHE_DIR']==str(C)
class MisalignedPage(MockTensor):
 @staticmethod
 def data_ptr():return 8

def index():return {str(p.relative_to(C)):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in C.rglob('*') if p.is_file()}
def save(name,d):
 p=R/(name+'.tmp');p.write_text(json.dumps(d,indent=2));p.replace(R/(name+'.json'))
start=time.monotonic();cases=[];save('before_cache',index());save('state',dict(phase='compiling_only',pid=os.getpid(),at=datetime.datetime.now(datetime.timezone.utc).isoformat(),bound_s=900,no_kernel_launch=True))
# Native serving cache-key audit recovered page-reserve lengths256/512. Cover128..4096 for every batch bucket and page alignment.
# First full32K D prefill uses the separate native allocator (>=200 new pages), not this JIT branch.
plan=[]
for bs in [1,2,4,8,16]:
 for extend in [128,256,512,1024,2048,4096]:
  for align in [True,False]:plan.append((bs,extend,align))
for bs,extend,align in plan:
 args=[MockTensor(torch.int32),MockTensor(torch.int32),MockTensor(torch.int32),(MockTensor if align else MisalignedPage)(torch.int64),MockTensor(torch.int64),bs,128,extend]
 st=time.monotonic();kernel=alloc_extend_kernel.warmup(*args,grid=(bs,))
 cases.append(dict(bs_upper=bs,max_num_extend_tokens=extend,free_page_aligned16=align,elapsed_s=time.monotonic()-st,kernel_hash=kernel.hash))
 save('state',dict(phase='compiling_only',pid=os.getpid(),completed=len(cases),total=len(plan),elapsed_s=time.monotonic()-start,last_case=cases[-1],no_kernel_launch=True))
save('after_cache',index());save('receipt',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,compile_only=True,kernel_launches=0,tensor_allocations=0,cases=cases,elapsed_s=time.monotonic()-start,scope='Native D page-reserve allocation specialization priming only; not model warmup, numerical proof, or performance result. Does not prove all future hot-prefix branches compiled.'));save('state',dict(phase='compile_only_complete',pid=os.getpid(),at=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.monotonic()-start,completed=len(cases)))
