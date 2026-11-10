#!/usr/bin/env python3
"""Execute exact Python control bodies with opaque launches and row tags, no Torch."""
import ast, copy, json, struct
from pathlib import Path
from hashlib import sha256
from types import SimpleNamespace as NS
R=Path('audit/execution-search'); D=json.loads((R/'compressor-path-dependencies.json').read_text())
def source(suffix):return Path(next(x['local_path'] for x in D['sources'] if x['runtime_path'].endswith(suffix)))
def original(path,name,env):
    n=next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name);n.decorator_list=[]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[n],type_ignores=[])),str(path),'exec'),env);return env[name]
class V:
    def __init__(self,x,shape=None):self.x=list(x) if isinstance(x,(list,tuple,range)) else [x];self.shape=shape or (len(self.x),);self.device='tag-device'
    def __getitem__(self,k):return V(self.x[k]) if isinstance(k,slice) else V(self.x[k],())
    def int(self):return self
    def clamp_max(self,b):return V([min(x,b) for x in self.x])
    def clamp_min(self,b):return V([max(x,b) for x in self.x])
    def remainder(self,b):return V([x%b for x in self.x])
    def op(self,b,f):
        y=b.x if isinstance(b,V) else [b];return V([f(x,y[i if len(y)>1 else 0]) for i,x in enumerate(self.x)])
    def __sub__(self,b):return self.op(b,lambda x,y:x-y)
    def __lt__(self,b):return self.op(b,lambda x,y:x<y)
    def __gt__(self,b):return self.op(b,lambda x,y:x>y)
    def __eq__(self,b):return self.op(b,lambda x,y:x==y)
    def __and__(self,b):return self.op(b,lambda x,y:x and y)
    def reshape(self,*shape):self.shape=shape;return self
    def expand(self,*shape):self.shape=shape;return self
class Buf:
    def __init__(self,n):self.x=['stale']*n;self.shape=(n,1,1,64)
    def __getitem__(self,k):return View(self,k)
    def copy_(self,v):self.x=v.x[:];return self
class View:
    def __init__(self,b,k):self.b=b;self.k=k
    def copy_(self,v):self.b.x[self.k]=v.x;return self
    def reshape(self,*s):return V(self.b.x[self.k],s)
class Table:
    def __init__(self,tag):self.tag=tag;self.shape=(100,1,1,64)
class Blocks:
    def __init__(self,p):self.p=p
    def __getitem__(self,k):return V(self.p[k[0]])
def where(c,a,b):
    aa=a.x if isinstance(a,V) else [a]*len(c.x);bb=b.x if isinstance(b,V) else [b]*len(c.x)
    return V([aa[i] if v else bb[i] for i,v in enumerate(c.x)])
def gather(table,dim,idx,out):assert dim==0;out.copy_(V([(table.tag,p) for p in idx.x]))
def meta_cases():
    torch=NS(arange=lambda n,device=None:V(range(n)),where=where,zeros_like=lambda v:V([0]*len(v.x)),gather=gather)
    path=source('attention/dsa_v41.py');cases=[]
    for name,qsl,seq,pos,pages,reqs,tokens,skip,rope,expected in [
        ('mixed-parity-short-and-append',[0,1,4],[1,6],[0,3,4,5],[7,9],2,4,False,True,[[0,3],[1,3],[0,1],[0,1],[7,9]]),
        ('physical-padding-clamp',[0,3,6],[3,6],[0,1,2,3,4,5],[7,9],1,2,False,True,[[0,3],[2,0],[0,3],[0,3],[7,0]]),
        ('skip-keeps-start-and-bases',[0,1,4],[1,6],[0,3,4,5],[7,9],2,4,True,True,[[0,3],[0,0],[0,1],[0,1],[0,0]]),
        ('no-full-RoPE-table',[0,1],[1],[0],[7],1,1,False,False,[[0],[1],[0],[0],[7]])]:
        n=len(seq);m=len(pos);self=NS(_uses_a5_packed_cache=False,_c2_complete_mask=Buf(m),_c2_source_positions=Buf(m),_c2_source_cos=Buf(m),_c2_source_sin=Buf(m));ring=[Buf(n) for _ in range(5)]
        env=dict(torch=torch,self=self,common=NS(query_start_loc=V(qsl),block_table_tensor=Blocks(pages)),seq_lens=V(seq),input_positions=V(pos),num_reqs=n,num_input_tokens=m,num_actual_reqs=reqs,num_actual_tokens=tokens,skip_ring_update=skip,ring_meta=ring,full_source_cos=Table('cos') if rope else None,full_source_sin=Table('sin') if rope else None)
        original(path,'build_c2_metadata',env)();actual=[r.x for r in ring];assert actual==expected,(name,actual,expected)
        valid_end=min(qsl[reqs],tokens);complete=[not skip and i<valid_end and p%2==1 for i,p in enumerate(pos)];positions=[p-1 if c else 0 for p,c in zip(pos,complete)]
        assert self._c2_complete_mask.x==complete and self._c2_source_positions.x==positions
        assert self._c2_source_cos.x==([('cos',p) for p in positions] if rope else ['stale']*m)
        assert self._c2_source_sin.x==([('sin',p) for p in positions] if rope else ['stale']*m)
        cases.append(dict(name=name,ring_metadata=actual,complete=complete,source_positions=positions,passed=True))
    return cases
class Tensor:
    def __init__(self,shape,dtype,device,contiguous=True):self.shape=shape;self.ndim=len(shape);self.dtype=dtype;self.device=device;self.contiguous=contiguous;self.zeroed=False
    def is_contiguous(self):return self.contiguous
    def zero_(self):self.zeroed=True;return self
    def __getitem__(self,k):return ('metadata-row',k)
class Kernel:
    def __init__(self,name,events):self.name=name;self.events=events
    def __getitem__(self,grid):
        def call(*args,**kwargs):self.events.append(dict(kernel=self.name,grid=grid,kwargs=kwargs))
        return call
def wrappers():
    cases=[];path=source('ops/triton/compressor/compressor_triton.py')
    definitions=[('live',4,2,4,'npu',None,None,True),('tokens-empty',0,2,0,'cpu',None,None,True),('batches-empty',4,0,4,'cpu',None,None,True),('max-query-empty',4,2,0,'cpu',None,None,True),('live-non-NPU',4,2,4,'cpu',None,'NPU',True),('invalid-dtype',4,2,4,'npu','dtype','FP32',False),('invalid-power2-width',4,2,4,'npu','width','power of two',False),('invalid-contiguity',4,2,4,'npu','contiguity','contiguous',False),('invalid-bounds',4,2,5,'npu',None,'bounds',False),('device-mismatch',4,2,4,'npu','device','one device',False)]
    for name,t,b,q,dev,change,error,zero in definitions:
        device=NS(type=dev);w=3 if change=='width' else 512;args=[Tensor((t,w),'float32',device),Tensor((t,w),'float32',device),Tensor((10,32,2*w),'float32',device),Tensor((5,b),'int32',device),Tensor((t,w),'bfloat16',device)]
        if change=='dtype':args[0].dtype='bfloat16'
        if change=='contiguity':args[2].contiguous=False
        if change=='device':args[2].device=NS(type='cpu')
        events=[];env=dict(torch=NS(float32='float32',bfloat16='bfloat16',int32='int32'),_pool_kernel=Kernel('pool',events),_cache_update_kernel=Kernel('update',events));fn=original(path,'compressor_from_projected',env)
        try:ret=fn(*args,max_query_len=q,num_cores=8);assert error is None and ret is args[-1]
        except ValueError as e:assert error and error in str(e),(name,str(e))
        assert args[-1].zeroed==zero
        if name=='live':
            assert [e['kernel'] for e in events]==['pool','update'];assert events[0]['kwargs']['TOKEN_ALIGNED'] and events[0]['kwargs']['eps']==0.;assert events[1]['kwargs']['PROTECT_NULL']
        else:assert not events
        cases.append(dict(name=name,out_zeroed=zero,events=events,expected_error=error,passed=True))
    return cases
def algebra():
    f=lambda x:struct.unpack('f',struct.pack('f',x))[0]
    def bf(x):
        b=struct.unpack('I',struct.pack('f',x))[0];return (b+0x7fff+((b>>16)&1))>>16
    a=f((2**24+2)*2**-20);b=f(-16.);original=f(f(.5*a)+f(.5*b));rewrite=f(a+f(.5*f(b-a)));assert bf(original)!=bf(rewrite)
    return dict(name='equal-score-FP32-weighted-sum-vs-lerp',values=[a,b],original_FP32=original,rewrite_FP32=rewrite,original_BF16_bits=hex(bf(original)),rewrite_BF16_bits=hex(bf(rewrite)),passed=True,scope='Finite IEEE operation-order counterexample, not actual model values or native Triton exp/RMS acceptance. Reject universal algebraic bit identity only.')
def main():
    cases=meta_cases()+wrappers()+[algebra()];out=dict(status='PASS_EXACT_PYTHON_CONTROL_ONLY',cases=cases,count=len(cases),source_body_method='AST original build_c2_metadata and compressor_from_projected bodies; no statement replacement. Opaque kernel launches, scalar/1D row tags and gather row tags. No Torch import or numerical kernel execution.',limits=['Wrapper launch records prove selected flags/order and zero/early-return protocol, not full kernel contents or device synchronization.','A3 metadata row/gather tags do not prove native vector/dtype/capture behavior or full-table bytes.','No new pool/update arithmetic tests: exact-SHA old full-state native cases already cover guarded fusion and alias hazards; reuse rather than rerun unchanged fixtures.'],every_replacement_bitwise=True,new_primitive_admissions=0,new_NPU_work=0,remote_writes=[],whole_chain_complete=False,inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in [source('attention/dsa_v41.py'),source('ops/triton/compressor/compressor_triton.py'),R/'compressor-path-dependencies.json']],probe_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (R/'compressor-path-controls.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(status=out['status'],controls=len(cases),native_admissions=0)))
if __name__=='__main__':main()
