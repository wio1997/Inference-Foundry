#!/usr/bin/env python3
"""Run the exact original helper AST against an explicit scalar integer model.

No Torch/NPU import. This checks address/memo/key hypotheses, not Torch kernels.
"""
import ast
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import json,struct

ROOT=Path('audit/execution-search')
ORDER={'bool':0,'int32':1,'int64':2}
def cast(v,dtype):
    if dtype=='bool':return bool(v)
    bits=32 if dtype=='int32' else 64
    return (int(v)+(1<<(bits-1)))%(1<<bits)-(1<<(bits-1))
class Vec:
    def __init__(self,values,dtype='int32',storage=None,index=None):
        self.dtype=dtype;self.device=SimpleNamespace(type='cpu')
        self.storage=list(map(lambda x:cast(x,dtype),values)) if storage is None else storage
        self.index=list(range(len(self.storage))) if index is None else index
    def vals(self):return [self.storage[i] for i in self.index]
    def __getitem__(self,key):
        if isinstance(key,slice):return Vec([],self.dtype,self.storage,self.index[key])
        return self.storage[self.index[key]]
    def __setitem__(self,key,value):self[key].copy_(value)
    def op(self,other,fn,boolean=False):
        values=other.vals() if isinstance(other,Vec) else [other]*len(self.index)
        assert len(values)==len(self.index)
        dtype='bool' if boolean else max([self.dtype,other.dtype] if isinstance(other,Vec) else [self.dtype],key=lambda d:ORDER[d])
        return Vec([fn(a,b) for a,b in zip(self.vals(),values)],dtype)
    def __sub__(self,x):return self.op(x,lambda a,b:a-b)
    def __gt__(self,x):return self.op(x,lambda a,b:a>b,True)
    def __and__(self,x):return self.op(x,lambda a,b:bool(a and b),True)
    def copy_(self,x):
        values=x.vals() if isinstance(x,Vec) else list(x)
        assert len(values)==len(self.index)
        for i,v in zip(self.index,values):self.storage[i]=cast(v,self.dtype)
        return self
    def fill_(self,v):return self.copy_([v]*len(self.index))
    def clone(self):return Vec(self.vals(),self.dtype)
    def numel(self):return len(self.index)
    def to(self,dtype):return self if dtype==self.dtype else Vec(self.vals(),dtype)
    def snapshot(self):
        fmt={'bool':'?','int32':'i','int64':'q'}[self.dtype]
        return dict(dtype=self.dtype,values=self.vals(),hex=struct.pack('<'+fmt*len(self.index),*self.vals()).hex())
class ScalarTorch:
    @staticmethod
    def clamp(x,min=None,max=None):
        return Vec([(min if min is not None and v<min else max if max is not None and v>max else v) for v in x.vals()],x.dtype)
    @staticmethod
    def clamp_min(x,min):return ScalarTorch.clamp(x,min=min)
    @staticmethod
    def cumsum(x,dim=0):
        assert dim==0;s=0;values=[]
        for v in x.vals():s=cast(s+v,'int64');values.append(s)
        return Vec(values,'int64')
    @staticmethod
    def tensor(values,dtype,device):return Vec(values,dtype)
    @staticmethod
    def cat(xs,dim=0):
        assert dim==0;dtype=max((x.dtype for x in xs),key=lambda d:ORDER[d])
        return Vec([v for x in xs for v in x.vals()],dtype)
    @staticmethod
    def zeros_like(x):return Vec([0]*x.numel(),x.dtype)
    @staticmethod
    def where(mask,a,b):
        aa=a.vals() if isinstance(a,Vec) else [a]*mask.numel()
        bb=b.vals() if isinstance(b,Vec) else [b]*mask.numel()
        dtype=max([v.dtype for v in [a,b] if isinstance(v,Vec)],key=lambda d:ORDER[d])
        return Vec([x if m else y for m,x,y in zip(mask.vals(),aa,bb)],dtype)

def packed(result):return [x.snapshot() if isinstance(x,Vec) else x for x in result]
def main():
    deps=json.loads((ROOT/'common-frame-key-dependencies.json').read_text())
    row=next(r for r in deps['sources'] if r['runtime_path'].endswith('/dsa_cp.py'))
    p=Path(row['local_path']);body=p.read_text();assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256']
    tree=ast.parse(body);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AscendDSACPMetadataBuilder')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['_local_token_range','_build_local_token_metadata']]
    assert len(methods)==2
    ctx=SimpleNamespace(world_size=8,rank_in_group=0)
    ns={'torch':ScalarTorch,'get_tp_group':lambda:ctx,'HAS_TRITON':False}
    mod=ast.Module(body=[ast.ClassDef(name='OriginalHelper',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod),str(p),'exec'),ns)
    original=ns['OriginalHelper']();cache={};cases=[];duplicates=0
    patterns=[([0],0),([0],24),([0,6],6),([0,6],24),([0,6,12,18,24],24),([0,1,2,20],24),([0,100,3328],3328),([0,3200,6540],6540)]
    for q,n in patterns:
      for prefix in [0,100,32768]:
       lengths=[prefix+q[i+1]-q[i] for i in range(len(q)-1)]
       for noncausal in [False,True]:
        for qdtype in ['int32','int64']:
         for ldtype in ['int32','int64']:
          for world in [1,8]:
           for rank in range(world):
            ctx.world_size=world;ctx.rank_in_group=rank
            key=(len(cases),'target',world,rank,len(q)-1,n,tuple(q),qdtype,tuple(lengths),ldtype,noncausal)
            def run():
                qin=Vec(q,qdtype);lin=Vec(lengths,ldtype)
                out=original._build_local_token_metadata(len(q)-1,n,qin,lin,is_noncausal=noncausal)
                assert out[4].storage is not qin.storage and out[5].storage is not lin.storage
                return out
            first=run();cache[key]=first;before=packed(first)
            # Same CPU computation is reused by 27 illustrative groups; all
            # actual group eligibility remains a separate runtime question.
            for group in range(27):
                reference=packed(run());shared=packed(cache[key]);assert reference==shared
                # CP wrapper's out-of-place clamp/cast must not mutate raw qsl.
                actual=min(q[-1],n);begin=min(first[0],actual);end=min(first[1],actual)
                clamped=ScalarTorch.clamp(first[4],max=end-begin).to('int32')
                assert packed(first)==before;duplicates+=1
            cases.append(dict(key=list(key),raw_result=before,post_qsl_example=clamped.snapshot(),repeat_comparisons=27))
    def sample(q,lengths,n=24,rank=0,world=8,noncausal=False,qdtype='int32',ldtype='int32'):
        ctx.world_size=world;ctx.rank_in_group=rank
        return original._build_local_token_metadata(len(q)-1,n,Vec(q,qdtype),Vec(lengths,ldtype),is_noncausal=noncausal)
    negative=[]
    a=sample([0,24],[100]);b=sample([0,24],[100],rank=1)
    negative.append(dict(recipe='shape_only_ignores_rank',different=packed(a)!=packed(b),before=packed(a),after=packed(b)))
    b=sample([0,24],[90]);negative.append(dict(recipe='same_bucket_ignores_CPU_length_content',different=packed(a)!=packed(b),before=packed(a),after=packed(b)))
    b=sample([0,24],[100],noncausal=True);negative.append(dict(recipe='causal_flag_ignored',different=packed(a)!=packed(b),before=packed(a),after=packed(b)))
    b=sample([0,24],[100],ldtype='int64');negative.append(dict(recipe='raw_output_dtype_ignored',different=packed(a)!=packed(b),before=packed(a),after=packed(b)))
    # CP selects _seq_lens_cpu first, while local_common.replace only updates
    # seq_lens_cpu. Merging those two fields changes the original contract.
    mirror=sample([0,24],[100]);merged=sample([0,24],[90])
    negative.append(dict(recipe='merge_canonical_and_public_CPU_length_fields',different=packed(mirror)!=packed(merged),before=packed(mirror),after=packed(merged)))
    raw=sample([0,24],[100]);full=ScalarTorch.clamp(raw[4],max=3);empty=ScalarTorch.clamp(raw[4],max=0)
    negative.append(dict(recipe='post_frame_ignores_actual_tokens',different=full.snapshot()!=empty.snapshot(),before=full.snapshot(),after=empty.snapshot()))
    # Original optional-out caller writes even when its returned values could
    # match a cache entry; a cache hit that omits writes changes caller state.
    ctx.world_size=8;ctx.rank_in_group=0
    qo=Vec([-7,-7]);lo=Vec([-7]);so=Vec([-7])
    original._build_local_token_metadata(1,24,Vec([0,24]),Vec([100]),qo,lo,so)
    negative.append(dict(recipe='global_method_cache_skips_optional_out_writes',different=qo.vals()!=[-7,-7] and lo.vals()!=[-7] and so.vals()!=[-7],qsl_after=qo.snapshot(),seq_after=lo.snapshot(),start_after=so.snapshot()))
    # These are explicit integer formulas for the CP wrapper's device suffix,
    # not execution of the original device op or a replacement native kernel.
    suffix=24-3;from_upper=max(100-suffix,0);from_device=max(98-suffix,0)
    negative.append(dict(recipe='CPU_raw_key_reused_for_device_authority',different=from_upper!=from_device,CPU_upper_result=from_upper,device_authoritative_result=from_device))
    assert all(r['different'] for r in negative)
    result=dict(status='EXACT_HELPER_AST_SCALAR_INTEGER_MEMO_MODEL',cases=cases,case_count=len(cases),same_key_repeat_comparisons=duplicates,negative_recipes=negative,
                source_path=str(p),source_sha256=row['runtime_sha256'],method_ranges=[[n.lineno,n.end_lineno] for n in methods],
                model_qualification='Original helper AST executed with explicit scalar CPU vector semantics: integer dtype/promotion/slicing/view/mutation. Not real Torch/NPU, not runtime group-argument/alias or graph/primitivebit qualification. 27 repeats are illustrative, not measured exact group identities. No CPUvalues/shape-only hotpath key implementation proposed.',
                tiers=dict(rawCPU='Same invocation epoch/role/actualTPcontext; numReq/numInput, selectedCPUlength andQSL immutablevalues/dtypes, causal and optionaloutmode. For CPcall118 outs areNone; preserve original firstproducer computation.',postCPU='Also actualTokens/clampend/start and destinationQSLdtype; original _seq_lens_cpu precedence retained.',device='Needs authoritative deviceseqlens producer-generation/dependency and capture/outputpointer contract. CPUmirror/equality never substitutes.',resource='Blocktable/slot/cachekind/ratio/ring/outputgroup/descriptor stayperresource; baseobject/dataclassfields preserved.'),
                new_primitive_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'common-frame-key-cpu.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(cases=len(cases),comparisons=duplicates,negative_recipes=len(negative),new_primitive_admissions=0)))
if __name__=='__main__':main()
