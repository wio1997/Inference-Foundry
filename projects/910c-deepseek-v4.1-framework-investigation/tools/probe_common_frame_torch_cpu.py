#!/usr/bin/env python3
"""Private CPU-only primitive/cost probe; never import a serving framework."""
import ast,json,os,sys,time,sysconfig,gc
from pathlib import Path
from hashlib import sha256
from types import SimpleNamespace
from statistics import median

def main():
    root=Path(sys.argv[1]);spec=json.loads((root/'spec.json').read_text())
    assert str(root).startswith('/data/tiankuan/wio/')
    (root/'started.json').write_text(json.dumps(dict(pid=os.getpid(),status='RUNNING_CPU_ONLY',start_ns=time.time_ns()))+'\n')
    assert os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']=='0'
    purelib=Path(sysconfig.get_path('purelib'));init=purelib/'torch/__init__.py'
    assert sha256(init.read_bytes()).hexdigest()==spec['torch_init_sha256']
    sys.path.insert(0,str(purelib))
    import torch
    def guard():
        assert not any(n=='torch_npu' or n.startswith('torch_npu.') or n=='vllm' or n.startswith('vllm.') or n=='transformers' for n in sys.modules)
        maps=Path('/proc/self/maps').read_text()
        banned=[s for s in ['libtorch_npu','libascendcl','libhccl'] if s in maps]
        assert not banned,banned
        return dict(backend_autoload=False,torch_npu_imported=False,serving_framework_imported=False,forbidden_NPU_libraries=banned)
    before=guard();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    source=Path(spec['helper_source_path']);code=source.read_text();assert sha256(source.read_bytes()).hexdigest()==spec['helper_source_sha256']
    tree=ast.parse(code);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AscendDSACPMetadataBuilder')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['_local_token_range','_build_local_token_metadata']]
    assert len(methods)==2;group=SimpleNamespace(world_size=8,rank_in_group=0)
    ns={'torch':torch,'get_tp_group':lambda:group,'HAS_TRITON':True}
    mod=ast.Module(body=[ast.ClassDef(name='OriginalHelper',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod),str(source),'exec'),ns);original=ns['OriginalHelper']()
    class CPUFrame:
        """One caller-supplied immutable input lease, not a global memo table.

        Tensor versions do not detect NumPy writes. Eligibility requires the
        runner's closed invocation: no writer until the frame is retired.
        """
        def __init__(self,epoch):self.epoch=epoch;self.key=None;self.value=None;self.inputs=None
        def call(self,nreq,ninput,q,l,local_query_start_loc=None,local_seq_lens=None,start_pos_out=None,is_noncausal=False):
            if q.device.type!='cpu' or l.device.type!='cpu' or any(v is not None for v in [local_query_start_loc,local_seq_lens,start_pos_out]):
                return original._build_local_token_metadata(nreq,ninput,q,l,local_query_start_loc,local_seq_lens,start_pos_out,is_noncausal)
            key=(self.epoch,id(q),id(l),q._version,l._version,q.dtype,l.dtype,tuple(q.shape),tuple(l.shape),group.world_size,group.rank_in_group,nreq,ninput,bool(is_noncausal))
            if self.key is None:
                # Private CPU snapshots protect the first computation; they do
                # not make arbitrary later mutations of caller inputs eligible.
                self.inputs=(q,l,q.clone(),l.clone());self.key=key
                self.value=original._build_local_token_metadata(nreq,ninput,self.inputs[2],self.inputs[3],is_noncausal=is_noncausal)
                return self.value
            if key!=self.key:return original._build_local_token_metadata(nreq,ninput,q,l,is_noncausal=is_noncausal)
            return self.value
    def packet(v):
        if isinstance(v,torch.Tensor):
            assert v.device.type=='cpu'
            return (str(v.dtype),tuple(v.shape),v.contiguous().numpy().tobytes())
        return v
    def equal(a,b):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert packet(x)==packet(y)
    def digest(values):
        h=sha256()
        for v in values:
            p=packet(v)
            if isinstance(p,tuple):h.update(str(p[:2]).encode());h.update(p[2])
            else:h.update(str(p).encode())
        return h.hexdigest()
    patterns=[([0],0),([0],24),([0,6],6),([0,6],24),([0,6,12,18,24],24),([0,1,2,20],24),([0,100,3328],3328),([0,3200,6540],6540)]
    records=[];comparisons=0
    for q,ninput in patterns:
      for prefix in [0,100,32768]:
       lv=[prefix+q[i+1]-q[i] for i in range(len(q)-1)]
       for noncausal in [False,True]:
        for qdtype in [torch.int32,torch.int64]:
         for ldtype in [torch.int32,torch.int64]:
          for world in [1,8]:
           for rank in range(world):
            group.world_size=world;group.rank_in_group=rank
            qi=torch.tensor(q,dtype=qdtype);li=torch.tensor(lv,dtype=ldtype);f=CPUFrame(len(records))
            before_q=packet(qi);before_l=packet(li)
            for i in range(26):
                a=original._build_local_token_metadata(len(q)-1,ninput,qi,li,is_noncausal=noncausal)
                b=f.call(len(q)-1,ninput,qi,li,is_noncausal=noncausal);equal(a,b);comparisons+=1
                assert packet(qi)==before_q and packet(li)==before_l
                # Original CP post-clamp is out-of-place; raw cache immutable.
                actual=q[-1];end=min(a[1],actual);start=min(a[0],end)
                _=b[4].clamp_max(end-start).to(torch.int32)
            records.append(dict(qsl=q,lengths=lv,input_tokens=ninput,world=world,rank=rank,noncausal=noncausal,q_dtype=str(qdtype),length_dtype=str(ldtype),output_sha256=digest(b)))
    # Optional-out callers are deliberately bypassed; preserve all mutations.
    group.world_size=8;group.rank_in_group=0
    q=torch.tensor([0,24],dtype=torch.int32);l=torch.tensor([100],dtype=torch.int32)
    aout=[torch.full((2,),-7,dtype=torch.int32),torch.full((1,),-7,dtype=torch.int32),torch.full((1,),-7,dtype=torch.int32)]
    bout=[v.clone() for v in aout]
    a=original._build_local_token_metadata(1,24,q,l,*aout);b=CPUFrame('out').call(1,24,q,l,*bout);equal(a,b)
    assert all(packet(x)==packet(y) for x,y in zip(aout,bout))
    # Demonstrate why object/version keys require the explicit immutable lease.
    x=torch.tensor([100],dtype=torch.int32);version=x._version;x.numpy()[0]=90
    numpy_version_unchanged=x._version==version;assert numpy_version_unchanged
    bench=[]
    cases=[('decode24-r0',[0,6,12,18,24],24,0),('decode24-r7',[0,6,12,18,24],24,7),('tail6-pad24',[0,6],24,0),('empty-pad24',[0],24,3),('prefill3328',[0,100,3328],3328,0),('prefill6540',[0,3200,6540],6540,7)]
    for name,qv,n,rank in cases:
        group.rank_in_group=rank;group.world_size=8
        lv=[32768+qv[i+1]-qv[i] for i in range(len(qv)-1)]
        q=torch.tensor(qv,dtype=torch.int32);l=torch.tensor(lv,dtype=torch.int32);nr=len(lv)
        def run(kind,frames):
            start=time.perf_counter_ns()
            for epoch in range(frames):
                if kind=='memo':f=CPUFrame(epoch)
                for _ in range(26):
                    if kind=='reference':v=original._build_local_token_metadata(nr,n,q,l)
                    else:v=f.call(nr,n,q,l)
            return (time.perf_counter_ns()-start)/1000/frames
        run('reference',2);run('memo',2);rounds=[]
        for j in range(15):
            order=['reference','memo'] if j%2==0 else ['memo','reference'];r={}
            for kind in order:r[kind+'_us_per_26call_frame']=run(kind,10)
            r['saved_us_per_frame']=r['reference_us_per_26call_frame']-r['memo_us_per_26call_frame'];rounds.append(r)
        bench.append(dict(case=name,qsl=qv,input_tokens=n,rank=rank,rounds=rounds,reference_median_us=median(r['reference_us_per_26call_frame'] for r in rounds),memo_median_us=median(r['memo_us_per_26call_frame'] for r in rounds),paired_saved_median_us=median(r['saved_us_per_frame'] for r in rounds)))
    after=guard()
    result=dict(status='PASS_ISOLATED_ORIGINAL_TORCH_CPU_HELPER_MEMO',pid=os.getpid(),torch_version=torch.__version__,python_version=sys.version,torch_threads=torch.get_num_threads(),CPU_cases=len(records),bitwise_comparisons=comparisons,bitwise_mismatches=0,optional_out_fallback_mutations_equal=True,numpy_mutation_keeps_version=numpy_version_unchanged,
                guards_before=before,guards_after=after,source_sha256=spec['helper_source_sha256'],torch_init_sha256=spec['torch_init_sha256'],case_records=records,benchmarks=bench,
                qualification='Original helper AST, actual installed Torch CPU, synthetic source-domain integer inputs and mocked TP geometry. Caller immutable-input lease is a prerequisite, not inferred from Tensorversion. Prototype is not installed in modelservice; perresource pins/deviceQSL/closures/graph/nativepackets untouched and not validated bythisprobe. CPUlocal cost only, not requestgain or NPU performance.',new_NPU_work=0,new_profile=0,model_loads=0,service_requests=0,service_actions=0)
    result['case_records_sha256']=sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
    (root/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],CPU_cases=len(records),comparisons=comparisons,bitwise_mismatches=0,torch_version=torch.__version__,benchmarks=[{k:v for k,v in r.items() if k!='rounds'} for r in bench],no_NPU_libraries=True)))
if __name__=='__main__':main()
