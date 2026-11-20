"""One-device integer combiner diagnostic; exact independent CPU reference."""
from pathlib import Path
import bisect, hashlib, importlib.util, json, time
import torch
import torch_npu

ROOT=Path('/work')
INSTALLED=Path('/sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsv4/sparse_prefill_kernels.py')

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

helper=load('guard_definition',ROOT/'compile_c0_topk_guard_g1502.py')
candidate_path=ROOT/'sparse_prefill_kernels_guarded.py'
candidate_sha=helper.define_candidate(candidate_path,INSTALLED.read_bytes())
assert candidate_sha=='b3e90d264767cde04f32e3901b8b04d0176a55ceb9e318bf36ff129ac17e8974'
candidate=load('guarded_combiner',candidate_path)
baseline=load('original_combiner',INSTALLED)
torch.npu.set_device(0)
results=[]

def one_case(requests,tokens,explicit,topk,ratio,changed):
    lengths=[tokens//requests+(r<tokens%requests) for r in range(requests)]
    starts=[37]
    for n in lengths:starts.append(starts[-1]+n)
    positions=[]
    seq=[];gather=[]
    for r,n in enumerate(lengths):
        first=256+changed*3+r
        positions.extend(range(first,first+n))
        seq.append(first+n);gather.append(n+127)
    # One unowned trailing query checks complete output overwrite and odd tiles.
    count=tokens+1
    positions.append(0)
    compressed=[r*100000+13 for r in range(requests)]
    swabase=[r*100000+20000 for r in range(requests)]
    k=max(topk,1)
    # Nonzero controls have padded physical rows so this test does not execute
    # the known original masked-copy hazard. Their semantics are still exact.
    physical_width=1 if topk==0 else 1024
    storage=torch.full((count,physical_width),-1,dtype=torch.int32)
    vals=torch.arange(count*k,dtype=torch.int32).reshape(count,k)%2048
    vals[(vals+changed)%11==0]=-1
    storage[:,:k]=vals
    top=storage[:,:k]
    swa_storage=torch.full((count,2048),-1,dtype=torch.int32)
    swa=swa_storage[:,1024:1152]
    sl=[]
    for t in range(count):
        n=0 if t==tokens else 1+(t+changed)%128
        sl.append(n)
        if n:swa[t,:n]=torch.arange(n,dtype=torch.int32)+t%13
        if n>3:swa[t,2]=-1
    width=((topk+128+127)//128)*128
    expected=torch.full((count,width),-1,dtype=torch.int32)
    elens=torch.zeros(count,dtype=torch.int32)
    for t in range(tokens):
        r=bisect.bisect_right([s-37 for s in starts],t)-1
        p=positions[t]
        nt=min((p+1)//ratio,topk)
        ns=sl[t] if explicit else min(p+1,128)
        for j in range(nt):
            v=int(top[t,j]);expected[t,j]=v+compressed[r] if v>=0 else -1
        for j in range(ns):
            if explicit:
                v=int(swa[t,j]);v=v+swabase[r] if v>=0 else -1
            else:
                v=swabase[r]+p-ns+1-(seq[r]-gather[r])+j
            expected[t,nt+j]=v
        elens[t]=nt+ns
    cpu={'topk_indices':top,'query_start_loc':torch.tensor(starts,dtype=torch.int32),
         'query_pos':torch.tensor(positions,dtype=torch.int32),'seq_lens':torch.tensor(seq,dtype=torch.int32),
         'gather_lens':torch.tensor(gather,dtype=torch.int32),'compressed_base':torch.tensor(compressed,dtype=torch.int32),
         'swa_base':torch.tensor(swabase,dtype=torch.int32)}
    # Preserve row stride on the NPU, including the original one-column C0 case.
    device_storage=storage.npu()
    args={name:value.npu() for name,value in cpu.items() if name!='topk_indices'}
    args['topk_indices']=device_storage[:,:k]
    args.update(window_size=128,compress_ratio=ratio,topk=topk)
    if explicit:args.update(swa_indices=swa_storage.npu()[:,1024:1152],swa_lengths=torch.tensor(sl,dtype=torch.int32).npu())
    idx=torch.full((count,width),777,dtype=torch.int32,device='npu')
    lens=torch.full((count,),777,dtype=torch.int32,device='npu')
    args.update(out_indices=idx,out_lens=lens)
    out,ol=candidate.combine_topk_swa_indices(**args)
    torch.npu.synchronize()
    actual=out.cpu();actual_lens=ol.cpu()
    if not torch.equal(actual,expected):
        diff=torch.nonzero(actual!=expected)
        raise AssertionError({'indices_mismatch':diff[:12].tolist(),'requests':requests,'tokens':tokens,'explicit':explicit,'topk':topk,'changed':changed})
    assert torch.equal(actual_lens,elens),('length_mismatch',requests,tokens,explicit,topk,changed)
    if topk:
        b,bl=baseline.combine_topk_swa_indices(**args)
        torch.npu.synchronize()
        assert torch.equal(b.cpu(),expected) and torch.equal(bl.cpu(),elens),'nonzero baseline control'
    row={'requests':requests,'tokens':tokens,'explicit_swa':explicit,'topk':topk,'compress_ratio':ratio,
         'changed_input':changed,'row_stride':device_storage.stride(0),'output_rows':count,'exact_indices':True,'exact_lengths':True}
    results.append(row)
    print(json.dumps(row),flush=True)
    (ROOT/'progress.json').write_text(json.dumps({'cases':results},indent=2)+'\n')

begin=time.monotonic()
for requests,tokens in [(1,1),(1,17),(2,17),(3,129),(8,1024),(16,8192)]:
    for explicit in (False,True):
        for changed in (0,1):one_case(requests,tokens,explicit,0,1,changed)
for topk,ratio in [(32,128),(512,2)]:
    for explicit in (False,True):one_case(3,129,explicit,topk,ratio,1)
report={'scope':'Single 910B1 NPU, integer combiner only; no full-model, quality, PD, graph, or performance acceptance',
        'candidate_sha256':candidate_sha,'installed_sha256':hashlib.sha256(INSTALLED.read_bytes()).hexdigest(),
        'cases':results,'passed':len(results),'elapsed_s':time.monotonic()-begin,'physical_device':0,
        'positive_controls_use_padded_rows':True}
(ROOT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
print('G1503_COMBINER_RESULT '+json.dumps(report),flush=True)
