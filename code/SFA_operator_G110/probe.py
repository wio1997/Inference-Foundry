"""Probe exact SWA/compressed partition softmax composition using fixed CANN operators."""
from pathlib import Path
import datetime,hashlib,json,math,os,traceback
import torch,torch_npu

rank=int(os.environ['LOCAL_RANK']);os.sched_setaffinity(0,{168+rank})
torch.set_num_threads(1);torch.npu.set_device(rank)
OUT=Path(__file__).resolve().parent/'evidence';OUT.mkdir(exist_ok=True)
record={'rank':rank,'pid':os.getpid(),'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'phase':'initializing','cases':[],'scope':'Constant Q/KV partition-count, finite sink and LSE composition primitive; not arbitrary Q, causal-mask, model or PD correctness'}
def save():
    p=OUT/f'rank{rank}.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(record,indent=2));q.replace(p)
def sha(t):return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
try:
    save()
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend
    init_npu_backend()
    from sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend import _sparse_attn_ops
    meta_op,attn_op=_sparse_attn_ops(is_dspark=True)
    record.update(device=torch.npu.get_device_name(rank),metadata_schema=str(meta_op._schemas),attention_schema=str(attn_op._schemas));save()
    for ratio in [1,2]:
        c={'model_compression_ratio':ratio,'phase':'metadata'};record['cases'].append(c);save()
        device=f'npu:{rank}';heads=64;dim=512;seq=256;cmp_count=seq//ratio
        q=torch.zeros((1,heads,dim),dtype=torch.bfloat16,device=device)
        ori=torch.full((2,128,1,dim),.125,dtype=torch.bfloat16,device=device)
        # A view with 32-row pages for the native C4 kernel; original C1/C2 latent values unchanged.
        cmp=torch.full(((cmp_count+31)//32,32,1,dim),.125,dtype=torch.bfloat16,device=device)
        cu_cpu=torch.tensor([0,1],dtype=torch.int32);cu=cu_cpu.to(device)
        lens_cpu=torch.tensor([seq],dtype=torch.int32);lens=lens_cpu.to(device)
        # Compressed-only explicit indices contain every legal latent and -1 padding.
        ids=torch.full((1,1,512),-1,dtype=torch.int32);ids[0,0,:cmp_count]=torch.arange(cmp_count,dtype=torch.int32);ids=ids.to(device)
        tables=torch.tensor([[0,1]],dtype=torch.int32,device=device)
        ctables=torch.arange(cmp.shape[0],dtype=torch.int32,device=device).view(1,-1)
        sink=torch.full((heads,),math.log(128),dtype=torch.float32,device=device)
        no_sink=torch.full((heads,),float('-inf'),dtype=torch.float32,device=device)
        common=dict(num_heads_q=heads,num_heads_kv=1,head_dim=dim,layout_q='TND',layout_kv='PA_ND',
                    cu_seqlens_q=cu_cpu,batch_size=1,ori_mask_mode=4,cmp_mask_mode=3,ori_win_left=127,ori_win_right=0)
        smeta=meta_op(**common,seqused_kv=lens_cpu,has_ori_kv=True,has_cmp_kv=False,cmp_ratio=1,cmp_topk=0)
        # Scheduler coordinates count existing compressed rows; Q remains in the true model RoPE basis.
        # This one-query control has no relative query-position ambiguity.
        clens_cpu=torch.tensor([cmp_count*4],dtype=torch.int32);clens=clens_cpu.to(device)
        cmeta=meta_op(**common,seqused_kv=clens_cpu,has_ori_kv=False,has_cmp_kv=True,cmp_ratio=4,cmp_topk=512)
        c.update(phase='execute',metadata_devices=[str(smeta.device),str(cmeta.device)]);save()
        runtime=dict(cu_seqlens_q=cu,ori_mask_mode=4,cmp_mask_mode=3,ori_win_left=127,ori_win_right=0,
                     layout_q='TND',layout_kv='PA_ND',softmax_scale=dim**-.5,return_softmax_lse=True)
        def execute():
            a,la=attn_op(q,**runtime,seqused_kv=lens,ori_kv=ori,ori_block_table=tables,
                         cmp_kv=None,cmp_block_table=None,sinks=sink,metadata=smeta,cmp_ratio=1)
            b,lb=attn_op(q,**runtime,seqused_kv=clens,ori_kv=None,ori_block_table=None,
                         cmp_kv=cmp,cmp_block_table=ctables,cmp_sparse_indices=ids,
                         sinks=no_sink,metadata=cmeta,cmp_ratio=4)
            return a,la,b,lb
        for _ in range(2):a,la,b,lb=execute();torch.npu.synchronize()
        c.update(output_shapes=[list(x.shape) for x in [a,la,b,lb]],LSE_dtype=[str(la.dtype),str(lb.dtype)]);save()
        # Finite sink belongs only to SWA partition; no sink is counted twice.
        torch.testing.assert_close(a.cpu(),torch.full(a.shape,.0625,dtype=torch.bfloat16),rtol=0,atol=.001)
        torch.testing.assert_close(b.cpu(),torch.full(b.shape,.125,dtype=torch.bfloat16),rtol=0,atol=.001)
        torch.testing.assert_close(la.cpu().float(),torch.full(la.shape,math.log(256)),rtol=0,atol=.003)
        torch.testing.assert_close(lb.cpu().float(),torch.full(lb.shape,math.log(cmp_count)),rtol=0,atol=.003)
        # All fixtures use one query: reduce shape conventions explicitly and preserve64 heads.
        def merged():
            a,la,b,lb=execute();la=la.reshape(1,heads,1).float();lb=lb.reshape(1,heads,1).float()
            total=torch.logaddexp(la,lb)
            return (torch.exp(la-total)*a.float()+torch.exp(lb-total)*b.float()).to(torch.bfloat16)
        expected=(128+cmp_count)*.125/(128+128+cmp_count)
        actual=merged();torch.npu.synchronize();torch.testing.assert_close(actual.cpu(),torch.full(actual.shape,expected,dtype=torch.bfloat16),rtol=0,atol=.001)
        c.update(phase='graph',composed_expected=expected,actual_max_abs_error=float((actual.float().cpu()-expected).abs().max()));save()
        graph=torch.npu.NPUGraph()
        with torch.npu.graph(graph,pool=torch.npu.graph_pool_handle(),auto_dispatch_capture=True):captured=merged()
        hashes=[]
        for value in [.25,.375,.5]:
            ori.fill_(value);cmp.fill_(value);graph.replay();torch.npu.synchronize();expected=(128+cmp_count)*value/(256+cmp_count)
            torch.testing.assert_close(captured.cpu(),torch.full(captured.shape,expected,dtype=torch.bfloat16),rtol=0,atol=.002);hashes.append(sha(captured.cpu()))
        assert len(set(hashes))==3
        c.update(phase='passed',graph_captures=1,changed_input_replays=3,hashes=hashes);save()
        del graph,q,ori,cmp,actual,captured;torch.npu.synchronize();torch.npu.empty_cache()
    record['phase']='passed';save()
except BaseException as exc:
    record.update(phase='failed',error=repr(exc),traceback=traceback.format_exc());save();raise
