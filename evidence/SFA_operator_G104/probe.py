"""Fixed native target/draft SFA contracts; weak constant-value invariant, not mask correctness."""
from pathlib import Path
import datetime,hashlib,json,os,traceback
import torch,torch_npu

rank=int(os.environ['LOCAL_RANK']);os.sched_setaffinity(0,{168+rank})
torch.set_num_threads(1);torch.npu.set_device(rank)
OUT=Path(__file__).resolve().parent/'evidence';OUT.mkdir(exist_ok=True)
record={'rank':rank,'pid':os.getpid(),'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'phase':'initializing','cases':[],'scope':'Independent fixed-binary metadata/constant-value/changed-input graph contracts; no full attention mask, model or PD acceptance'}

def save():
    p=OUT/f'rank{rank}.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(record,indent=2));q.replace(p)

def sha(t):return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

try:
    save()
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend
    init_npu_backend()
    from sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend import _sparse_attn_ops
    record['device']=torch.npu.get_device_name(rank)
    record['custom_target_metadata_registered']=hasattr(torch.ops.custom,'npu_sparse_attn_sharedkv_metadata')
    save()
    # First native C128 control, then C4 indexed control, then actual C1/C2 contracts.
    for draft in [False,True]:
        metadata_op,attention_op=_sparse_attn_ops(is_dspark=draft)
        for ratio in [128,4,1,2]:
            case={'API':'draft_builtin_host' if draft else 'target_custom_device','ratio':ratio,
                  'phase':'metadata','cmp_topk':0 if ratio==128 else 512,
                  'metadata_schema':str(metadata_op._schemas),'attention_schema':str(attention_op._schemas)}
            record['cases'].append(case);save()
            device=f'npu:{rank}';heads=64;dim=512;seq=256;page=128;cmp_page=max(16,page//ratio)
            q=torch.zeros((1,heads,dim),dtype=torch.bfloat16,device=device)
            ori=torch.full((2,page,1,dim),.125,dtype=torch.bfloat16,device=device)
            cmp=torch.full((max(2,(seq//ratio+cmp_page-1)//cmp_page),cmp_page,1,dim),.125,dtype=torch.bfloat16,device=device)
            cu=torch.tensor([0,1],dtype=torch.int32,device=device)
            lens=torch.tensor([seq],dtype=torch.int32,device=device)
            ori_table=torch.tensor([[0,1]],dtype=torch.int32,device=device)
            cmp_table=torch.arange(cmp.shape[0],dtype=torch.int32,device=device).view(1,-1)
            common=dict(cu_seqlens_q=cu,seqused_kv=lens,cmp_ratio=ratio,ori_mask_mode=4,cmp_mask_mode=3,
                        ori_win_left=127,ori_win_right=0,layout_q='TND',layout_kv='PA_ND')
            metadata_common=dict(common)
            if draft:
                metadata_common.update(cu_seqlens_q=cu.cpu(),seqused_kv=lens.cpu())
            try:
                metadata=metadata_op(batch_size=1,num_heads_q=heads,num_heads_kv=1,head_dim=dim,
                                     has_ori_kv=True,has_cmp_kv=True,cmp_topk=case['cmp_topk'],**metadata_common)
            except RuntimeError as exc:
                # Metadata scheduling rejects params before attention launch. Record the entire matrix.
                case.update(phase='metadata_rejected',error=repr(exc),traceback=traceback.format_exc())
                save();torch.npu.synchronize();continue
            case.update(phase='execute',metadata_device=str(metadata.device));save()
            indices=None
            if ratio!=128:
                # Model indexer selects up to512 causal compressed IDs; unused slots use native -1.
                host_indices=torch.full((1,1,512),-1,dtype=torch.int32)
                host_indices[0,0,:seq//ratio]=torch.arange(seq//ratio,dtype=torch.int32)
                indices=host_indices.to(device)
            kwargs=dict(common,ori_kv=ori,ori_block_table=ori_table,
                        sinks=torch.full((heads,),float('-inf'),dtype=torch.float32,device=device),
                        metadata=metadata,softmax_scale=dim**-.5,cmp_kv=cmp,
                        cmp_block_table=cmp_table,cmp_sparse_indices=indices)
            def execute():return attention_op(q,**kwargs)[0]
            # An attention execution/device error aborts this rank; no reuse of a faulted device.
            for _ in range(2):out=execute();torch.npu.synchronize()
            actual=out.cpu();torch.testing.assert_close(actual,torch.full(actual.shape,.125,dtype=torch.bfloat16),rtol=0,atol=.001)
            case.update(phase='graph',shape=list(actual.shape),constant_value_max_abs_error=float((actual.float()-.125).abs().max()));save()
            graph=torch.npu.NPUGraph()
            with torch.npu.graph(graph,pool=torch.npu.graph_pool_handle(),auto_dispatch_capture=True):captured=execute()
            hashes=[]
            for value in [.25,.375,.5]:
                ori.fill_(value);cmp.fill_(value);graph.replay();torch.npu.synchronize();actual=captured.cpu()
                torch.testing.assert_close(actual,torch.full(actual.shape,value,dtype=torch.bfloat16),rtol=0,atol=.001);hashes.append(sha(actual))
            assert len(set(hashes))==3
            case.update(phase='passed',graph_captures=1,changed_input_replays=3,hashes=hashes);save()
            del graph,q,ori,cmp,out,captured;torch.npu.synchronize();torch.npu.empty_cache()
    record['phase']='contracts_measured' if any(x['phase']=='metadata_rejected' for x in record['cases']) else 'passed';save()
except BaseException as exc:
    record.update(phase='failed',error=repr(exc),traceback=traceback.format_exc());save();raise
