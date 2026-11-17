"""Native 910B SFA ratio support + constant-value invariant; not full attention/model correctness."""
from pathlib import Path
import datetime,hashlib,json,os,traceback
import torch
import torch_npu
rank=int(os.environ['LOCAL_RANK']);os.sched_setaffinity(0,{168+rank});torch.set_num_threads(1);torch.npu.set_device(rank)
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'evidence';OUT.mkdir(exist_ok=True)
record={'rank':rank,'pid':os.getpid(),'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':'initializing','cases':[],'scope':'Native fixed-image SFA metadata/execute/graph acceptance for C1/C2; constant-value invariant only, not causal masks, full model or PD'}
def save():
 p=OUT/f'rank{rank}.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(record,indent=2));q.replace(p)
def sha(t):return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
try:
 save()
 from sglang.srt.hardware_backend.npu.utils import init_npu_backend
 init_npu_backend()
 record['custom_target_metadata_registered']=hasattr(torch.ops.custom,'npu_sparse_attn_sharedkv_metadata')
 from sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend import _sparse_attn_ops
 metadata_op,attention_op=_sparse_attn_ops(is_dspark=True)
 record.update(native_API='Fixed image DSpark host-metadata + builtin CANN sparse_attn_sharedkv; custom namespace unavailable in G89; no production backend acceptance',device=torch.npu.get_device_name(rank),native_head_contract='fixed model _kernel_num_heads pads local TP8 heads to 64 on NPU; G96 unpadded8 fixture rejected',metadata_schema=str(metadata_op._schemas),attention_schema=str(attention_op._schemas));save()
 # C128 is the existing fixed-image path without indexer top-k. C1/C2 are the new checkpoint requirements.
 for ratio in [128,1,2]:
  case={'ratio':ratio,'phase':'metadata'};record['cases'].append(case);save()
  seq=256;heads=64;dim=512;tokens=1;page=128;cmp_page=max(16,page//ratio)
  q=torch.zeros((tokens,heads,dim),dtype=torch.bfloat16,device=f'npu:{rank}')
  ori=torch.full((2,page,1,dim),.125,dtype=torch.bfloat16,device=f'npu:{rank}')
  cmp=torch.full((2,cmp_page,1,dim),.125,dtype=torch.bfloat16,device=f'npu:{rank}')
  cu=torch.tensor([0,1],dtype=torch.int32,device=f'npu:{rank}');lens=torch.tensor([seq],dtype=torch.int32,device=f'npu:{rank}')
  ori_table=torch.tensor([[0,1]],dtype=torch.int32,device=f'npu:{rank}');cmp_table=torch.tensor([[0,1]],dtype=torch.int32,device=f'npu:{rank}')
  common=dict(cu_seqlens_q=cu,seqused_kv=lens,cmp_ratio=ratio,ori_mask_mode=4,cmp_mask_mode=3,ori_win_left=127,ori_win_right=0,layout_q='TND',layout_kv='PA_ND')
  metadata_common=dict(common);metadata_common['cu_seqlens_q']=cu.cpu();metadata_common['seqused_kv']=lens.cpu()
  metadata=metadata_op(batch_size=1,num_heads_q=heads,num_heads_kv=1,head_dim=dim,has_ori_kv=True,has_cmp_kv=True,**metadata_common)
  case.update(phase='execute',metadata_device=str(metadata.device),sinks_contract='native FP32 head vector; -inf sink leaves constant-value invariant unchanged');save()
  kwargs=dict(common,ori_kv=ori,ori_block_table=ori_table,sinks=torch.full((heads,),float('-inf'),dtype=torch.float32,device=f'npu:{rank}'),metadata=metadata,softmax_scale=dim**-.5,cmp_kv=cmp,cmp_block_table=cmp_table,cmp_sparse_indices=None)
  def execute():return attention_op(q,**kwargs)[0]
  for _ in range(2):out=execute();torch.npu.synchronize()
  expected=torch.full(out.shape,.125,dtype=torch.bfloat16);actual=out.cpu();torch.testing.assert_close(actual,expected,rtol=0,atol=.001)
  case.update(phase='graph',shape=list(out.shape),constant_value_max_abs_error=float((actual.float()-expected.float()).abs().max()));save()
  graph=torch.npu.NPUGraph()
  with torch.npu.graph(graph,pool=torch.npu.graph_pool_handle(),auto_dispatch_capture=True):captured=execute()
  hashes=[]
  for value in [.25,.375,.5]:
   ori.fill_(value);cmp.fill_(value);graph.replay();torch.npu.synchronize();actual=captured.cpu();torch.testing.assert_close(actual,torch.full(actual.shape,value,dtype=torch.bfloat16),rtol=0,atol=.001);hashes.append(sha(actual))
  assert len(set(hashes))==3
  case.update(phase='passed',graph_captures=1,changed_input_replays=3,hashes=hashes);save();del graph,q,ori,cmp,out,captured;torch.npu.synchronize();torch.npu.empty_cache()
 record['phase']='passed';save()
except BaseException as exc:
 record.update(phase='failed',error=repr(exc),traceback=traceback.format_exc());save();raise
