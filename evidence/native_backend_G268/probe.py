"""Native V4.1 backend/pool/metadata and NPU attention integration gate.

Uses checkpoint ModelConfig, a runner interface fixture, supplied KV selections
and the real native metadata constructors. Not a loaded model or network PD.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,math,os,traceback
import torch,torch_npu
rank=int(os.environ['LOCAL_RANK']);os.sched_setaffinity(0,{168+rank});torch.set_num_threads(1);torch.set_grad_enabled(False);torch.npu.set_device(rank)
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'evidence';OUT.mkdir(exist_ok=True)
record={'rank':rank,'pid':os.getpid(),'phase':'initializing','cases':[],'scope':'Native checkpoint ModelConfig + complete backend constructor + native metadata + actual NPU BF16 attention/pool integration with supplied selection; no full model weights, source/indexer selection, PD, capacity or performance acceptance'}
def save():
 p=OUT/f'rank{rank}.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(record,indent=2));q.replace(p)
def oracle(q,ori,cmp,oi,ci,sink):
 result=[]
 for row in range(q.shape[0]):
  kv=torch.cat([ori[oi[row][oi[row]>=0].long()],cmp[ci[row][ci[row]>=0].long()]],0).double()
  logits=q[row].double()@kv.T/math.sqrt(512);weight=torch.softmax(torch.cat([logits,sink.double()[:,None]],1),1)[:,:kv.shape[0]];result.append((weight@kv).float())
 return torch.stack(result)
try:
 save()
 from sglang.srt.server_args import prepare_server_args
 from sglang.srt.runtime_context import create_context,use_context,SpawnRanks
 from sglang.srt.configs.model_config import ModelConfig
 from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
 from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
 from sglang.srt.model_executor.forward_mode import ForwardMode
 import sglang
 package=Path(sglang.__file__).resolve().parent.parent
 for name,expected in json.loads((ROOT/'native_pool_sha256.json').read_text()).items():assert hashlib.sha256((package/name).read_bytes()).hexdigest()==expected
 from npu_backend import install_native_metadata_scope,install_bf16_request_window,DeepseekV41NPUBackend
 from lowratio_pool import LowRatioNPUTokenToKVPool
 install_native_metadata_scope(ROOT);install_bf16_request_window()
 candidate=json.loads((ROOT/'candidate_d.json').read_text());argv=candidate['argv'][1:];argv[argv.index('--disaggregation-mode')+1]='null';argv.remove('--disaggregation-decode-enable-radix-cache');argv+=['--dsa-topk-backend','torch','--speculative-dsa-topk-backend','torch']
 args=prepare_server_args(argv);context=create_context(args,role='model_runner',ranks=SpawnRanks(world_rank=rank,gpu_id=rank));cfg=json.loads(Path('/models/ds41/config.json').read_text())['text_config']
 with use_context(context):
  model=ModelConfig.from_server_args(args);record['model_config']={'class':type(model).__name__,'hf_model_type':model.hf_config.model_type,'context':model.context_len,'head_dim':model.head_dim,'v_head_dim':model.v_head_dim,'quantization':model.quantization};save()
  kwargs=dict(max_num_reqs=16,num_req_slots=17,swa_size=8192,c4_size=0,c128_size=0,c4_state_pool_size=0,c128_state_pool_size=0,page_size=128,swa_page_size=128,dtype=torch.bfloat16,c4_state_dtype=torch.float32,c128_state_dtype=torch.float32,qk_nope_head_dim=448,qk_rope_head_dim=64,indexer_head_dim=128,layer_num=40,device=f'npu:{rank}',enable_memory_saver=False,compression_ratios=cfg['compress_ratios'],start_layer=0,end_layer=40,sliding_window=128,online_mtp_max_draft_tokens=6,kv_source_layers=cfg['kv_source_layer_ids'],full_size=8192)
  pool=LowRatioNPUTokenToKVPool(**kwargs);pool.register_mapping(torch.arange(8320,dtype=torch.int64,device=f'npu:{rank}'))
  req=ReqToTokenPool(16,263296,f'npu:{rank}',False)
  for slot in range(1,17):req.req_to_token[slot,:512]=torch.arange(128+(slot-1)*512,128+slot*512,dtype=torch.int32,device=f'npu:{rank}')
  runner=SimpleNamespace(device=f'npu:{rank}',dtype=torch.bfloat16,model_config=model,page_size=128,req_to_token_pool=req,token_to_kv_pool=pool,hisparse_coordinator=None,spec_algorithm=SpeculativeAlgorithm.DSPARK,is_draft_worker=False)
  backend=DeepseekV41NPUBackend(runner);record.update(phase='backend_constructed',backend=type(backend).__name__,native_page_size=backend.page_size,request_window=pool.request_window is not None,low_ratios=list(backend.low_ratios));save()
  gen=torch.Generator().manual_seed(268)
  for rows_per_req in [1,6]:
   rows=16*rows_per_req;requests=torch.arange(1,17,device=f'npu:{rank}');seq=torch.full((16,),512,dtype=torch.int32,device=f'npu:{rank}');extend=torch.full((16,),rows_per_req,dtype=torch.int32,device=f'npu:{rank}');outloc=torch.cat([req.req_to_token[s,512-rows_per_req:512] for s in range(1,17)]).long()
   metadata=backend.init_forward_metadata_prefill(512,requests,seq,[512]*16,outloc,rows,extend,[rows_per_req]*16,extend_start_loc=torch.arange(0,rows,rows_per_req,dtype=torch.int32,device=f'npu:{rank}'));backend.forward_metadata=metadata
   core=metadata.core_metadata;expected_pos=torch.arange(512-rows_per_req,512).repeat(16);assert torch.equal(core.positions_casual.cpu(),expected_pos)
   fb=SimpleNamespace(forward_mode=ForwardMode.EXTEND,batch_size=16,out_cache_loc=outloc)
   backend.get_swa_out_cache_loc(fb)
   for ratio,layer_id in [(0,0),(1,20),(2,2)]:
    case={'rows':rows,'ratio':ratio,'layer_id':layer_id,'phase':'running'};record['cases'].append(case);save()
    if pool.request_window is not None:
     window=pool.request_window;buffer=window.state.kv_buffer[layer_id];window.prepared=None;layout=window.layout
     window.tags[layer_id,layout.history_src]=layout.history_pos
    else:buffer=pool.swa_kv_pool.kv_buffer[layer_id]
    value=(torch.randn(buffer.shape,generator=gen)*.2).bfloat16();buffer.copy_(value)
    key=(torch.randn((rows,512),generator=gen)*.2).bfloat16().to(f'npu:{rank}');backend.store_cache(layer_id,key,fb)
    oi=core.swa_page_indices.reshape(rows,-1).int().contiguous()
    if ratio:
     buf=pool.layer_mapping[layer_id].compress_kv_pool.kv_buffer[pool.layer_mapping[layer_id].compress_layer_id];buf.copy_((torch.randn(buf.shape,generator=gen)*.2).bfloat16())
     positions=torch.arange(512,device=f'npu:{rank}')[None,:].expand(rows,-1);reqrows=requests.repeat_interleave(rows_per_req);slots=req.req_to_token[reqrows[:,None],(positions*ratio).clamp_max(511)]//ratio
     ci=torch.where(positions<(core.positions_casual[:,None]+1)//ratio,slots,-1).int();core.sparse_page_indices(ratio).copy_(ci)
    else:ci=torch.full((rows,512),-1,dtype=torch.int32,device=f'npu:{rank}')
    q_cpu=(torch.randn((rows,8,512),generator=gen)*.2).bfloat16();q=q_cpu.to(f'npu:{rank}');sink=torch.linspace(-2,2,8,dtype=torch.float32,device=f'npu:{rank}');layer=SimpleNamespace(layer_id=layer_id,scaling=512**-.5,v_head_dim=512)
    def call():return backend.forward(q,None,None,layer,fb,compress_ratio=ratio,save_kv_cache=False,attn_sink=sink)
    actual=call();torch.npu.synchronize();original=backend._view(pool.get_swa_raw_buffer(layer_id)).cpu();compressed=backend._empty_kv.cpu() if not ratio else backend._view(buf).cpu();oi_cpu=oi.cpu();ci_cpu=ci.cpu();expected=oracle(q_cpu,original,compressed,oi_cpu,ci_cpu,sink.cpu());torch.testing.assert_close(actual.cpu().float(),expected,rtol=.02,atol=.003)
    if pool.request_window is not None:pool.request_window.prepared=None
    graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph,pool=torch.npu.graph_pool_handle(),auto_dispatch_capture=True):captured=call()
    errors=[]
    for step in range(3):
     new=(q_cpu.float()+.03125*(step+1)).bfloat16();q.copy_(new);graph.replay();torch.npu.synchronize();expected=oracle(new,original,compressed,oi_cpu,ci_cpu,sink.cpu());torch.testing.assert_close(captured.cpu().float(),expected,rtol=.02,atol=.003);errors.append(float((captured.cpu().float()-expected).abs().max()))
    case.update(phase='passed',graph_captures=1,changed_input_replays=3,max_abs_error=max(errors),independent_FP64_all_rows=True,metadata_positions_exact=True);save()
  record['phase']='passed';save()
except BaseException as exc:record.update(phase='failed',error=repr(exc),traceback=traceback.format_exc());save();raise
