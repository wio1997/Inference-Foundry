"""Actual A3 HcPre statistics, lag independence, graph and workspace probe."""
from pathlib import Path
import datetime,hashlib,json,sys,time,traceback
R=Path('/task');O=R/'probes/fused_rms_g42';result={'status':'running','checks':{},'cases':[]}
def stamp(stage):
 result.update(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=stage)
 p=O/'result.tmp';p.write_text(json.dumps(result,indent=2)+'\n');p.replace(O/'result.json');print(stage,flush=True)
try:
 import torch,torch_npu
 from safetensors import safe_open
 torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.npu.set_device(0)
 from sglang.srt.hardware_backend.npu.utils import init_npu_backend
 init_npu_backend()
 from sglang.srt.models import deepseek_v4_mhc as reference
 assert reference._is_npu and reference._NPU_MHC_STATS_CHUNK_ROWS==4096
 assert hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest()=='cb64efc1c76453d94f80186144dee5e666fb148369b3c05173e36d823d29be15'
 import importlib.util
 source=R/'compat/runtime/g42/native/sglang/srt/models/deepseek_v4_mhc.py'
 spec=importlib.util.spec_from_file_location('sglang.srt.models.mhc_fused_rms_g42',source)
 candidate=importlib.util.module_from_spec(spec);sys.modules[spec.name]=candidate;spec.loader.exec_module(candidate)
 assert candidate._NPU_MHC_FUSED_RMS and candidate._NPU_MHC_STATS_CHUNK_ROWS==4096
 result['sources']={'reference':hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest(),'candidate':hashlib.sha256(source.read_bytes()).hexdigest()}
 model=Path('/model');cfg=json.loads((model/'config.json').read_text());cfg=cfg.get('text_config',cfg)
 index=json.loads((model/'quant_model_weights.safetensors.index.json').read_text())['weight_map']
 assert (cfg['hidden_size'],cfg['hc_mult'])==(5120,4)
 config=reference.HcConfig(4,cfg['hc_sinkhorn_iters'],cfg['hc_eps'],cfg['rms_norm_eps'],5120,True,False)
 result['config']=config._asdict();dev=torch.device('npu:0');rng=torch.Generator(device='cpu').manual_seed(1664001)
 def layer(i,kind):
  names=[f'layers.{i}.hc_{kind}_{suffix}' for suffix in ['fn','scale','base']];values=[]
  for name in names:
   with safe_open(model/index[name],framework='pt',device='cpu') as f:values.append(f.get_tensor(name).float().contiguous().to(dev))
  return reference.HcSubLayer(config,*values,norm=None),names
 def stats(hc,x):return candidate._mix_stats_impl(hc,x)
 def check(hc,x,label):
  torch.npu.synchronize();base=torch.npu.memory_allocated();torch.npu.reset_peak_memory_stats()
  a=reference._mix_stats_impl(hc,x);torch.npu.synchronize();old=torch.npu.max_memory_allocated()-base
  base=torch.npu.memory_allocated();torch.npu.reset_peak_memory_stats();b=stats(hc,x);torch.npu.synchronize();new=torch.npu.max_memory_allocated()-base
  errors=[float((u-v).abs().max()) if u.numel() else 0. for u,v in zip(a,b)]
  row={'label':label,'rows':x.shape[0],'FP32_shapes':[list(t.shape) for t in b],
       'max_abs_errors':errors,'bitwise_exact':all(torch.equal(u,v) for u,v in zip(a,b)),
       'reference_peak_allocated_delta_bytes':old,'fused_peak_allocated_delta_bytes':new,
       'atol_rtol':[2e-5,2e-5]}
  result['cases'].append(row);stamp(label+'_observed')
  assert all(t.dtype==torch.float32 and t.device.type=='npu' and torch.isfinite(t).all() for t in b)
  assert [tuple(t.shape) for t in b]==[(x.shape[0],4),(x.shape[0],4),(x.shape[0],4,4)]
  assert all(torch.allclose(u,v,atol=2e-5,rtol=2e-5) for u,v in zip(a,b)),row
  return b
 with torch.inference_mode():
  hc,names=layer(0,'attn');result['first_attention_weight_names']=names
  for n in [0,1,6,24,4096,4097,8192,16384]:
   stamp('first_attn_rows_'+str(n));x=torch.randn((n,4,5120),generator=rng).bfloat16().to(dev)
   b=check(hc,x,'first_attn_'+str(n));del b,x;torch.npu.empty_cache()
  hc,names=layer(39,'ffn');result['last_ffn_weight_names']=names
  x=torch.randn((8192,4,5120),generator=rng).bfloat16().to(dev);check(hc,x,'last_ffn_8192');del x;torch.npu.empty_cache()
  for layer_id in range(40):
   for kind in ['attn','ffn']:
    hc,names=layer(layer_id,kind);x=torch.randn((32,4,5120),generator=rng).bfloat16().to(dev)
    check(hc,x,f'layer{layer_id}_{kind}_32');del x
  # Tiny/zero activation cases catch epsilon and statistics dtype mistakes.
  hc,_=layer(0,'attn')
  for value in [0.,1e-5,1e-2,10.]:
   x=torch.full((6,4,5120),value,dtype=torch.bfloat16,device=dev)
   check(hc,x,'constant_'+str(value))
  result['checks']['actual_checkpoint_first_attn_last_ffn_coefficients_and_row_boundaries']=True
  hc,_=layer(0,'attn');x=torch.randn((24,4,5120),generator=rng).bfloat16().to(dev)
  result['checks']['FP32_projection_and_original_lagged_combine_source_preserved']=True
  for _ in range(3):stats(hc,x)
  torch.npu.synchronize();g=torch.npu.NPUGraph()
  with torch.npu.graph(g):captured=stats(hc,x)
  for scale,shift in [(.91,.03),(.83,-.05)]:
   x.copy_(x*scale+shift);eager=stats(hc,x);g.replay();torch.npu.synchronize()
   assert all(torch.equal(a,b) for a,b in zip(eager,captured))
  result['checks']['changed_input_graph_eager_stats_bitwise']=True
  # Component timing is diagnostic; no end-to-end benchmark claim.
  timing={}
  for label,fn in [('reference',lambda:reference._mix_stats_impl(hc,x)),('fused_RMS_FP32_projection',lambda:stats(hc,x))]:
   for _ in range(5):fn()
   torch.npu.synchronize();samples=[]
   for _ in range(3):
    t=time.perf_counter()
    for _ in range(30):fn()
    torch.npu.synchronize();samples.append((time.perf_counter()-t)/30)
   timing[label]=samples
  result['diagnostic_seconds_per_call_rows24']=timing
 result['status']='passed_NPU_fused_RMS_FP32_projection_statistics_contract'
 result['scope']='Actual checkpoint FP32 stats, previous-pre independence, graph replay and allocator peak; no full-model quality, fit or AISBench gain claim'
 stamp('terminal_passed')
except BaseException:
 result['status']='failed';result['error']=traceback.format_exc();stamp('terminal_failed');raise
